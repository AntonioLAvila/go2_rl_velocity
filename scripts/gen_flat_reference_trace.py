#!/usr/bin/env python3
"""Generate the reference trace that test_flat_policy_parity replays.

    REPO=/home/antonio/GitHub/quadruped_walking
    $REPO/.venv/bin/python scripts/gen_flat_reference_trace.py --repo $REPO

Run this on a workstation that has the training repo and its venv -- never on the robot.
Commit the resulting CSV; it is the contract between the trained policy and this package.

Why it exists
-------------
For the rugged policy this file's sibling, gen_reference_trace.py, exists mainly to pin
down a stacked history whose layout fails silently. The flat task stacks no history, so
there is less to get wrong -- but there is *more* riding on the test, because the flat
ONNX exports carry no metadata at all. mjlab's ``get_base_metadata`` starts with
``env.action_manager.get_term("joint_pos")``; the flat task's action term is named
``joint_effort``, so the lookup raises, the runner's ``save()`` swallows it, and the file
is written with an empty custom metadata map. FlatPolicyRunner can therefore only check
the graph's input and output width. Joint order, torque scale, term order and the gravity
sign convention are guarded by this trace and nothing else.

So the expected values do not come from this file's idea of the layout. The layout is
asserted against the CHECKPOINT'S OWN frozen config -- ``params/env.yaml`` inside the run
directory, written by mjlab at training time -- rather than against the training repo's
current HEAD, which may have moved on since. And projected_gravity is computed with
SciPy's rotation code rather than the closed form the C++ uses, so the two are genuinely
independent.

Inputs are emitted in **Unitree motor order** (FR, FL, RR, RL), which is what LowState
actually delivers, so replaying the trace exercises the joint remap in both directions.
Outputs are emitted as torques, post-saturation: exactly what fillLowCmd would put in
MotorCmd.tau.

Regenerate this after ANY retrain.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import yaml
from scipy.spatial.transform import Rotation

# Unitree LowState/LowCmd motor index for each policy joint, i.e. policy order
# (FL, FR, RL, RR) into SDK order (FR, FL, RR, RL). Mirrors flat::kPolicyToUnitree.
POLICY_TO_UNITREE = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]

# The observation terms, in order, that flat/policy_constants.hpp's kTerms declares.
# Asserted against the checkpoint's frozen config below.
EXPECTED_TERMS = [
    "joint_pos", "joint_vel", "base_ang_vel", "projected_gravity", "last_action", "command",
]

# Rows where the consumer must forget the previous action. There is no history to
# backfill, so a reset is just last_action <- 0; the second one catches a re-entry.
RESET_ROWS = {0, 100}


class _TolerantLoader(yaml.SafeLoader):
    """SafeLoader that ignores mjlab's python object tags.

    env.yaml is dumped with python/object tags for enums and dataclasses whose modules we
    do not want to import here (importing mjlab would defeat the point of reading the
    frozen config in the first place). Everything this script reads is plain scalars.
    """


def _ignore_tag(loader, tag_suffix, node):
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node, deep=True)
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node, deep=True)
    return loader.construct_scalar(node)


_TolerantLoader.add_multi_constructor("tag:yaml.org,2002:python/", _ignore_tag)
_TolerantLoader.add_multi_constructor("!", _ignore_tag)


def load_from_repo(repo: Path):
    """Import the training repo's shared constants.

    There is no go2/flat/constants.py -- the flat task's numbers live partly in the shared
    go2/constants.py and partly as literals in go2/flat/env_cfg.py. Only the stdlib-only
    module is imported here; the task-specific values come from the frozen env.yaml.
    """
    sys.path.insert(0, str(repo))

    from go2.constants import (
        CTRL_DT,
        DEFAULT_JOINT_POS,
        JOINT_NAMES,
        JOINT_TORQUE_LIMITS_FLAT,
    )

    return {
        "JOINT_NAMES": JOINT_NAMES,
        "DEFAULT_JOINT_POS": np.asarray(DEFAULT_JOINT_POS, dtype=np.float32),
        "TORQUE_LIMITS": np.asarray(JOINT_TORQUE_LIMITS_FLAT, dtype=np.float32),
        "CTRL_DT": CTRL_DT,
    }


def load_frozen_cfg(env_yaml: Path, joint_names) -> dict:
    """Read and check the checkpoint's own training config.

    Everything asserted here is something FlatPolicyRunner cannot check at startup,
    because there is no ONNX metadata to check it against. If any of it trips, the
    constants in flat/policy_constants.hpp no longer describe this checkpoint and the
    trace would be certifying the wrong contract.
    """
    with env_yaml.open() as handle:
        cfg = yaml.load(handle, Loader=_TolerantLoader)

    terms = cfg["observations"]["actor"]["terms"]
    assert list(terms) == EXPECTED_TERMS, (
        f"actor observation terms are {list(terms)}, expected {EXPECTED_TERMS} -- "
        "flat/policy_constants.hpp kTerms is out of date"
    )
    for name, term in terms.items():
        assert term["history_length"] == 0, (
            f"term '{name}' stacks {term['history_length']} frames of history; the flat "
            "controller packs a single frame (kObsDim == kFrameDim == 45)"
        )
        assert term["scale"] is None, f"term '{name}' has a scale; the C++ feeds raw SI"
        assert term["clip"] is None, f"term '{name}' is clipped; the C++ does not clip"
        assert term["delay_max_lag"] == 0, f"term '{name}' has observation delay"

    actions = cfg["actions"]
    assert list(actions) == ["joint_effort"], (
        f"action terms are {list(actions)}, expected a single 'joint_effort' -- this is "
        "a position-control checkpoint, not a torque one"
    )
    effort = actions["joint_effort"]
    assert effort["offset"] == 0.0, "joint_effort has a non-zero offset"
    assert effort["clip"] is None, "joint_effort is clipped; the C++ saturates instead"

    # scale is regex-keyed (mjlab's JOINT_REGEX); expand it against the joint names, in
    # policy order.
    scale = np.zeros(len(joint_names), dtype=np.float32)
    for i, joint in enumerate(joint_names):
        matches = [v for pattern, v in effort["scale"].items()
                   if re.fullmatch(pattern, joint)]
        assert len(matches) == 1, f"joint '{joint}' matched {len(matches)} scale patterns"
        scale[i] = matches[0]

    bounds = cfg["commands"]["twist"]["bounds"]

    return {
        "ACTION_SCALE": scale,
        "COMMAND_BOUNDS": tuple(float(b) for b in bounds),
        "CTRL_DT": float(cfg["sim"]["mujoco"]["timestep"]) * int(cfg["decimation"]),
    }


def sample_inputs(rng: np.random.Generator, default_pos_unitree: np.ndarray,
                  command_bounds) -> dict:
    """One plausible sensor reading, in Unitree motor order.

    Deliberately not a physically consistent trajectory -- this is a layout and wiring
    test, not a dynamics test. But the values stay in the range the policy actually saw
    during training so the network runs in its normal operating regime rather than in
    saturation, where tiny numerical differences would amplify.
    """
    q = default_pos_unitree + rng.uniform(-0.35, 0.35, 12).astype(np.float32)
    dq = rng.uniform(-4.0, 4.0, 12).astype(np.float32)
    gyro = rng.uniform(-1.5, 1.5, 3).astype(np.float32)

    # Near-upright orientation: small random tilt, free yaw. Kept inside the flat task's
    # pi/4 termination so the states are ones the policy was actually trained on.
    rpy = np.array([
        rng.uniform(-0.35, 0.35),
        rng.uniform(-0.35, 0.35),
        rng.uniform(-np.pi, np.pi),
    ])
    quat_xyzw = Rotation.from_euler("xyz", rpy).as_quat().astype(np.float64)

    cmd = np.array([
        rng.uniform(-command_bounds[0], command_bounds[0]),
        rng.uniform(-command_bounds[1], command_bounds[1]),
        rng.uniform(-command_bounds[2], command_bounds[2]),
    ], dtype=np.float32)

    return {"q": q, "dq": dq, "gyro": gyro, "quat_xyzw": quat_xyzw, "cmd": cmd}


def projected_gravity(quat_xyzw: np.ndarray) -> np.ndarray:
    """R^T @ (0, 0, -1), via SciPy.

    Independent of the closed-form quaternion expansion in the C++, on purpose: if both
    sides used the same algebra, a sign error would agree with itself and the test would
    pass while the robot fell over.
    """
    rot = Rotation.from_quat(quat_xyzw)
    return rot.inv().apply(np.array([0.0, 0.0, -1.0])).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True,
                        help="path to the quadruped_walking training repo")
    parser.add_argument("--onnx", type=Path,
                        default=Path(__file__).resolve().parents[1] /
                        "config/policy/flat-v0.onnx")
    parser.add_argument("--run-dir", type=Path, default=None,
                        help="training run directory holding params/env.yaml. Defaults to "
                             "logs/rsl_rl/go2_velocity/feet-time-25 inside --repo.")
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).resolve().parents[1] /
                        "test/reference/flat_feet_time_25_reference.csv")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260731)
    args = parser.parse_args()

    repo_path = args.repo.resolve()
    repo = load_from_repo(repo_path)
    joint_names = repo["JOINT_NAMES"]
    default_policy = repo["DEFAULT_JOINT_POS"]
    torque_limits = repo["TORQUE_LIMITS"]

    run_dir = args.run_dir or (repo_path / "logs/rsl_rl/go2_velocity/feet-time-25")
    frozen = load_frozen_cfg(run_dir / "params" / "env.yaml", joint_names)
    action_scale = frozen["ACTION_SCALE"]

    # The scale IS the effort limit for a JointEffortActionCfg (go2/constants.py:123-125),
    # so these must agree. If they ever diverge, kTorqueScale and kTorqueLimit in the C++
    # can no longer be the same array.
    assert np.allclose(action_scale, torque_limits), (
        f"action scale {action_scale} != effort limits {torque_limits}"
    )
    assert abs(frozen["CTRL_DT"] - repo["CTRL_DT"]) < 1e-12, (
        f"checkpoint control dt {frozen['CTRL_DT']} != go2/constants.py {repo['CTRL_DT']}"
    )

    # Sanity: the mapping table must actually correspond to the joint names the repo uses.
    unitree_names = ["FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
                     "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
                     "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
                     "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint"]
    for policy_idx, unitree_idx in enumerate(POLICY_TO_UNITREE):
        assert joint_names[policy_idx] == unitree_names[unitree_idx], (
            f"joint remap is wrong at policy index {policy_idx}: "
            f"{joint_names[policy_idx]} != {unitree_names[unitree_idx]}"
        )

    # Default pose expressed in Unitree order, for sampling.
    default_unitree = np.zeros(12, dtype=np.float32)
    for policy_idx, unitree_idx in enumerate(POLICY_TO_UNITREE):
        default_unitree[unitree_idx] = default_policy[policy_idx]

    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 1
    session_options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(args.onnx), session_options,
                                   providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    obs_dim = 45
    assert session.get_inputs()[0].shape[-1] == obs_dim, (
        f"ONNX input width {session.get_inputs()[0].shape[-1]} != 45 -- is this a rugged "
        "checkpoint?"
    )

    rng = np.random.default_rng(args.seed)
    last_action = np.zeros(12, dtype=np.float32)

    header = (
        ["step", "reset"]
        + [f"q_u{i}" for i in range(12)]
        + [f"dq_u{i}" for i in range(12)]
        + ["gyro_x", "gyro_y", "gyro_z"]
        + ["quat_w", "quat_x", "quat_y", "quat_z"]
        + ["cmd_vx", "cmd_vy", "cmd_wz"]
        + [f"obs_{i}" for i in range(obs_dim)]
        + [f"act_{i}" for i in range(12)]
        + [f"tau_u{i}" for i in range(12)]
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)

        for step in range(args.steps):
            if step in RESET_ROWS:
                # A fresh entry into the policy state: the previous action is forgotten.
                last_action = np.zeros(12, dtype=np.float32)

            sample = sample_inputs(rng, default_unitree, frozen["COMMAND_BOUNDS"])

            # Unitree order -> policy order.
            q_policy = sample["q"][POLICY_TO_UNITREE]
            dq_policy = sample["dq"][POLICY_TO_UNITREE]

            # No history: the frame IS the observation, in term order.
            obs = np.concatenate([
                (q_policy - default_policy).astype(np.float32),
                dq_policy.astype(np.float32),
                sample["gyro"].astype(np.float32),
                projected_gravity(sample["quat_xyzw"]),
                last_action.astype(np.float32),
                sample["cmd"].astype(np.float32),
            ]).astype(np.float32)
            assert obs.shape == (obs_dim,)

            action = session.run(None, {input_name: obs.reshape(1, -1)})[0][0]
            action = action.astype(np.float32)

            # tau = action * effort_limit, saturated, then back into Unitree order.
            torque_policy = np.clip(
                action * action_scale, -torque_limits, torque_limits
            ).astype(np.float32)
            torque_unitree = np.zeros(12, dtype=np.float32)
            for policy_idx, unitree_idx in enumerate(POLICY_TO_UNITREE):
                torque_unitree[unitree_idx] = torque_policy[policy_idx]

            qx, qy, qz, qw = sample["quat_xyzw"]
            row = (
                [step, 1 if step in RESET_ROWS else 0]
                + list(sample["q"]) + list(sample["dq"]) + list(sample["gyro"])
                + [qw, qx, qy, qz]
                + list(sample["cmd"]) + list(obs) + list(action) + list(torque_unitree)
            )
            writer.writerow([f"{v:.9g}" if isinstance(v, (float, np.floating)) else v
                             for v in row])

            last_action = action

    print(f"wrote {args.steps} rows to {args.out}")
    print(f"  obs dim {obs_dim} (single frame), resets at rows {sorted(RESET_ROWS)}")
    print(f"  policy  {args.onnx}")
    print(f"  layout asserted against {run_dir / 'params' / 'env.yaml'}")


if __name__ == "__main__":
    main()
