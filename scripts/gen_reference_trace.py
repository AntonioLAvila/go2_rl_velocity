#!/usr/bin/env python3
"""Generate the reference trace that test_policy_parity replays.

    REPO=/home/antonio/GitHub/quadruped_walking
    $REPO/.venv/bin/python scripts/gen_reference_trace.py --repo $REPO

Run this on a workstation that has the training repo and its uv-managed ``.venv`` --
never on the robot. It needs the full training environment, not just NumPy: importing the
repo's ``check_obs_layout.py`` pulls in torch and mjlab. Commit the resulting CSV; it is
the contract between the trained policy and this package.

Why it exists
-------------
The observation the policy reads is a 225-dim stacked history whose layout is term-major
and oldest-first, and whose joint ordering differs from the Unitree SDK's. Every one of
those conventions fails *silently* when reimplemented: the vector still has the right
length, the robot still stands, and it walks badly in a way indistinguishable from a bad
policy. A unit test written against my own understanding of the layout would encode the
same misunderstanding as the code.

So the expected values here do not come from this file's idea of the layout. They come
from ``scripts/check_obs_layout.py::HistoryPacker`` in the training repo -- the packer
that repo asserts against the live mjlab environment at ``atol=0`` -- and from the
exported ONNX itself. And projected_gravity is computed with SciPy's rotation code rather
than the closed form the C++ uses, so the two are genuinely independent.

The command envelope is the one part of the contract mjlab does not write into the ONNX
metadata, so nothing can check it at startup. It is therefore read from the CHECKPOINT'S
OWN frozen config -- ``params/env.yaml`` in the run directory the ONNX names in its
``run_path`` metadata -- rather than from the training repo's HEAD, which may have moved
on, and it is pinned into the trace: two rows carry the envelope's exact lower and upper
corners, and test_policy_parity requires kCommandLower/kCommandUpper to equal them. The
rest of the frozen config (terms, history, action law, timing) is asserted against the
repo's constants on the way, so a HEAD that no longer describes the checkpoint fails here
instead of certifying the wrong layout.

Inputs are emitted in **Unitree motor order** (FR, FL, RR, RL), which is what LowState
actually delivers, so replaying the trace exercises the joint remap in both directions.

Regenerate this after ANY retrain.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import re
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import yaml
from scipy.spatial.transform import Rotation

# Unitree LowState/LowCmd motor index for each policy joint, i.e. policy order
# (FL, FR, RL, RR) into SDK order (FR, FL, RR, RL). Mirrors kPolicyToUnitree.
POLICY_TO_UNITREE = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]

# Rows where the consumer must backfill history instead of shifting it. Row 0 is the
# entry into the policy state; the second one catches a re-entry mid-trace.
RESET_ROWS = {0, 100}

# Rows whose command is not sampled but set to one exact corner of the envelope, so the
# trace carries the envelope itself and the C++ constants can be asserted against it.
ENVELOPE_LOWER_ROW = 1
ENVELOPE_UPPER_ROW = 2


class _TolerantLoader(yaml.SafeLoader):
    """SafeLoader that ignores mjlab's python object tags.

    env.yaml is dumped with python/object tags for enums and dataclasses whose modules we
    do not want to import here. Everything this script reads is plain scalars.
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
    """Import the training repo's constants and its blessed observation packer."""
    sys.path.insert(0, str(repo))

    from go2.constants import DEFAULT_JOINT_POS, JOINT_NAMES
    from go2.rugged.constants import (
        ACTOR_OBS_DIM,
        ACTOR_TERM_WIDTHS,
        FRAME_DIM,
        HISTORY_LENGTH,
        POSITION_ACTION_SCALE_FLAT,
        POSITION_TIMING,
    )

    spec = importlib.util.spec_from_file_location(
        "_check_obs_layout", repo / "scripts" / "check_obs_layout.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return {
        "HistoryPacker": module.HistoryPacker,
        "JOINT_NAMES": JOINT_NAMES,
        "DEFAULT_JOINT_POS": np.asarray(DEFAULT_JOINT_POS, dtype=np.float32),
        "ACTION_SCALE": np.asarray(POSITION_ACTION_SCALE_FLAT, dtype=np.float32),
        "TERM_WIDTHS": ACTOR_TERM_WIDTHS,
        "FRAME_DIM": FRAME_DIM,
        "OBS_DIM": ACTOR_OBS_DIM,
        "HISTORY_LENGTH": HISTORY_LENGTH,
        "CTRL_DT": POSITION_TIMING.ctrl_dt,
    }


def load_frozen_cfg(run_dir: Path, repo: dict) -> dict:
    """Read the checkpoint's own training config; return its command envelope.

    Returns ``(lower, upper)``, each ``[vx, vy, wz]``. Along the way, everything the
    HistoryPacker and the action law below take from the repo's *current* constants is
    asserted against what the checkpoint was actually trained with.
    """
    env_yaml = run_dir / "params" / "env.yaml"
    agent_yaml = run_dir / "params" / "agent.yaml"
    assert env_yaml.is_file(), (
        f"{env_yaml} not found. The command envelope is read from the run the ONNX was "
        "exported from; pass --run-dir if that run lives somewhere else."
    )
    with env_yaml.open() as handle:
        cfg = yaml.load(handle, Loader=_TolerantLoader)
    with agent_yaml.open() as handle:
        agent = yaml.load(handle, Loader=_TolerantLoader)

    actor = cfg["observations"]["actor"]
    expected_terms = [name for name, _ in repo["TERM_WIDTHS"]]
    assert list(actor["terms"]) == expected_terms, (
        f"checkpoint actor terms are {list(actor['terms'])}, repo HEAD packs {expected_terms}"
    )
    assert actor["history_length"] == repo["HISTORY_LENGTH"], (
        f"checkpoint stacks {actor['history_length']} frames, repo HEAD "
        f"{repo['HISTORY_LENGTH']}"
    )
    for name, term in actor["terms"].items():
        assert term.get("scale") is None, f"term '{name}' has a scale; the C++ feeds raw SI"
        assert term.get("clip") is None, f"term '{name}' is clipped; the C++ does not clip"

    actions = cfg["actions"]
    assert list(actions) == ["joint_pos"], (
        f"action terms are {list(actions)}, expected a single 'joint_pos' -- this is not "
        "a PD position checkpoint"
    )
    joint_pos = actions["joint_pos"]
    assert joint_pos["use_default_offset"] is True, "action is not offset by the default pose"
    assert joint_pos["clip"] is None, "action is clipped; the C++ does not clip"
    scale = np.zeros(len(repo["JOINT_NAMES"]), dtype=np.float32)
    for i, joint in enumerate(repo["JOINT_NAMES"]):
        matches = [v for pattern, v in joint_pos["scale"].items()
                   if re.fullmatch(pattern, joint)]
        assert len(matches) == 1, f"joint '{joint}' matched {len(matches)} scale patterns"
        scale[i] = matches[0]
    assert np.array_equal(scale, repo["ACTION_SCALE"]), (
        f"checkpoint action scale {scale} != repo HEAD {repo['ACTION_SCALE']}"
    )

    ctrl_dt = float(cfg["sim"]["mujoco"]["timestep"]) * int(cfg["decimation"])
    assert abs(ctrl_dt - repo["CTRL_DT"]) < 1e-12, (
        f"checkpoint control dt {ctrl_dt} != repo HEAD {repo['CTRL_DT']}"
    )

    # The envelope. `commands.twist.ranges` is only where the curriculum STARTS; the
    # `command_vel` curriculum widens it at fixed env-step counts, so what the finished
    # policy trained on is the last stage the run was long enough to reach.
    ranges = dict(cfg["commands"]["twist"]["ranges"])
    axes = ("lin_vel_x", "lin_vel_y", "ang_vel_z")
    # mjlab compares stage steps against env.common_step_counter, which advances once per
    # env.step(): num_steps_per_env per training iteration.
    total_steps = int(agent["max_iterations"]) * int(agent["num_steps_per_env"])
    curriculum = (cfg.get("curriculum") or {}).get("command_vel")
    stages = curriculum["params"]["velocity_stages"] if curriculum else []
    for stage in sorted(stages, key=lambda item: item["step"]):
        if stage["step"] > total_steps:
            print(f"  note: command stage at step {stage['step']} is past the end of the "
                  f"run ({total_steps} steps) and was never reached")
            continue
        for axis in axes:
            if stage.get(axis) is not None:
                ranges[axis] = stage[axis]

    lower = np.array([ranges[axis][0] for axis in axes], dtype=np.float32)
    upper = np.array([ranges[axis][1] for axis in axes], dtype=np.float32)
    assert np.all(lower < upper), f"degenerate command envelope {lower} .. {upper}"
    return {"COMMAND_LOWER": lower, "COMMAND_UPPER": upper}


def sample_inputs(rng: np.random.Generator, default_pos_unitree: np.ndarray,
                  command_lower: np.ndarray, command_upper: np.ndarray) -> dict:
    """One plausible sensor reading, in Unitree motor order.

    Deliberately not a physically consistent trajectory -- this is a layout and wiring
    test, not a dynamics test. But the values stay in the range the policy actually saw
    during training so the network runs in its normal operating regime rather than in
    saturation, where tiny numerical differences would amplify.
    """
    q = default_pos_unitree + rng.uniform(-0.35, 0.35, 12).astype(np.float32)
    dq = rng.uniform(-4.0, 4.0, 12).astype(np.float32)
    gyro = rng.uniform(-1.5, 1.5, 3).astype(np.float32)

    # Near-upright orientation: small random tilt, free yaw.
    rpy = np.array([
        rng.uniform(-0.35, 0.35),
        rng.uniform(-0.35, 0.35),
        rng.uniform(-np.pi, np.pi),
    ])
    quat_xyzw = Rotation.from_euler("xyz", rpy).as_quat().astype(np.float64)

    cmd = rng.uniform(command_lower, command_upper).astype(np.float32)

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
                        default=Path(__file__).resolve().parents[1] / "config/policy/latest.onnx")
    parser.add_argument("--run-dir", type=Path, default=None,
                        help="training run the ONNX came from (holds params/env.yaml). "
                             "Default: <repo>/logs/rsl_rl/go2_rugged/<run_path from the "
                             "ONNX metadata>")
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).resolve().parents[1] /
                        "test/reference/latest_reference.csv")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260730)
    args = parser.parse_args()

    repo_path = args.repo.resolve()
    repo = load_from_repo(repo_path)
    packer_cls = repo["HistoryPacker"]
    joint_names = repo["JOINT_NAMES"]
    default_policy = repo["DEFAULT_JOINT_POS"]
    action_scale = repo["ACTION_SCALE"]
    obs_dim = repo["OBS_DIM"]

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
    assert session.get_inputs()[0].shape[-1] == obs_dim, "ONNX input width != ACTOR_OBS_DIM"

    run_path = session.get_modelmeta().custom_metadata_map.get("run_path")
    run_dir = args.run_dir
    if run_dir is None:
        assert run_path, "ONNX has no run_path metadata; pass --run-dir"
        run_dir = repo_path / "logs/rsl_rl/go2_rugged" / run_path
    frozen = load_frozen_cfg(run_dir.resolve(), repo)
    command_lower = frozen["COMMAND_LOWER"]
    command_upper = frozen["COMMAND_UPPER"]

    rng = np.random.default_rng(args.seed)
    packer = packer_cls(repo["HISTORY_LENGTH"])
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
        + [f"tgt_u{i}" for i in range(12)]
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)

        for step in range(args.steps):
            if step in RESET_ROWS:
                # A fresh entry into the policy state: history is backfilled and the
                # previous action is forgotten.
                packer = packer_cls(repo["HISTORY_LENGTH"])
                last_action = np.zeros(12, dtype=np.float32)

            sample = sample_inputs(rng, default_unitree, command_lower, command_upper)
            if step == ENVELOPE_LOWER_ROW:
                sample["cmd"] = command_lower.copy()
            elif step == ENVELOPE_UPPER_ROW:
                sample["cmd"] = command_upper.copy()

            # Unitree order -> policy order.
            q_policy = sample["q"][POLICY_TO_UNITREE]
            dq_policy = sample["dq"][POLICY_TO_UNITREE]

            frame = np.concatenate([
                (q_policy - default_policy).astype(np.float32),
                dq_policy.astype(np.float32),
                sample["gyro"].astype(np.float32),
                projected_gravity(sample["quat_xyzw"]),
                last_action.astype(np.float32),
                sample["cmd"].astype(np.float32),
            ]).astype(np.float32)

            packer.append(frame)
            obs = packer.flat().astype(np.float32)
            action = session.run(None, {input_name: obs.reshape(1, -1)})[0][0]
            action = action.astype(np.float32)

            # target = action * scale + default, then back into Unitree order.
            target_policy = (action * action_scale + default_policy).astype(np.float32)
            target_unitree = np.zeros(12, dtype=np.float32)
            for policy_idx, unitree_idx in enumerate(POLICY_TO_UNITREE):
                target_unitree[unitree_idx] = target_policy[policy_idx]

            qx, qy, qz, qw = sample["quat_xyzw"]
            row = (
                [step, 1 if step in RESET_ROWS else 0]
                + list(sample["q"]) + list(sample["dq"]) + list(sample["gyro"])
                + [qw, qx, qy, qz]
                + list(sample["cmd"]) + list(obs) + list(action) + list(target_unitree)
            )
            writer.writerow([f"{v:.9g}" if isinstance(v, (float, np.floating)) else v
                             for v in row])

            last_action = action

    print(f"wrote {args.steps} rows to {args.out}")
    print(f"  obs dim {obs_dim}, resets at rows {sorted(RESET_ROWS)}")
    print(f"  policy  {args.onnx}")
    print(f"  run     {run_path} ({run_dir})")
    print(f"  command envelope [vx, vy, wz]: lower {command_lower.tolist()}, "
          f"upper {command_upper.tolist()}")
    print("  -> kTrainedRunPath, kCommandLower and kCommandUpper in policy_constants.hpp "
          "must match these")


if __name__ == "__main__":
    main()
