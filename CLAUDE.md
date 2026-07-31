# go2_rl_velocity

ROS 2 Jazzy controller that runs a trained RL locomotion policy on a Unitree Go2.
Deployment target is a **Jetson Orin Nano on JetPack 7.2**, talking to the robot through
the `unitree_ros2` low-level bridge.

Edit this file — not one in `ros2_ws/` — when working on this package.

## What this is

**Two independent controllers**, one per task in `/home/antonio/GitHub/quadruped_walking`
(mjlab + rsl_rl PPO). Both are blind velocity-tracking locomotion with no height scan and
no base linear velocity: the actor sees only joint encoders, the IMU and the operator's
command, which is what makes them deployable without an elevation map. That repo's
`CLAUDE.md` is accurate and worth reading before changing anything here.

| | rugged | flat |
|---|---|---|
| task | `Mjlab-Velocity-Rugged-Unitree-Go2` | `Mjlab-Velocity-Flat-Unitree-Go2` |
| control | PD position, motor boards close the loop | **direct torque** |
| rate | 50 Hz (`0.005 x 4`) | **200 Hz (`0.0025 x 2`)** |
| observation | 225 = 45 x 5, term-major history | **45, single frame** |
| commands | `vx +/-2.0`, `vy +/-1.0`, `wz +/-1.5` | **`+/-1.5`, `+/-0.8`, `+/-1.2`** |
| tilt limit | `gravity.z > -0.6` (53 deg) | **`> -0.7071` (45 deg)** |
| ONNX metadata | full, validated at startup | **absent — shape check only** |
| checkpoint | `config/policy/rugged-v2.onnx` | `config/policy/flat-feet-time-25.onnx` |
| from | `logs/rsl_rl/go2_rugged/2026-07-30_10-38-23_rugged-v2/` | `logs/rsl_rl/go2_velocity/feet-time-25/` |
| namespace | `go2_rl_velocity::policy` | `go2_rl_velocity::flat` |
| library | `go2_rl_policy` | `go2_rl_flat_policy` |
| node / executable | `go2_rl_velocity` / `go2_rl_velocity_node` | `go2_rl_flat` / `go2_rl_flat_node` |

**The networks are architecturally identical** — a 512-256-128 ELU MLP with a scalar-std
Gaussian head exported deterministically, `rnn_type: null`, `cnn_cfg: null`,
`class_name: MLPModel` in both frozen `agent.yaml`s, and the same
`Sub, Div, Gemm, Elu, Gemm, Elu, Gemm, Elu, Gemm` graph. The only weight-level difference
is `mlp.0.weight` being `[512,45]` instead of `[512,225]`. Nothing is recurrent; there is
no encoder and no distilled teacher. Everything that matters for deployment is in the
table above, not in the topology.

**Keep the two apart.** Values that happen to agree (joint order, `q_default`, the gravity
convention, the stand gains) are duplicated between the two constants headers on purpose,
so a number changed for one policy cannot silently move the other. Only genuinely
policy-agnostic code is shared, in `go2_rl_common`: `crc32` and `unitree_remote`.

## Commands

```bash
# Build. ONNXRUNTIME_ROOT is required unless ORT is installed system-wide.
cd ~/ros2_ws
colcon build --packages-select go2_rl_velocity \
  --cmake-args -DONNXRUNTIME_ROOT=/path/to/onnxruntime

# Test.
colcon test --packages-select go2_rl_velocity && colcon test-result --verbose

# Run. One at a time -- both nodes publish /lowcmd.
ros2 launch go2_rl_velocity go2_rl_velocity.launch.py          # rugged, PD position
ros2 launch go2_rl_velocity go2_rl_flat.launch.py              # flat, torque

# Verify the wireless remote's button layout before trusting any transition.
ros2 launch go2_rl_velocity go2_rl_velocity.launch.py remote_debug:=true

# Full controller, no LowCmd published. For bench work against a live robot.
ros2 launch go2_rl_velocity go2_rl_velocity.launch.py dry_run:=true

# First hardware run of the flat policy: hoisted, derated. See "Flat-ground deployment".
ros2 launch go2_rl_velocity go2_rl_flat.launch.py torque_scale:=0.3
```

Without `unitree_go` on the prefix path, the build **deliberately skips both nodes** and
builds only `go2_rl_policy`, `go2_rl_flat_policy` and their tests. That is not a failure;
it is what lets the policy pipelines be tested on any machine. Watch for the CMake warning
naming what was missing.

## The contract that breaks silently

Everything below is mirrored in `include/go2_rl_velocity/policy_constants.hpp`, which is
the only place numbers live. `PolicyRunner` cross-checks most of it against the ONNX
metadata at startup and refuses to run on a mismatch.

**Observation: 225 = 45 x 5, TERM-MAJOR, oldest first.** mjlab stacks history per term
and concatenates afterwards. This is the opposite of legged_gym's time-major convention,
and getting it wrong yields a correctly-sized vector of garbage — the robot walks badly in
a way indistinguishable from a bad policy.

| Offset | Term | Content |
|---|---|---|
| `[0:60]` | `joint_pos` x5 | `q - q_default` |
| `[60:120]` | `joint_vel` x5 | `qd` |
| `[120:135]` | `base_ang_vel` x5 | IMU gyro, body frame |
| `[135:150]` | `projected_gravity` x5 | `R^T (0,0,-1)` |
| `[150:210]` | `last_action` x5 | **raw** pre-scale `a_{t-1}` |
| `[210:225]` | `command` x5 | `[vx, vy, wz]` |

- **No scaling, no clipping, no noise.** All terms are raw SI. Observation normalisation
  is baked into the ONNX graph as leading `Sub`/`Div` nodes, so feed raw values in.
- **Backfill, never zero-pad.** On entering RL, every history slot gets the first frame.
- **`last_action` is the raw action**, not the scaled position target.

**Joint order.** Policy: `FL, FR, RL, RR`. Unitree LowState/LowCmd: `FR(0-2), FL(3-5),
RR(6-8), RL(9-11)`. Forgetting the remap swaps left and right legs — the robot stands
happily and falls the moment it moves.

**Control law.** `target = action * action_scale + q_default`, sent as `motor_cmd[].q`
with `kp = {20,20,40}x4`, `kd = {1,1,2}x4`. The motor boards close the PD loop; this
package never computes torque. Use `0.29375 / 0.28125` from source — the ONNX metadata
rounds action scale to 3 decimals.

**Command envelope:** `vx ±2.0 m/s`, `vy ±1.0`, `wz ±1.5 rad/s`. The checkpoint trained
against a 2.5 m/s ceiling that the training repo later measured as counterproductive and
capped; 2.0 is the real limit.

## The flat contract

Mirrored in `include/go2_rl_velocity/flat/policy_constants.hpp`. Same six terms in the same
order as the rugged frame — but **no history**, so the 45-dim frame *is* the observation
and term order is the entire layout contract. `[0:12] joint_pos · [12:24] joint_vel ·
[24:27] base_ang_vel · [27:30] projected_gravity · [30:42] last_action · [42:45] command`,
raw SI, no scale, no clip, normalisation baked into the graph. Joint order and `q_default`
are identical to rugged.

**Control law: `tau = action * effort_limit`.** `JointEffortActionCfg`, offset 0, scale
`{23.5, 23.5, 45.0}` per leg — `go2/constants.py`'s `GO2_ACTION_SCALE`, which *is* the
effort limit. Saturation to the same numbers stands in for MuJoCo's `<motor ctrlrange>`.
Sent as `motor_cmd[].tau` with `kp = 0`.

**`kd = 2.0` in RL is load-bearing, not a stylistic choice.** Every joint in training
carried `GO2_ACTUATORS[*].damping = 2.0 N·m·s/rad` of passive viscous damping that the real
Go2 does not have, and a torque policy has no PD loop to hide behind. The motor board
computes `tau_total = kp(q_des−q) + kd(dq_des−dq) + tau`, so `kd = 2.0` with `dq_des = 0`
reproduces sim's `−2.0·qd` exactly. At a 10 rad/s swing that is ~20 N·m, comparable to the
effort limit — omitting it lets the same torques produce far faster joint motion than the
policy ever saw. The contract value is `kPolicyKd`; `passive_damping_kd` in the YAML
defaults to it and exists so it can be dialled on the bench without a rebuild.

**No metadata to validate.** mjlab's `get_base_metadata` opens with
`env.action_manager.get_term("joint_pos")`; the flat task's term is `joint_effort`, the
lookup raises, `save()` swallows it, and the ONNX is written with an empty custom metadata
map. `flat::PolicyRunner` therefore checks only 45-in/12-out and warns at startup. Fixing
this upstream would need a torque-mode branch in the exporter — `joint_stiffness` and
`joint_damping` are read from `actuator_gainprm`/`actuator_biasprm`, meaningless for a
`<motor>`.

**Command envelope:** `vx ±1.5 m/s`, `vy ±0.8`, `wz ±1.2 rad/s`. Fixed — the flat task has
no command curriculum.

## After any retrain

1. Copy the new ONNX to `config/policy/`.
2. Re-check the matching `policy_constants.hpp` against the training repo — rugged against
   `go2/rugged/constants.py`, flat against `go2/constants.py` plus `go2/flat/env_cfg.py`
   (there is no `go2/flat/constants.py`).
3. **Regenerate the reference trace** — the tests are meaningless against a stale one:
   ```bash
   /home/antonio/python_venvs/general/bin/python scripts/gen_reference_trace.py \
     --repo /home/antonio/GitHub/quadruped_walking
   /home/antonio/python_venvs/general/bin/python scripts/gen_flat_reference_trace.py \
     --repo /home/antonio/GitHub/quadruped_walking --run-dir <new run dir>
   ```
   The flat generator asserts the layout against the checkpoint's own frozen
   `params/env.yaml`, not against training HEAD, so point `--run-dir` at the run the ONNX
   came from.
4. `colcon test`. For rugged, a changed `HISTORY_LENGTH` also shows up as a startup
   contract error. **For flat there is no such backstop** — the trace is the only guard.

## Testing strategy

There is no simulation backend here (by design — the training repo's
`scripts/verify_rugged.py` is the Drake sim-to-sim check). Correctness rests on:

- **`test_policy_parity`** — the primary guard. Replays `test/reference/*.csv`, whose
  expected values come from the *training repo's own* observation packer
  (`scripts/check_obs_layout.py::HistoryPacker`, asserted against the live mjlab env at
  `atol=0`) and from the ONNX itself. Covers layout, history direction, backfill, the
  joint remap in both directions, the gravity sign convention, the `last_action`
  off-by-one, and action→target.
- **`PolicyParityDiscrimination`** — proves the trace *rejects* time-major packing,
  reversed history, a missing joint remap, and a flipped gravity sign. Without these, a
  passing parity test proves nothing. If any of them starts passing, the trace has stopped
  constraining what it was written to constrain.
- **`test_flat_policy_parity`** — the same idea for the flat policy, and the *only* guard
  on its contract, since there is no metadata to check at startup. Covers term order, the
  joint remap, the gravity sign, the `last_action` off-by-one, and action→torque→saturation.
  Its `FlatPolicyParityDiscrimination` suite replaces the history checks (meaningless
  without history) with ones that constrain a single frame: a missing joint remap, a
  flipped gravity sign, raw `q` instead of `q - q_default`, a swapped gyro/gravity block,
  and — because both controllers live in this package — the rugged *position* action law
  applied to flat actions.
- **`test_flat_controller_fsm`** — the flat FSM is in the core library, not the node target
  (unlike the rugged one), specifically so the position→torque handover and the two
  torque-mode watchdogs can be tested without a robot. Covers the blend endpoints, the
  torque ramp, `torque_scale`, every fault path, and that no held torque survives a fault.
- `test_observation_builder`, `test_joint_map`, `test_unitree_remote`,
  `test_flat_state_conversion` — structure, the SDK motor table, edge-triggered chords, CRC
  behaviour, the torque law and the soft-limit band.

`projected_gravity` expectations are generated with SciPy, deliberately, so the C++
closed-form quaternion expansion is checked against an independent implementation rather
than a copy of its own algebra.

### Compiling the nodes without unitree_ros2

To type-check the nodes on a machine lacking `unitree_go`, reconstruct the message package
from the SDK IDL (`~/unitree_sdk2_python/unitree_sdk2py/idl/unitree_go/msg/dds_/`) into a
throwaway workspace and build against it — `LowCmd`, `LowState`, `MotorCmd`, `MotorState`,
`IMUState`, `BmsCmd`, `BmsState`. A faithful reconstruction gives
`sizeof(LowCmd) == 812` with `crc` at 808, which the `static_assert`s check, so a wrong one
fails loudly. That is how both nodes in this repo were verified, including an end-to-end
run driving each FSM with synthetic LowState. Build it into a separate `--build-base` so a
binary linked against fake messages never lands in the real install tree, and keep the
workspace out of `ros2_ws/src` — this repo is single-package on purpose.

## Architecture

Two rates in each controller, mirroring Unitree's own deploy FSM (`CtrlFSM` at 1 kHz,
policy thread at `step_dt`):

- **`/lowcmd` publisher, 500 Hz** — always sends the currently held command, in every
  state. The motor boards want feeding continuously. Republishing a held value between
  policy ticks is the same zero-order hold mjlab's decimation applies in training.
- **Policy timer, 50 Hz (rugged) / 200 Hz (flat)** — matches the training decimation and
  only recomputes the command. Running inference at the publish rate would be faster than
  anything the policy saw in training.
- `/lowstate` is latched newest-wins (`KeepLast(1)`, best-effort). A queue would feed the
  policy stale joint angles.

| File | Role |
|---|---|
| `policy_constants.hpp` | Every number. Mirrors the training repo. |
| `observation_builder.*` | The 5-frame term-major history buffer. |
| `policy_runner.*` | ORT session, contract validation, warm-up. |
| `state_conversion.hpp` | Quaternion→gravity, joint remap, action→target, limit clamp. |
| `controller_fsm.*` | States, transitions, watchdogs, command shaping. |
| `rl_controller_node.*` | ROS wiring, the two timers, LowCmd assembly + CRC. |
| `unitree_remote.*` | Wireless remote decode, edge-triggered chords. |
| `crc32.*` | Unitree's MPEG-2 CRC. The Go2 silently drops LowCmd without it. |

`flat/` mirrors the first six of those, in `namespace go2_rl_velocity::flat`, with three
differences: `observation.hpp` is a bare frame struct rather than a history buffer,
`state_conversion.hpp` carries `actionToTorque`/`clampToTorqueLimits`/`outsideSoftJointLimits`
instead of `actionToTarget`, and `controller_fsm.cpp` lives in the *library* rather than the
node target so the handover is testable.

`go2_rl_policy` and `go2_rl_flat_policy` have no ROS and no `unitree_go` dependency. Keep
them that way; it is what makes the parity tests runnable anywhere. `go2_rl_common`
(`crc32`, `unitree_remote`) is the only code both controllers share — everything else is
duplicated deliberately.

## FSM

| State | Behaviour | Gains |
|---|---|---|
| `PASSIVE` | Limp, damping only | kp 0, kd 3 |
| `FIXSTAND` | measured → crouch `[0,1.36,-2.65]` → **`[0,0.9,-1.8]`**, two smoothstep segments | kp `[60,80,80]`, kd `[5,4,4]` |
| `RL` | 0.5 s gain blend holding default pose, then the policy | kp `[20,20,40]`, kd `[1,1,2]` |
| `DAMPING` | Soft collapse, then back to `PASSIVE` | kp 0, kd 3 |

Transitions: `L2+↑` stand · `R2+A` hand over to the policy · `L2+B` e-stop · any fault →
`DAMPING`. Services: `~/passive`, `~/stand`, `~/start_policy`, `~/estop`. `RL` can only be
entered from a completed `FIXSTAND` — the policy's observations are relative to its
default pose, so it must start there. That is why `FIXSTAND` targets `0.9/-1.8` and not
Unitree's stock `0.8/-1.5`.

Watchdogs, all faulting to `DAMPING`: LowState staleness (>100 ms), tilt
(`projected_gravity.z > -0.6`), non-finite policy output, sustained joint-limit clamping,
sustained inference overrun. A stale `/cmd_vel` decays the command to zero rather than
faulting — losing operator input should stop the robot, not drop it.

### Flat FSM

Same four states, same chords, same services, same scripted position-controlled stand. It
differs only where the control mode forces it to:

| State | Behaviour | Gains |
|---|---|---|
| `PASSIVE` | Limp, damping only, `tau = 0` | kp 0, kd 3 |
| `FIXSTAND` | Identical to rugged, `tau = 0` | kp `[60,80,80]`, kd `[5,4,4]` |
| `RL` | 0.5 s blend `kp → 0`, `kd → 2.0` holding the default pose, then `torque_ramp_s` of smoothstepped torque | **kp 0, kd 2, `tau` from the policy** |
| `DAMPING` | Soft collapse, `tau = 0`, then `PASSIVE` | kp 0, kd 3 |

The handover works because the motor board *sums* the PD and feed-forward terms, so the
two authorities overlap continuously instead of switching. Every non-RL state clears `tau`
explicitly — a stale torque would keep being republished at 500 Hz.

Watchdogs: staleness (>50 ms, half of rugged's — 5 ms control period), tilt
(`> -0.7071`), non-finite output, inference overrun, and two that replace the rugged
joint-target clamp, which has no meaning without a position target:

- **sustained torque saturation** (`torque_saturation_fault_steps`), and
- **sustained joint excursion** (`joint_excursion_fault_steps`) — a *measured* joint
  outside mjlab's soft limit band (0.95 of the mechanical range about its midpoint).
  Nothing prevents a torque policy driving a joint into its stop, so the excursion is
  detected after the fact rather than prevented.

`torque_scale` derates the whole command multiplicatively before saturation, for bring-up.

## Jetson notes

- **CPU execution provider, 1 intra-op thread.** Measured 12.7 us per inference on x86;
  the graph is ~280k MACs against a 20 ms budget (5 ms for flat). A CUDA EP would add
  transfer and launch overhead exceeding the compute, plus jitter. Do not "optimise" this
  onto the GPU.
- **The flat controller's 5 ms period is the tight one.** Inference is not the problem;
  LowState delivery, DDS and scheduler jitter all now have to fit inside one control
  period. Measure `~/status` inference times under load before trusting it.
- Needs ONNX Runtime ≥ 1.15 (opset 18). `FindONNXRUNTIME.cmake` fails at configure time
  if `ORT_API_VERSION` is older.
- `CMAKE_INSTALL_RPATH_USE_LINK_PATH` keeps ORT's lib dir in the installed RUNPATH.
  Without it the node builds and installs cleanly, then dies at exec with
  `libonnxruntime.so.1: cannot open shared object file`.
- The node calls `mlockall`; grant `RLIMIT_MEMLOCK` headroom or it logs a warning and
  runs unlocked.
- Set the board to MAXN and run `jetson_clocks` before any timing measurement.
- `unitree_go` has no rosdep key. Source the `unitree_ros2` workspace as an underlay.

## Gotchas

- The wireless-remote **bit layout is undocumented**. The mapping in `unitree_remote.hpp`
  is the one every Unitree example uses, but confirm it with `remote_debug:=true` against
  the physical remote before relying on a transition.
- Joystick axis signs are parameters (`joystick_sign`) for the same reason — flip one in
  YAML rather than rebuilding.
- LowCmd carries 20 motor slots; the Go2 uses 12. Slots 12–19 are explicitly idled every
  message.
- The CRC covers every word except its own. `static_assert`s in `rl_controller_node.cpp`
  and `flat/flat_controller_node.cpp` pin `sizeof(LowCmd) == 812` with `crc` at offset 808;
  a different `unitree_go` version will fail the build rather than emit commands the robot
  silently discards.
- Both YAMLs are keyed by node name, not `/**`. Passing one to the wrong launch file then
  does nothing instead of something subtle.
- **Only run one controller at a time.** Both publish `/lowcmd`; nothing in either node
  detects the other.

## Flat-ground deployment

Treat it as higher-risk than rugged, for reasons that are properties of the training run
rather than of this code:

- **No motor-strength randomisation was possible.** mjlab's `dr.pd_gains` and
  `dr.effort_limits` both raise `TypeError` on a torque `<motor>`, so the flat policy has
  never seen actuator-gain or torque-limit uncertainty — precisely the quantities (torque
  constant, thermal derating, gearbox efficiency) you cannot trust on hardware. This is
  why `torque_scale` exists. Bring it up hoisted at 0.3.
- **It has never seen a crouch.** `root_height_below_minimum(0.245)` terminated the episode
  2.5 cm below the 0.27 m nominal stance, so there is no recovery behaviour from a low
  pose. The handover must land exactly on `[0, 0.9, -1.8]`.
- **Zero command is out of distribution.** The flat sampler has no standing-still resamples
  (rugged has `REL_STANDING_PROB = 0.1`), so the stale-`/cmd_vel` decay-to-zero puts the
  robot somewhere it rarely trained. Expect foot shuffle at rest.
- **Flat plane only.** No slope, no step, ever, in training.
