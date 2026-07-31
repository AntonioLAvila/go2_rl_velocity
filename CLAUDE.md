# go2_rl_velocity

ROS 2 Jazzy controller that runs a trained RL locomotion policy on a Unitree Go2.
Deployment target is a **Jetson Orin Nano on JetPack 7.2**, talking to the robot through
the `unitree_ros2` low-level bridge.

Edit this file — not one in `ros2_ws/` — when working on this package.

## What this is

The policy comes from `/home/antonio/GitHub/quadruped_walking` (mjlab + rsl_rl PPO), from
the **rugged** task: blind velocity-tracking locomotion, PD position control at 50 Hz, no
height scan and no base linear velocity. The actor sees only joint encoders, the IMU, and
the operator's command, which is exactly what makes it deployable without an elevation
map. That repo's `CLAUDE.md` is accurate and worth reading before changing anything here.

Checkpoint in use: `config/policy/rugged-v2.onnx`, copied verbatim from
`logs/rsl_rl/go2_rugged/2026-07-30_10-38-23_rugged-v2/`.

## Commands

```bash
# Build. ONNXRUNTIME_ROOT is required unless ORT is installed system-wide.
cd ~/ros2_ws
colcon build --packages-select go2_rl_velocity \
  --cmake-args -DONNXRUNTIME_ROOT=/path/to/onnxruntime

# Test.
colcon test --packages-select go2_rl_velocity && colcon test-result --verbose

# Run.
ros2 launch go2_rl_velocity go2_rl_velocity.launch.py

# Verify the wireless remote's button layout before trusting any transition.
ros2 launch go2_rl_velocity go2_rl_velocity.launch.py remote_debug:=true

# Full controller, no LowCmd published. For bench work against a live robot.
ros2 launch go2_rl_velocity go2_rl_velocity.launch.py dry_run:=true
```

Without `unitree_go` on the prefix path, the build **deliberately skips the node** and
builds only `go2_rl_policy` plus its tests. That is not a failure; it is what lets the
policy pipeline be tested on any machine. Watch for the CMake warning naming what was
missing.

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

## After any retrain

1. Copy the new ONNX to `config/policy/`.
2. Re-check `policy_constants.hpp` against the training repo's `go2/rugged/constants.py`.
3. **Regenerate the reference trace** — the tests are meaningless against a stale one:
   ```bash
   /home/antonio/python_venvs/general/bin/python scripts/gen_reference_trace.py \
     --repo /home/antonio/GitHub/quadruped_walking
   ```
4. `colcon test`. A changed `HISTORY_LENGTH` shows up as a startup contract error.

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
- `test_observation_builder`, `test_joint_map`, `test_unitree_remote` — structure, the
  SDK motor table, edge-triggered chords, CRC behaviour.

`projected_gravity` expectations are generated with SciPy, deliberately, so the C++
closed-form quaternion expansion is checked against an independent implementation rather
than a copy of its own algebra.

### Compiling the node without unitree_ros2

To type-check the node on a machine lacking `unitree_go`, reconstruct the message package
from the SDK IDL (`~/unitree_sdk2_python/unitree_sdk2py/idl/unitree_go/msg/dds_/`) into a
throwaway workspace and build against it. That is how the node in this repo was verified,
including an end-to-end run driving the FSM with synthetic LowState. Keep it out of
`ros2_ws/src` — this repo is single-package on purpose.

## Architecture

Two rates, mirroring Unitree's own deploy FSM (`CtrlFSM` at 1 kHz, policy thread at
`step_dt`):

- **`/lowcmd` publisher, 500 Hz** — always sends the currently held target, in every
  state. The motor boards want feeding continuously.
- **Policy timer, 50 Hz** — matches the training decimation and only recomputes the
  target. Running inference at the publish rate would be 4x faster than anything the
  policy saw in training.
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

`go2_rl_policy` (the core library) has no ROS and no `unitree_go` dependency. Keep it
that way; it is what makes the parity test runnable anywhere.

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

## Jetson notes

- **CPU execution provider, 1 intra-op thread.** Measured 12.7 us per inference on x86;
  the graph is ~280k MACs against a 20 ms budget. A CUDA EP would add transfer and launch
  overhead exceeding the compute, plus jitter. Do not "optimise" this onto the GPU.
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
  pin `sizeof(LowCmd) == 812` with `crc` at offset 808; a different `unitree_go` version
  will fail the build rather than emit commands the robot silently discards.
