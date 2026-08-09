# go2_rl_velocity

Runs a reinforcement-learning locomotion policy on a Unitree Go2 as a ROS 2 Jazzy node.

The policies are blind, proprioceptive velocity-tracking controllers trained in
[mjlab](https://github.com/mujocolab/mjlab) with rsl_rl PPO and exported to ONNX. They need
no elevation map and no base-velocity estimate — joint encoders, the IMU and the operator's
command are enough — which is what makes them deployable on hardware.

There are **two independent controllers**, one per training task. Only run one at a time;
they both publish `/lowcmd`.

| | `go2_rl_velocity_node` (rugged) | `go2_rl_flat_node` (flat) |
| --- | --- | --- |
| terrain | generated, rough | flat ground only |
| control | PD position, motor boards close the loop | **direct torque** |
| rate | 50 Hz | **200 Hz** |
| observation | 225 = 45 × 5 frames of history | **45, single frame** |
| command envelope | `vx ±2.0`, `vy ±1.0`, `wz ±1.5` | **`±1.5`, `±0.8`, `±1.2`** |
| launch file | `go2_rl_velocity.launch.py` | `go2_rl_flat.launch.py` |

The rugged controller is the safer of the two — see [Flat-ground controller](#flat-ground-controller)
before running the flat one on hardware.

Deployment target: **Jetson Orin Nano, JetPack 7.2**, driving the robot through the
`unitree_ros2` low-level bridge.

## Requirements

| | |
| --- | --- |
| ROS 2 | Jazzy |
| ONNX Runtime | ≥ 1.15 (the policy is opset 18). C++ API, CPU build is sufficient. |
| `unitree_ros2` | Provides the `unitree_go` messages and the DDS bridge to the robot. |

`unitree_go` has no rosdep key — build [unitree_ros2](https://github.com/unitreerobotics/unitree_ros2)
and source it as an underlay. Its patched CycloneDDS is the fiddliest part of the setup;
validate it standalone (`ros2 topic echo /lowstate`) before wiring this node in.

## Build and test

```bash
cd ~/ros2_ws
source /opt/ros/jazzy/setup.bash
source /path/to/unitree_ros2/install/setup.bash      # provides unitree_go

colcon build --packages-select go2_rl_velocity \
  --cmake-args -DONNXRUNTIME_ROOT=/opt/onnxruntime

colcon test --packages-select go2_rl_velocity && colcon test-result --verbose
```

`ONNXRUNTIME_ROOT` can be omitted if ONNX Runtime is installed under `/usr` or
`/usr/local`, and can also be given as an environment variable.

Without `unitree_go` on the prefix path the build **skips both controller nodes** and
builds only the policy core libraries and their tests. That is intentional — it lets the
observation/inference pipeline be tested on a workstation. A CMake warning names what was
missing.

The tests that matter are `test_policy_parity` and `test_flat_policy_parity`: each replays
a trace derived from the training repo and proves this package reproduces it exactly. The
flat one carries extra weight, because flat ONNX exports have no metadata for the node to
validate at startup. See `CLAUDE.md` for what they cover.

## Run

```bash
ros2 launch go2_rl_velocity go2_rl_velocity.launch.py
```

Put the robot in low-level mode first (release sport mode), lying on a flat surface.

| Input | Effect |
| --- | --- |
| `L2` + `↑` | Stand up to the policy's default pose |
| `R2` + `A` | Hand control to the policy |
| Left stick | Forward / lateral velocity |
| Right stick (horizontal) | Yaw rate |
| `L2` + `B` | E-stop — soft collapse |

Equivalent services: `~/stand`, `~/start_policy`, `~/passive`, `~/estop`.

To drive from ROS instead of the remote:

```bash
ros2 launch go2_rl_velocity go2_rl_velocity.launch.py use_joystick_command:=false
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist '{linear: {x: 0.5}}'
```

### Topics

| Topic | Type | Direction |
| --- | --- | --- |
| `/lowstate` | `unitree_go/msg/LowState` | in |
| `/lowcmd` | `unitree_go/msg/LowCmd` | out, 500 Hz |
| `/cmd_vel` | `geometry_msgs/msg/Twist` | in, optional |
| `~/status` | `std_msgs/msg/String` | out, 2 Hz |

The flat controller is the same in every respect above, under `go2_rl_flat.launch.py` and
node name `go2_rl_flat`.

## Bringing it up safely

Do this in order. Steps 1–3 need the robot on a stand with its feet clear of the ground.

1. **Check the bridge.** `ros2 topic hz /lowstate` should show a few hundred Hz.
2. **Check the remote.** Launch with `remote_debug:=true` and press every button used by a
   transition. The Go2's wireless-remote bit layout is undocumented; the decoder follows
   the conventional mapping, but confirm it rather than assume it. Fix stick directions
   with the `joystick_sign` parameter, not a rebuild.
3. **Check the loop without commanding.** Launch with `dry_run:=true` and confirm
   `~/status` reports sensible states and inference times.
4. **Stand.** `L2+↑`. The legs should reach `[0, 0.9, -1.8]` smoothly.
5. **Hand over, stationary.** `R2+A` with sticks centred. The policy should hold a near
   still stance.
6. **E-stop.** `L2+B` — confirm immediate soft collapse. Then confirm the staleness
   watchdog by killing the bridge; the controller must fall back to `DAMPING`.
7. **Ground test.** Low speed first (`vx ≤ 0.5`), flat ground, safety handle attached.
   Only work up toward 2.0 m/s once the tilt watchdog and e-stop have both been seen to
   fire cleanly.

## Configuration

`config/go2_rl_velocity.yaml` and `config/go2_rl_flat.yaml` hold deployment behaviour —
rates, timeouts, the operator interface. Every parameter is commented there. Each is keyed
by its node's name, so passing one to the wrong launch file has no effect rather than a
subtle one.

Policy-contract values (observation layout, action scale, gains, joint order, command
bounds) are **not** parameters. They live in `policy_constants.hpp` — one per controller —
because they must match the trained checkpoint exactly. The rugged node cross-checks them
against the ONNX metadata at startup and refuses to run on a mismatch. **The flat node
cannot**: mjlab's exporter is hard-wired to position control and writes flat checkpoints
with an empty metadata map, so the flat node validates only the 45-in/12-out shape and
warns at startup.

## Swapping in a retrained policy

```bash
# rugged
cp <run>/<run>.onnx config/policy/rugged-v2.onnx
/home/antonio/python_venvs/general/bin/python scripts/gen_reference_trace.py \
  --repo /home/antonio/GitHub/quadruped_walking

# flat
cp <run>/<run>.onnx config/policy/flat-feet-time-25.onnx
/home/antonio/python_venvs/general/bin/python scripts/gen_flat_reference_trace.py \
  --repo /home/antonio/GitHub/quadruped_walking --run-dir <run>

colcon test --packages-select go2_rl_velocity
```

Regenerating the reference trace is not optional — the tests are meaningless against a
stale one, and for the flat policy the trace is the *only* contract guard there is. A
rugged retrain that changed the observation terms or history length makes the node refuse
to start; the same change on the flat side is caught only by the regenerated trace.

## Flat-ground controller

```bash
ros2 launch go2_rl_velocity go2_rl_flat.launch.py torque_scale:=0.3
```

Same operator interface, same states, same services, under node name `go2_rl_flat`. What
differs is that **the policy commands joint torque directly**: in `RL` the node sends
`kp = 0`, `kd = 2.0`, `tau = policy`.

That `kd` is load-bearing. Every joint in training carried 2.0 N·m·s/rad of passive viscous
damping that the real Go2 does not have, and a torque policy has no PD loop to hide behind.
The Go2 computes `tau_total = kp(q_des−q) + kd(dq_des−dq) + tau`, so `kd = 2.0` with
`dq_des = 0` reproduces sim's `−2.0·qd` exactly. Removing it lets the same torques produce
far faster joint motion than the policy ever saw.

Standing up is still position controlled. The handover blends `kp` from the stand's
60/80/80 down to zero while `kd` moves to 2.0, then ramps torque in over `torque_ramp_s`.

Treat it as higher-risk than rugged, for reasons that are properties of the training run
rather than of this code. Follow the seven bring-up steps above, plus:

- **Start derated.** `torque_scale:=0.3`, hoisted, before anything else. The flat task
  could not randomize motor strength at all — mjlab's `dr.pd_gains` and `dr.effort_limits`
  both raise on a torque `<motor>` — so the policy has never seen actuator-gain or
  torque-limit uncertainty. That is exactly what you cannot trust on real hardware.
- **Flat indoor floor only.** It trained on a plane and has never seen a slope or a step.
  Its tilt watchdog is correspondingly tighter (45°, not 53°).
- **Expect shuffling at rest.** The flat command sampler never drew a standing-still
  command, so zero velocity is comparatively out of distribution.
- **Do not let it sag.** The flat task terminated on a base height 2.5 cm below nominal, so
  the policy has no recovery from a low pose. The handover must land on `[0, 0.9, -1.8]`,
  which is what `FIXSTAND` targets.

Watchdogs differ too: with no position target to clamp, a sustained torque saturation or a
joint outside mjlab's soft limit band faults to `DAMPING` instead.

## License

MIT. See `LICENSE`.
