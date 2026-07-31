# go2_rl_velocity

Runs a reinforcement-learning locomotion policy on a Unitree Go2 as a ROS 2 Jazzy node.

The policy is a blind, proprioceptive velocity-tracking controller trained in
[mjlab](https://github.com/mujocolab/mjlab) with rsl_rl PPO and exported to ONNX. It needs
no elevation map and no base-velocity estimate — joint encoders, the IMU and the
operator's command are enough — which is what makes it deployable on hardware.

Deployment target: **Jetson Orin Nano, JetPack 7.2**, driving the robot through the
`unitree_ros2` low-level bridge.

---

## Requirements

| | |
|---|---|
| ROS 2 | Jazzy |
| ONNX Runtime | ≥ 1.15 (the policy is opset 18). C++ API, CPU build is sufficient. |
| `unitree_ros2` | Provides the `unitree_go` messages and the DDS bridge to the robot. |

`unitree_go` has no rosdep key — build [unitree_ros2](https://github.com/unitreerobotics/unitree_ros2)
and source it as an underlay. Its patched CycloneDDS is the fiddliest part of the setup;
validate it standalone (`ros2 topic echo /lowstate`) before wiring this node in.

## Build

```bash
cd ~/ros2_ws
source /opt/ros/jazzy/setup.bash
source /path/to/unitree_ros2/install/setup.bash      # provides unitree_go

colcon build --packages-select go2_rl_velocity \
  --cmake-args -DONNXRUNTIME_ROOT=/opt/onnxruntime
```

`ONNXRUNTIME_ROOT` can be omitted if ONNX Runtime is installed under `/usr` or
`/usr/local`; it can also be given as an environment variable.

If `unitree_go` is not on the prefix path the build **skips the controller node** and
builds only the policy core library and its tests. This is intentional — it lets the
observation/inference pipeline be tested on a workstation. A CMake warning names what was
missing.

## Test

```bash
colcon test --packages-select go2_rl_velocity
colcon test-result --verbose
```

The important one is `test_policy_parity`: it replays a recorded trace whose expected
values come from the *training repo's own* observation packer, and proves this package
reproduces it exactly. See `CLAUDE.md` for what that covers and why it is built that way.

## Run

```bash
ros2 launch go2_rl_velocity go2_rl_velocity.launch.py
```

Put the robot in low-level mode first (release sport mode), lying on a flat surface.

| Input | Effect |
|---|---|
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
|---|---|---|
| `/lowstate` | `unitree_go/msg/LowState` | in |
| `/lowcmd` | `unitree_go/msg/LowCmd` | out, 500 Hz |
| `/cmd_vel` | `geometry_msgs/msg/Twist` | in, optional |
| `~/status` | `std_msgs/msg/String` | out, 2 Hz |

Command limits: `vx ±2.0 m/s`, `vy ±1.0 m/s`, `wz ±1.5 rad/s`.

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

`config/go2_rl_velocity.yaml` holds deployment behaviour — rates, timeouts, the operator
interface. Every parameter is commented there.

Policy-contract values (observation layout, action scale, PD gains, joint order, command
bounds) are **not** parameters. They live in `include/go2_rl_velocity/policy_constants.hpp`
because they must match the trained checkpoint exactly, and are cross-checked against the
ONNX metadata at startup: a mismatch refuses to start rather than walking badly.

## Swapping in a retrained policy

```bash
cp <run>/<run>.onnx config/policy/rugged-v2.onnx
/home/antonio/python_venvs/general/bin/python scripts/gen_reference_trace.py \
  --repo /home/antonio/GitHub/quadruped_walking
colcon test --packages-select go2_rl_velocity
```

Regenerating the reference trace is not optional — the tests are meaningless against a
stale one. If the retrain changed the observation terms or the history length, the node
will refuse to start with an explicit contract error.

## License

MIT. See `LICENSE`.
