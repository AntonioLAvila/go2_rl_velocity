// Copyright (c) 2026 Antonio Avila. MIT License.

#ifndef GO2_RL_VELOCITY__CONTROLLER_FSM_HPP_
#define GO2_RL_VELOCITY__CONTROLLER_FSM_HPP_

#include <array>
#include <string>

#include "go2_rl_velocity/observation_builder.hpp"
#include "go2_rl_velocity/policy_constants.hpp"
#include "go2_rl_velocity/policy_runner.hpp"
#include "go2_rl_velocity/unitree_remote.hpp"

namespace go2_rl_velocity
{

enum class ControllerState
{
  /// Limp, damping only. Startup state and the resting place after a fault.
  Passive,
  /// Scripted interpolation up to the policy's default pose.
  FixStand,
  /// The policy is driving.
  Rl,
  /// Soft collapse after an e-stop or a watchdog trip.
  Damping,
};

const char * toString(ControllerState state);

/// Everything the FSM needs from one LowState, already converted to policy conventions
/// and policy joint order.
struct RobotObservation
{
  std::array<float, policy::kNumJoints> joint_pos{};
  std::array<float, policy::kNumJoints> joint_vel{};
  std::array<float, 3> base_ang_vel{};
  std::array<float, 4> imu_quaternion{};   ///< w, x, y, z
  RemoteState remote{};
  /// The operator's velocity command from /cmd_vel, used when joystick input is off.
  std::array<float, 3> external_command{};
  bool external_command_fresh{false};
  /// False when LowState has gone stale; the FSM faults on it.
  bool lowstate_fresh{true};
};

/// Joint-space command, in policy joint order. The node remaps to Unitree motor order.
struct MotorCommand
{
  std::array<float, policy::kNumJoints> joint_target{};
  std::array<float, policy::kNumJoints> kp{};
  std::array<float, policy::kNumJoints> kd{};
};

/// The operator-facing state machine.
///
/// Mirrors Unitree's own deploy FSM: a scripted stand, a policy state, and two limp
/// states, with the wireless remote driving transitions.
///
/// step() is called at the policy rate (50 Hz); the node holds and republishes its command
/// at a much higher rate, because the motor boards need feeding continuously while the
/// policy recomputes every 20 ms. That split is what mjlab's decimation models in
/// training, so it is not merely convenient.
class ControllerFsm
{
public:
  struct Config
  {
    /// Duration of each of the two stand-up segments (measured -> crouch -> default).
    double stand_segment_s{1.0};
    /// Time spent blending stand gains into policy gains before the policy engages.
    double gain_ramp_s{0.5};
    /// Command is pinned to zero for this long after the policy engages, matching
    /// CMD_WARMUP_S in the training repo's sim-to-sim check.
    double command_warmup_s{0.5};
    /// Upper bound on projected_gravity z. Upright is -1; -0.6 is about 53 degrees.
    double tilt_limit{-0.6};
    /// Consecutive clamped steps tolerated before faulting.
    int joint_clamp_fault_steps{25};
    /// Per-step inference budget. Exceeding it repeatedly is a fault.
    double inference_warn_s{0.010};
    int inference_overrun_fault_steps{10};
    /// Seconds spent limp in Damping before dropping back to Passive.
    double damping_settle_s{1.5};
    /// Velocity command acceleration limits (m/s^2, m/s^2, rad/s^2).
    std::array<double, 3> command_accel_limit{{3.0, 3.0, 6.0}};
    /// Joystick axis signs, so a remote that disagrees with the assumed convention can
    /// be corrected without a rebuild. See README.md.
    std::array<double, 3> joystick_sign{{1.0, -1.0, -1.0}};
    /// Take the velocity command from the remote's sticks rather than /cmd_vel.
    bool use_joystick_command{true};
  };

  ControllerFsm(const Config & config, PolicyRunner & policy);

  /// Advance one control step. `now_s` is a monotonic clock reading in seconds.
  void step(const RobotObservation & obs, double now_s);

  /// The command to send. Valid after the first step().
  const MotorCommand & command() const {return command_;}

  ControllerState state() const {return state_;}

  /// Why the controller last faulted; empty if it has not.
  const std::string & faultReason() const {return fault_reason_;}

  /// The velocity command currently being fed to the policy, after clamping and slewing.
  const std::array<float, 3> & activeCommand() const {return command_filtered_;}

  /// Ask for a transition from outside the remote (a ROS service). Rejected, with a
  /// reason, when the requested transition is not one the remote could make either --
  /// notably jumping straight into Rl without a stand.
  bool requestState(ControllerState target, double now_s, std::string & reason);

  /// Force a fault. Used by the node's e-stop service.
  void fault(const std::string & reason, double now_s);

private:
  void enter(ControllerState next, double now_s);
  void stepPassive(const RobotObservation & obs, double now_s);
  void stepFixStand(const RobotObservation & obs, double now_s);
  void stepRl(const RobotObservation & obs, double now_s);
  void stepDamping(const RobotObservation & obs, double now_s);

  /// Watchdogs that apply in every state. Returns true if a fault was raised.
  bool checkCommonWatchdogs(const RobotObservation & obs, double now_s);

  /// Resolve, clamp and slew-limit the operator's velocity command.
  void updateVelocityCommand(const RobotObservation & obs, bool warming_up);

  void setGains(
    const std::array<double, policy::kNumJoints> & kp,
    const std::array<double, policy::kNumJoints> & kd);
  void setDampingGains();
  void holdPosition(const RobotObservation & obs);

  Config config_;
  PolicyRunner & policy_;

  ControllerState state_{ControllerState::Passive};
  double state_entered_s_{0.0};
  bool entered_{false};
  std::string fault_reason_;

  MotorCommand command_{};

  // FixStand. The starting pose is whatever the legs happen to be in when the state is
  // entered, captured on the first step rather than at entry so that a transition
  // requested by service gets the same treatment as one from the remote.
  std::array<float, policy::kNumJoints> stand_start_pos_{};
  bool stand_start_captured_{false};

  // Rl.
  ObservationBuilder observation_;
  std::array<float, policy::kActionDim> last_action_{};
  bool policy_engaged_{false};
  double policy_engaged_s_{0.0};
  int clamp_streak_{0};
  int inference_overrun_streak_{0};

  std::array<float, 3> command_filtered_{};

  UnitreeRemote remote_;
};

}  // namespace go2_rl_velocity

#endif  // GO2_RL_VELOCITY__CONTROLLER_FSM_HPP_
