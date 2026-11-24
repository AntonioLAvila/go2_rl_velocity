// Copyright (c) 2026 Antonio Avila. MIT License.

#ifndef GO2_RL_VELOCITY__FLAT__CONTROLLER_FSM_HPP_
#define GO2_RL_VELOCITY__FLAT__CONTROLLER_FSM_HPP_

#include <array>
#include <string>

#include "go2_rl_velocity/flat/observation.hpp"
#include "go2_rl_velocity/flat/policy_constants.hpp"
#include "go2_rl_velocity/flat/policy_runner.hpp"
#include "go2_rl_velocity/unitree_remote.hpp"

namespace go2_rl_velocity::flat
{

using go2_rl_velocity::RemoteButton;
using go2_rl_velocity::RemoteState;
using go2_rl_velocity::UnitreeRemote;

enum class ControllerState
{
  /// Limp, damping only. Startup state and the resting place after a fault.
  Passive,
  /// Scripted position-controlled interpolation up to the policy's default pose.
  FixStand,
  /// The policy is driving, in torque.
  Rl,
  /// Soft collapse after an e-stop or a watchdog trip.
  Damping,
};

const char * toString(ControllerState state);

/// Everything the FSM needs from one LowState, already converted to policy conventions
/// and policy joint order.
struct RobotObservation
{
  std::array<float, kNumJoints> joint_pos{};
  std::array<float, kNumJoints> joint_vel{};
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
///
/// Unlike the rugged controller's MotorCommand this carries a feed-forward torque. The
/// motor board computes `kp*(joint_target - q) + kd*(0 - qd) + tau`, so all four fields
/// are live at once and the two authorities blend smoothly instead of switching -- which
/// is what makes the position-to-torque handover safe.
struct MotorCommand
{
  std::array<float, kNumJoints> joint_target{};
  std::array<float, kNumJoints> kp{};
  std::array<float, kNumJoints> kd{};
  /// Feed-forward joint torque, N*m.
  std::array<float, kNumJoints> tau{};
};

/// The operator-facing state machine for the flat-ground torque policy.
///
/// Mirrors the rugged controller's ControllerFsm -- same four states, same remote chords,
/// same scripted stand -- differing only where the control mode forces it to:
///
///   * RL commands torque with kp = 0, so there is no position target to clamp. Joint
///     protection watches the measured position instead.
///   * The handover blends the stand's position authority out and the policy's torque in.
///   * kd does not go to zero in RL: it reproduces the passive joint damping the policy
///     was trained with. See kPolicyKd in policy_constants.hpp.
///
/// step() is called at the policy rate (200 Hz); the node holds and republishes its
/// command at a higher rate, the same zero-order hold mjlab's decimation applies.
class ControllerFsm
{
public:
  struct Config
  {
    /// Duration of each of the two stand-up segments (measured -> crouch -> default).
    double stand_segment_s{1.0};
    /// Time spent blending the stand's position authority out before the policy engages.
    double gain_ramp_s{0.5};
    /// Time spent ramping the policy's torque in once it has engaged. The gain ramp above
    /// hands over authority; this one keeps the first torque from arriving as a step.
    double torque_ramp_s{0.2};
    /// Command is pinned to zero for this long after the policy engages.
    double command_warmup_s{0.5};
    /// Global derate on the policy's torque, for bring-up. 1.0 is the trained authority.
    double torque_scale{1.0};
    /// kd held in the RL state, emulating the passive joint damping the policy trained
    /// with. Defaults to the contract value in kPolicyKd; exposed so it can be dialled on
    /// the bench without a rebuild. Zero means pure torque, which the policy never saw.
    double passive_damping_kd{kPolicyKd[0]};
    /// Upper bound on projected_gravity z. Upright is -1; the flat policy was terminated
    /// in training at pi/4, i.e. -0.7071.
    double tilt_limit{kTiltLimit};
    /// Consecutive steps with a saturated torque tolerated before faulting.
    int torque_saturation_fault_steps{25};
    /// Consecutive steps with a joint outside the soft limit band before faulting.
    int joint_excursion_fault_steps{10};
    /// Per-step inference budget. Exceeding it repeatedly is a fault.
    double inference_warn_s{0.002};
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
    const std::array<double, kNumJoints> & kp,
    const std::array<double, kNumJoints> & kd);
  void setDampingGains();
  void holdPosition(const RobotObservation & obs);
  /// Zero the feed-forward torque. Every non-RL state must do this explicitly -- a stale
  /// torque left in the held command would keep being republished at 500 Hz.
  void clearTorque();

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
  std::array<float, kNumJoints> stand_start_pos_{};
  bool stand_start_captured_{false};

  // Rl.
  std::array<float, kActionDim> last_action_{};
  bool policy_engaged_{false};
  double policy_engaged_s_{0.0};
  int torque_saturation_streak_{0};
  int joint_excursion_streak_{0};
  int inference_overrun_streak_{0};

  std::array<float, 3> command_filtered_{};

  UnitreeRemote remote_;
};

}  // namespace go2_rl_velocity::flat

#endif  // GO2_RL_VELOCITY__FLAT__CONTROLLER_FSM_HPP_
