// Copyright (c) 2026 Antonio Avila. MIT License.

#include "go2_rl_velocity/controller_fsm.hpp"

#include <algorithm>
#include <cmath>

#include "go2_rl_velocity/state_conversion.hpp"

namespace go2_rl_velocity
{

namespace
{

/// Smooth 0->1 interpolation. Zero velocity at both ends, so the scripted stand does not
/// start or stop with a step in commanded velocity.
float smoothStep(double t)
{
  const double x = std::clamp(t, 0.0, 1.0);
  return static_cast<float>(x * x * (3.0 - 2.0 * x));
}

template<typename A, typename B>
std::array<float, policy::kNumJoints> lerp(const A & from, const B & to, float alpha)
{
  std::array<float, policy::kNumJoints> out{};
  for (int i = 0; i < policy::kNumJoints; ++i) {
    out[i] = static_cast<float>(from[i]) +
      alpha * (static_cast<float>(to[i]) - static_cast<float>(from[i]));
  }
  return out;
}

}  // namespace

const char * toString(ControllerState state)
{
  switch (state) {
    case ControllerState::Passive: return "PASSIVE";
    case ControllerState::FixStand: return "FIXSTAND";
    case ControllerState::Rl: return "RL";
    case ControllerState::Damping: return "DAMPING";
  }
  return "UNKNOWN";
}

ControllerFsm::ControllerFsm(const Config & config, PolicyRunner & policy)
: config_(config), policy_(policy)
{
  setDampingGains();
}

void ControllerFsm::setGains(
  const std::array<double, policy::kNumJoints> & kp,
  const std::array<double, policy::kNumJoints> & kd)
{
  for (int i = 0; i < policy::kNumJoints; ++i) {
    command_.kp[i] = static_cast<float>(kp[i]);
    command_.kd[i] = static_cast<float>(kd[i]);
  }
}

void ControllerFsm::setDampingGains()
{
  command_.kp.fill(0.0F);
  command_.kd.fill(static_cast<float>(policy::kDampingKd));
}

void ControllerFsm::holdPosition(const RobotObservation & obs)
{
  // With kp = 0 the target is ignored by the motor boards, but tracking the measured
  // position keeps the published command meaningful for logging and makes the handover
  // into a stiff state continuous.
  command_.joint_target = obs.joint_pos;
}

void ControllerFsm::enter(ControllerState next, double now_s)
{
  state_ = next;
  state_entered_s_ = now_s;
  entered_ = true;

  switch (next) {
    case ControllerState::Passive:
    case ControllerState::Damping:
      setDampingGains();
      break;
    case ControllerState::FixStand:
      setGains(policy::kStandKp, policy::kStandKd);
      stand_start_captured_ = false;
      break;
    case ControllerState::Rl:
      // Gains are blended in over gain_ramp_s; the policy does not engage until that is
      // done. Slamming from the stand gains straight to the policy's much softer ones
      // drops the robot several centimetres before the first action lands.
      policy_engaged_ = false;
      clamp_streak_ = 0;
      inference_overrun_streak_ = 0;
      last_action_.fill(0.0F);
      command_filtered_.fill(0.0F);
      // Never inherit history from a previous RL session.
      observation_ = ObservationBuilder{};
      break;
  }
}

bool ControllerFsm::requestState(ControllerState target, double now_s, std::string & reason)
{
  if (target == state_) {
    reason = "already in that state";
    return false;
  }
  // The only guarded transition: the policy expects to start from its default pose, so
  // it can only be entered from a completed stand.
  if (target == ControllerState::Rl && state_ != ControllerState::FixStand) {
    reason = "RL can only be entered from FIXSTAND; stand the robot first";
    return false;
  }
  enter(target, now_s);
  return true;
}

void ControllerFsm::fault(const std::string & reason, double now_s)
{
  if (state_ == ControllerState::Damping) {
    return;
  }
  fault_reason_ = reason;
  enter(ControllerState::Damping, now_s);
}

bool ControllerFsm::checkCommonWatchdogs(const RobotObservation & obs, double now_s)
{
  if (!obs.lowstate_fresh) {
    fault("lowstate went stale", now_s);
    return true;
  }

  // Orientation. Only meaningful once the robot is meant to be upright -- in Passive it
  // is normally lying down and in Damping it is on its way there.
  if (state_ == ControllerState::FixStand || state_ == ControllerState::Rl) {
    const auto gravity = projectedGravity(obs.imu_quaternion);
    if (gravity[2] > static_cast<float>(config_.tilt_limit)) {
      fault("tilted past the orientation limit", now_s);
      return true;
    }
  }

  // E-stop, from any state.
  if (remote_.chordPressed(RemoteButton::L2, RemoteButton::B)) {
    fault("operator e-stop (L2+B)", now_s);
    return true;
  }
  return false;
}

void ControllerFsm::step(const RobotObservation & obs, double now_s)
{
  if (!entered_) {
    enter(state_, now_s);
  }
  remote_.update(obs.remote);

  if (checkCommonWatchdogs(obs, now_s)) {
    stepDamping(obs, now_s);
    return;
  }

  switch (state_) {
    case ControllerState::Passive: stepPassive(obs, now_s); break;
    case ControllerState::FixStand: stepFixStand(obs, now_s); break;
    case ControllerState::Rl: stepRl(obs, now_s); break;
    case ControllerState::Damping: stepDamping(obs, now_s); break;
  }
}

void ControllerFsm::stepPassive(const RobotObservation & obs, double now_s)
{
  setDampingGains();
  holdPosition(obs);

  if (remote_.chordPressed(RemoteButton::L2, RemoteButton::Up)) {
    enter(ControllerState::FixStand, now_s);
  }
}

void ControllerFsm::stepFixStand(const RobotObservation & obs, double now_s)
{
  if (!stand_start_captured_) {
    // Captured here rather than in enter() so it reflects the pose at the first control
    // step of the state, whichever way the state was entered.
    stand_start_pos_ = obs.joint_pos;
    stand_start_captured_ = true;
    state_entered_s_ = now_s;
  }
  const double elapsed = now_s - state_entered_s_;

  setGains(policy::kStandKp, policy::kStandKd);

  // Two segments: wherever the legs are now -> a tucked crouch -> the policy's default
  // pose. Going straight to the default pose from a sprawl drags the feet.
  if (elapsed < config_.stand_segment_s) {
    const float alpha = smoothStep(elapsed / config_.stand_segment_s);
    command_.joint_target = lerp(stand_start_pos_, policy::kCrouchJointPos, alpha);
  } else {
    const float alpha =
      smoothStep((elapsed - config_.stand_segment_s) / config_.stand_segment_s);
    command_.joint_target = lerp(policy::kCrouchJointPos, policy::kDefaultJointPos, alpha);
  }

  const bool stand_complete = elapsed >= 2.0 * config_.stand_segment_s;
  if (stand_complete && remote_.chordPressed(RemoteButton::R2, RemoteButton::A)) {
    enter(ControllerState::Rl, now_s);
  }
}

void ControllerFsm::updateVelocityCommand(const RobotObservation & obs, bool warming_up)
{
  std::array<float, 3> target{};
  if (!warming_up) {
    if (config_.use_joystick_command) {
      // Sticks map to the full command envelope. Signs are configurable because the
      // remote's convention is undocumented -- verify with remote_debug before driving.
      target = {
        static_cast<float>(obs.remote.ly * config_.joystick_sign[0] *
        policy::kCommandBounds[0]),
        static_cast<float>(obs.remote.lx * config_.joystick_sign[1] *
        policy::kCommandBounds[1]),
        static_cast<float>(obs.remote.rx * config_.joystick_sign[2] *
        policy::kCommandBounds[2]),
      };
    } else if (obs.external_command_fresh) {
      target = obs.external_command;
    }
    // A stale /cmd_vel decays the command to zero rather than faulting: losing the
    // operator's input should stop the robot, not drop it.
  }

  for (int i = 0; i < 3; ++i) {
    const auto bound = static_cast<float>(policy::kCommandBounds[i]);
    target[i] = std::clamp(target[i], -bound, bound);
    if (!std::isfinite(target[i])) {
      target[i] = 0.0F;
    }

    // Slew limit, so a flicked stick cannot ask for a step change in velocity.
    const auto max_delta =
      static_cast<float>(config_.command_accel_limit[i] * policy::kControlDt);
    const float delta = std::clamp(target[i] - command_filtered_[i], -max_delta, max_delta);
    command_filtered_[i] += delta;
  }
}

void ControllerFsm::stepRl(const RobotObservation & obs, double now_s)
{
  const double elapsed = now_s - state_entered_s_;

  if (!policy_engaged_) {
    // Blend stand gains into policy gains while holding the default pose.
    const float alpha = smoothStep(elapsed / config_.gain_ramp_s);
    std::array<double, policy::kNumJoints> kp{};
    std::array<double, policy::kNumJoints> kd{};
    for (int i = 0; i < policy::kNumJoints; ++i) {
      kp[i] = policy::kStandKp[i] + alpha * (policy::kPolicyKp[i] - policy::kStandKp[i]);
      kd[i] = policy::kStandKd[i] + alpha * (policy::kPolicyKd[i] - policy::kStandKd[i]);
    }
    setGains(kp, kd);
    for (int i = 0; i < policy::kNumJoints; ++i) {
      command_.joint_target[i] = static_cast<float>(policy::kDefaultJointPos[i]);
    }

    if (elapsed >= config_.gain_ramp_s) {
      policy_engaged_ = true;
      policy_engaged_s_ = now_s;
      last_action_.fill(0.0F);
      command_filtered_.fill(0.0F);
      // The policy takes over on the *next* step, whose frame becomes the one every
      // history slot is backfilled with -- mjlab's post-reset behaviour.
    }
    return;
  }

  setGains(policy::kPolicyKp, policy::kPolicyKd);

  const bool warming_up =
    (now_s - policy_engaged_s_) < config_.command_warmup_s;
  updateVelocityCommand(obs, warming_up);

  ObservationFrame frame;
  for (int i = 0; i < policy::kNumJoints; ++i) {
    frame.joint_pos_rel[i] =
      obs.joint_pos[i] - static_cast<float>(policy::kDefaultJointPos[i]);
    frame.joint_vel[i] = obs.joint_vel[i];
  }
  frame.base_ang_vel = obs.base_ang_vel;
  frame.projected_gravity = projectedGravity(obs.imu_quaternion);
  frame.last_action = last_action_;
  frame.command = command_filtered_;

  if (!observation_.initialised()) {
    observation_.reset(frame);
  } else {
    observation_.push(frame);
  }

  const auto & action = policy_.run(observation_.pack());

  if (policy_.last_inference_time_s() > config_.inference_warn_s) {
    if (++inference_overrun_streak_ >= config_.inference_overrun_fault_steps) {
      fault("policy inference is persistently over budget", now_s);
      return;
    }
  } else {
    inference_overrun_streak_ = 0;
  }

  for (const float a : action) {
    if (!std::isfinite(a)) {
      fault("policy produced a non-finite action", now_s);
      return;
    }
  }

  auto target = actionToTarget(action);
  if (clampToJointLimits(target)) {
    // One clamped step is a policy pushing into a limit, which happens. Many in a row
    // means it is fighting the hardware and will not recover on its own.
    if (++clamp_streak_ >= config_.joint_clamp_fault_steps) {
      fault("joint targets clamped for too many consecutive steps", now_s);
      return;
    }
  } else {
    clamp_streak_ = 0;
  }

  command_.joint_target = target;
  last_action_ = action;
}

void ControllerFsm::stepDamping(const RobotObservation & obs, double now_s)
{
  setDampingGains();
  holdPosition(obs);

  // The observation history must not carry across a fault into the next RL entry.
  observation_ = ObservationBuilder{};
  policy_engaged_ = false;

  if (now_s - state_entered_s_ >= config_.damping_settle_s) {
    enter(ControllerState::Passive, now_s);
  }
}

}  // namespace go2_rl_velocity
