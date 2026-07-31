// Copyright (c) 2026 Antonio Avila. MIT License.
//
// Pure conversions between what the Go2 reports and what the flat policy expects. Kept
// header-inline and free of any dependency so the parity test exercises exactly the code
// the controller node runs.
//
// The sibling of ../state_conversion.hpp. The gravity and joint-remap helpers are the
// same algebra, re-declared in this namespace rather than shared, so the two controllers
// stay independent. The control law is the part that genuinely differs: actionToTorque
// replaces actionToTarget, and clampToTorqueLimits replaces clampToJointLimits as the
// thing applied to the policy's output.

#ifndef GO2_RL_VELOCITY__FLAT__STATE_CONVERSION_HPP_
#define GO2_RL_VELOCITY__FLAT__STATE_CONVERSION_HPP_

#include <algorithm>
#include <array>
#include <cmath>

#include "go2_rl_velocity/flat/policy_constants.hpp"

namespace go2_rl_velocity::flat
{

/// Gravity direction expressed in the body frame: R^T * (0, 0, -1).
///
/// Equivalent to negating the third row of the rotation matrix. Upright gives
/// (0, 0, -1); the policy's observation normaliser has a z-mean near -1, which is a
/// useful smell test that the sign convention is right.
///
/// `quat` is (w, x, y, z), matching LowState.imu_state.quaternion. It is normalised here
/// because the IMU's is only approximately unit and the error goes straight into an
/// observation the policy is sensitive to.
inline std::array<float, 3> projectedGravity(const std::array<float, 4> & quat)
{
  const double w = quat[0], x = quat[1], y = quat[2], z = quat[3];
  const double norm = std::sqrt(w * w + x * x + y * y + z * z);
  if (!(norm > 1e-6)) {
    // Degenerate quaternion: report "upright" rather than NaN. The orientation watchdog
    // cannot fire on a NaN, so a garbage IMU must not be allowed to produce one.
    return {0.0F, 0.0F, -1.0F};
  }
  const double nw = w / norm, nx = x / norm, ny = y / norm, nz = z / norm;

  return {
    static_cast<float>(2.0 * (nw * ny - nx * nz)),
    static_cast<float>(-2.0 * (nw * nx + ny * nz)),
    static_cast<float>(2.0 * (nx * nx + ny * ny) - 1.0),
  };
}

/// Reorder a per-joint array from Unitree motor order (FR, FL, RR, RL) into policy order
/// (FL, FR, RL, RR).
template<typename T>
inline std::array<T, kNumJoints> unitreeToPolicy(const std::array<T, kNumJoints> & unitree)
{
  std::array<T, kNumJoints> out{};
  for (int i = 0; i < kNumJoints; ++i) {
    out[i] = unitree[kPolicyToUnitree[i]];
  }
  return out;
}

/// Inverse of unitreeToPolicy.
template<typename T>
inline std::array<T, kNumJoints> policyToUnitree(
  const std::array<T, kNumJoints> & policy_order)
{
  std::array<T, kNumJoints> out{};
  for (int i = 0; i < kNumJoints; ++i) {
    out[kPolicyToUnitree[i]] = policy_order[i];
  }
  return out;
}

/// The action-to-torque law: tau = action * effort_limit.
///
/// mjlab's JointEffortActionCfg has offset 0 and `clip: null`, so the deterministic
/// action is used as-is, exactly as in training. Saturation is applied separately, by the
/// caller, via clampToTorqueLimits -- mirroring MuJoCo, where the action term does no
/// clipping and the <motor ctrlrange> does it downstream.
inline std::array<float, kNumJoints> actionToTorque(
  const std::array<float, kActionDim> & action)
{
  std::array<float, kNumJoints> torque{};
  for (int i = 0; i < kNumJoints; ++i) {
    torque[i] = static_cast<float>(static_cast<double>(action[i]) * kTorqueScale[i]);
  }
  return torque;
}

/// Clamp joint torques to the actuators' effort limits, in policy order.
/// Returns true if any element had to be clamped.
///
/// Stands in for MuJoCo's <motor ctrlrange>, which is what bounded the policy's authority
/// in training. The Go2's motor boards apply their own limit as well; doing it here means
/// the value we publish is the value that gets applied, so the saturation watchdog sees
/// the truth.
inline bool clampToTorqueLimits(std::array<float, kNumJoints> & torque)
{
  bool clamped = false;
  for (int i = 0; i < kNumJoints; ++i) {
    const auto limit = static_cast<float>(kTorqueLimit[i]);
    const float original = torque[i];
    torque[i] = std::clamp(torque[i], -limit, limit);
    if (torque[i] != original) {
      clamped = true;
    }
  }
  return clamped;
}

/// Clamp joint position targets to the Go2's mechanical limits, in policy order.
/// Returns true if any element had to be clamped.
///
/// Only the scripted stand uses this -- the RL state commands torque and has no position
/// target to clamp.
inline bool clampToJointLimits(std::array<float, kNumJoints> & target)
{
  bool clamped = false;
  for (int i = 0; i < kNumJoints; ++i) {
    const auto lo = static_cast<float>(kJointPosLower[i]);
    const auto hi = static_cast<float>(kJointPosUpper[i]);
    const float original = target[i];
    target[i] = std::clamp(target[i], lo, hi);
    if (target[i] != original) {
      clamped = true;
    }
  }
  return clamped;
}

/// True if any measured joint is outside the soft limit band mjlab trained inside.
///
/// The torque-control replacement for the rugged controller's clamp watchdog: nothing
/// stops a torque policy driving a joint into its hard stop, so the excursion has to be
/// detected on the measured position instead of prevented on the target.
inline bool outsideSoftJointLimits(const std::array<float, kNumJoints> & joint_pos)
{
  for (int i = 0; i < kNumJoints; ++i) {
    if (joint_pos[i] < static_cast<float>(softJointPosLower(i)) ||
      joint_pos[i] > static_cast<float>(softJointPosUpper(i)))
    {
      return true;
    }
  }
  return false;
}

}  // namespace go2_rl_velocity::flat

#endif  // GO2_RL_VELOCITY__FLAT__STATE_CONVERSION_HPP_
