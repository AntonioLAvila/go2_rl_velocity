// Copyright (c) 2026 Antonio Avila. MIT License.
//
// Pure conversions between what the Go2 reports and what the policy expects. Kept
// header-inline and free of any dependency so the parity test exercises exactly the code
// the controller node runs.

#ifndef GO2_RL_VELOCITY__STATE_CONVERSION_HPP_
#define GO2_RL_VELOCITY__STATE_CONVERSION_HPP_

#include <algorithm>
#include <array>
#include <cmath>

#include "go2_rl_velocity/policy_constants.hpp"

namespace go2_rl_velocity
{

/// Gravity direction expressed in the body frame: R^T * (0, 0, -1).
///
/// Equivalent to negating the third row of the rotation matrix. Upright gives
/// (0, 0, -1); the policy's observation normaliser has a z-mean of about -0.996, which is
/// a useful smell test that the sign convention is right.
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
inline std::array<T, policy::kNumJoints> unitreeToPolicy(
  const std::array<T, policy::kNumJoints> & unitree)
{
  std::array<T, policy::kNumJoints> out{};
  for (int i = 0; i < policy::kNumJoints; ++i) {
    out[i] = unitree[policy::kPolicyToUnitree[i]];
  }
  return out;
}

/// Inverse of unitreeToPolicy.
template<typename T>
inline std::array<T, policy::kNumJoints> policyToUnitree(
  const std::array<T, policy::kNumJoints> & policy_order)
{
  std::array<T, policy::kNumJoints> out{};
  for (int i = 0; i < policy::kNumJoints; ++i) {
    out[policy::kPolicyToUnitree[i]] = policy_order[i];
  }
  return out;
}

/// The action-to-target law: target = action * scale + default_pose.
///
/// mjlab's JointPositionActionCfg(use_default_offset=True). Actions are NOT clipped --
/// the policy's deterministic output is used as-is, exactly as in training. Joint limits
/// are enforced separately, on the target, by the caller.
inline std::array<float, policy::kNumJoints> actionToTarget(
  const std::array<float, policy::kActionDim> & action)
{
  std::array<float, policy::kNumJoints> target{};
  for (int i = 0; i < policy::kNumJoints; ++i) {
    target[i] = static_cast<float>(
      static_cast<double>(action[i]) * policy::kActionScale[i] + policy::kDefaultJointPos[i]);
  }
  return target;
}

/// Clamp joint position targets to the Go2's mechanical limits, in policy order.
/// Returns true if any element had to be clamped.
inline bool clampToJointLimits(std::array<float, policy::kNumJoints> & target)
{
  bool clamped = false;
  for (int i = 0; i < policy::kNumJoints; ++i) {
    const auto lo = static_cast<float>(policy::kJointPosLower[i]);
    const auto hi = static_cast<float>(policy::kJointPosUpper[i]);
    const float original = target[i];
    target[i] = std::clamp(target[i], lo, hi);
    if (target[i] != original) {
      clamped = true;
    }
  }
  return clamped;
}

/// Map a stick deflection in [-1, 1] onto one axis of the command envelope.
///
/// Full deflection reaches the envelope's edge in *that direction*, so on an asymmetric
/// axis the two halves of the stick have different gains: vx is 3.0 m/s at full forward
/// and 1.0 m/s at full back. Scaling both halves by the larger bound and clamping instead
/// would put the whole reverse range in the first third of the stick's travel.
inline float stickToCommand(double stick, int axis)
{
  if (!std::isfinite(stick)) {
    return 0.0F;
  }
  const double s = std::clamp(stick, -1.0, 1.0);
  return static_cast<float>(
    s >= 0.0 ? s * policy::kCommandUpper[axis] : -s * policy::kCommandLower[axis]);
}

/// Clamp one axis of a velocity command to the envelope the policy trained on. A
/// non-finite request becomes zero: "stop" is the only safe reading of a garbage command.
inline float clampCommand(float value, int axis)
{
  if (!std::isfinite(value)) {
    return 0.0F;
  }
  return std::clamp(
    value, static_cast<float>(policy::kCommandLower[axis]),
    static_cast<float>(policy::kCommandUpper[axis]));
}

}  // namespace go2_rl_velocity

#endif  // GO2_RL_VELOCITY__STATE_CONVERSION_HPP_
