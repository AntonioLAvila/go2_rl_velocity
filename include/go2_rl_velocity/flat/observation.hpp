// Copyright (c) 2026 Antonio Avila. MIT License.

#ifndef GO2_RL_VELOCITY__FLAT__OBSERVATION_HPP_
#define GO2_RL_VELOCITY__FLAT__OBSERVATION_HPP_

#include <algorithm>
#include <array>

#include "go2_rl_velocity/flat/policy_constants.hpp"

namespace go2_rl_velocity::flat
{

/// The 45-dim observation, in policy joint order (FL, FR, RL, RR).
///
/// All values are raw SI, exactly as sensed. mjlab applies no scaling and no clipping to
/// any actor term -- the checkpoint's frozen env.yaml has `scale: null` and `clip: null`
/// on all six -- and the observation normalisation the policy needs is baked into the
/// ONNX graph as its leading Sub/Div. So anything done to these numbers here is a bug.
///
/// There is no history buffer, and no equivalent of the rugged controller's
/// ObservationBuilder. Every term in the flat task has `history_length: 0`, so mjlab's
/// per-term stacking degenerates to the identity and this single frame *is* the whole
/// observation. That removes the two conventions most likely to be got wrong on the
/// rugged side -- term-major ordering and the reset backfill -- and leaves term order as
/// the entire layout contract.
///
/// Per-step call order is still load-bearing:
///
///   1. build the frame, whose last_action is a_{t-1}
///   2. flatten()  ->  infer  ->  a_t
///   3. remember a_t as the next step's last_action
struct ObservationFrame
{
  /// q - kDefaultJointPos.
  std::array<float, kNumJoints> joint_pos_rel{};
  /// qd.
  std::array<float, kNumJoints> joint_vel{};
  /// Body-frame angular velocity (IMU gyro).
  std::array<float, 3> base_ang_vel{};
  /// R^T * (0, 0, -1). Points down in the body frame; z is ~-1 when upright.
  std::array<float, 3> projected_gravity{};
  /// The RAW, pre-scale action from the previous control step -- a torque coefficient in
  /// roughly [-1, 1], NOT the N*m the motors were asked for.
  std::array<float, kNumJoints> last_action{};
  /// [vx, vy, wz].
  std::array<float, 3> command{};

  /// Flatten into the observation vector, in kTerms declaration order.
  std::array<float, kObsDim> flatten() const
  {
    std::array<float, kObsDim> out{};
    auto it = out.begin();
    it = std::copy(joint_pos_rel.begin(), joint_pos_rel.end(), it);
    it = std::copy(joint_vel.begin(), joint_vel.end(), it);
    it = std::copy(base_ang_vel.begin(), base_ang_vel.end(), it);
    it = std::copy(projected_gravity.begin(), projected_gravity.end(), it);
    it = std::copy(last_action.begin(), last_action.end(), it);
    std::copy(command.begin(), command.end(), it);
    return out;
  }
};

}  // namespace go2_rl_velocity::flat

#endif  // GO2_RL_VELOCITY__FLAT__OBSERVATION_HPP_
