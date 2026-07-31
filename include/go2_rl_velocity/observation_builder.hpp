// Copyright (c) 2026 Antonio Avila. MIT License.

#ifndef GO2_RL_VELOCITY__OBSERVATION_BUILDER_HPP_
#define GO2_RL_VELOCITY__OBSERVATION_BUILDER_HPP_

#include <array>

#include "go2_rl_velocity/policy_constants.hpp"

namespace go2_rl_velocity
{

/// One 45-dim observation frame, in policy joint order (FL, FR, RL, RR).
///
/// All values are raw SI, exactly as sensed. mjlab applies no scaling and no clipping to
/// any actor term, and the observation normalisation the policy needs is baked into the
/// ONNX graph itself -- so anything done to these numbers here is a bug.
struct ObservationFrame
{
  /// q - kDefaultJointPos.
  std::array<float, policy::kNumJoints> joint_pos_rel{};
  /// qd.
  std::array<float, policy::kNumJoints> joint_vel{};
  /// Body-frame angular velocity (IMU gyro).
  std::array<float, 3> base_ang_vel{};
  /// R^T * (0, 0, -1). Points down in the body frame; z is ~-1 when upright.
  std::array<float, 3> projected_gravity{};
  /// The RAW, pre-scale action from the previous control step.
  std::array<float, policy::kNumJoints> last_action{};
  /// [vx, vy, wz].
  std::array<float, 3> command{};

  /// Flatten into the 45-dim frame, in kTerms declaration order.
  std::array<float, policy::kFrameDim> flatten() const;
};

/// Rolling history of the last kHistoryLength frames, packed into the flat observation
/// vector the policy expects.
///
/// Term-major, oldest to newest -- see kTerms in policy_constants.hpp. The current frame
/// is included: the vector the policy reads at time t covers [t-4 .. t].
///
/// Per-step call order is load-bearing:
///
///   1. build frame_t, whose last_action is a_{t-1}
///   2. push(frame_t)
///   3. pack()  ->  infer  ->  a_t
///   4. remember a_t as the next step's last_action
///
/// Packing before pushing would leave the policy one frame stale; scripts/verify_rugged.py
/// documents the same off-by-one on the training side.
class ObservationBuilder
{
public:
  ObservationBuilder();

  /// Fill every history slot with `frame`.
  ///
  /// This is what mjlab does after a reset -- history is backfilled with the first frame,
  /// never zero-padded. Must be called when entering the policy state, otherwise the
  /// policy's first four steps see a phantom transient from an all-zero past.
  void reset(const ObservationFrame & frame);

  /// Append a frame, evicting the oldest.
  void push(const ObservationFrame & frame);

  /// Flat observation, term-major and oldest-first. Valid until the next push/reset.
  const std::array<float, policy::kObsDim> & pack();

  /// True once reset() or push() has been called at least once.
  bool initialised() const {return initialised_;}

private:
  /// Ring of frames. `head_` is the index of the OLDEST frame.
  std::array<std::array<float, policy::kFrameDim>, policy::kHistoryLength> frames_{};
  std::size_t head_{0};
  bool initialised_{false};

  std::array<float, policy::kObsDim> packed_{};
};

}  // namespace go2_rl_velocity

#endif  // GO2_RL_VELOCITY__OBSERVATION_BUILDER_HPP_
