// Copyright (c) 2026 Antonio Avila. MIT License.

#include "go2_rl_velocity/observation_builder.hpp"

#include <algorithm>

namespace go2_rl_velocity
{

using policy::kFrameDim;
using policy::kHistoryLength;
using policy::kTerms;

std::array<float, kFrameDim> ObservationFrame::flatten() const
{
  std::array<float, kFrameDim> out{};
  auto it = out.begin();
  it = std::copy(joint_pos_rel.begin(), joint_pos_rel.end(), it);
  it = std::copy(joint_vel.begin(), joint_vel.end(), it);
  it = std::copy(base_ang_vel.begin(), base_ang_vel.end(), it);
  it = std::copy(projected_gravity.begin(), projected_gravity.end(), it);
  it = std::copy(last_action.begin(), last_action.end(), it);
  it = std::copy(command.begin(), command.end(), it);
  return out;
}

ObservationBuilder::ObservationBuilder() = default;

void ObservationBuilder::reset(const ObservationFrame & frame)
{
  const auto flat = frame.flatten();
  frames_.fill(flat);
  head_ = 0;
  initialised_ = true;
}

void ObservationBuilder::push(const ObservationFrame & frame)
{
  if (!initialised_) {
    // A push before reset would leave four zero frames in the history. Treat it as a
    // reset so the caller cannot accidentally feed the policy a phantom transient.
    reset(frame);
    return;
  }
  // head_ points at the oldest frame, which is exactly the slot the new frame replaces;
  // advancing head_ then makes the new frame the newest.
  frames_[head_] = frame.flatten();
  head_ = (head_ + 1) % kHistoryLength;
}

const std::array<float, policy::kObsDim> & ObservationBuilder::pack()
{
  // Term-major: all of term A's history, then all of term B's. Within a term, oldest
  // frame first. Output offset for term starting at `frame_offset` with width `w` is
  // frame_offset * kHistoryLength + h * w.
  int frame_offset = 0;
  int out_offset = 0;
  for (const auto & term : kTerms) {
    for (std::size_t h = 0; h < kHistoryLength; ++h) {
      const auto & frame = frames_[(head_ + h) % kHistoryLength];
      std::copy_n(frame.begin() + frame_offset, term.width, packed_.begin() + out_offset);
      out_offset += term.width;
    }
    frame_offset += term.width;
  }
  return packed_;
}

}  // namespace go2_rl_velocity
