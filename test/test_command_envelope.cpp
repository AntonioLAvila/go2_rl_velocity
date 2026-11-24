// Copyright (c) 2026 Antonio Avila. MIT License.
//
// The velocity command envelope and the two functions that enforce it.
//
// The envelope is the one part of the policy contract that mjlab does not write into the
// ONNX metadata, so PolicyRunner cannot validate it and a wrong one loads without
// complaint. It is also asymmetric in vx (3.0 m/s forward, 1.0 m/s back), which a
// symmetric clamp or a single stick gain gets wrong in a way that only shows up when the
// operator pulls the stick back. That the *numbers* match the checkpoint is asserted
// against the reference trace in test_policy_parity; this file covers the shaping.

#include <cmath>
#include <limits>

#include <gtest/gtest.h>

#include "go2_rl_velocity/policy_constants.hpp"
#include "go2_rl_velocity/state_conversion.hpp"

namespace
{

using go2_rl_velocity::clampCommand;
using go2_rl_velocity::stickToCommand;
namespace policy = go2_rl_velocity::policy;

constexpr int kVx = 0;
constexpr int kVy = 1;
constexpr int kWz = 2;

TEST(CommandEnvelope, IsWellFormed)
{
  for (int axis = 0; axis < 3; ++axis) {
    // Zero has to be inside the envelope on every axis: it is what a stale /cmd_vel and
    // the post-handover warm-up both command.
    EXPECT_LT(policy::kCommandLower[axis], 0.0) << "axis " << axis;
    EXPECT_GT(policy::kCommandUpper[axis], 0.0) << "axis " << axis;
  }
  // Lateral and yaw are symmetric; only vx is not.
  EXPECT_DOUBLE_EQ(policy::kCommandLower[kVy], -policy::kCommandUpper[kVy]);
  EXPECT_DOUBLE_EQ(policy::kCommandLower[kWz], -policy::kCommandUpper[kWz]);
  EXPECT_GT(policy::kCommandUpper[kVx], -policy::kCommandLower[kVx]);
}

TEST(CommandEnvelope, StickReachesEachEdgeAtFullDeflection)
{
  for (int axis = 0; axis < 3; ++axis) {
    EXPECT_FLOAT_EQ(stickToCommand(1.0, axis), static_cast<float>(policy::kCommandUpper[axis]));
    EXPECT_FLOAT_EQ(
      stickToCommand(-1.0, axis), static_cast<float>(policy::kCommandLower[axis]));
    EXPECT_FLOAT_EQ(stickToCommand(0.0, axis), 0.0F);
  }
}

// The asymmetric axis is the reason stickToCommand exists. Scaling the whole stick by the
// forward bound would ask for three times the reverse speed the policy trained on.
TEST(CommandEnvelope, ForwardAndReverseHaveTheirOwnGain)
{
  EXPECT_FLOAT_EQ(
    stickToCommand(0.5, kVx), static_cast<float>(0.5 * policy::kCommandUpper[kVx]));
  EXPECT_FLOAT_EQ(
    stickToCommand(-0.5, kVx), static_cast<float>(0.5 * policy::kCommandLower[kVx]));
  EXPECT_NE(stickToCommand(0.5, kVx), -stickToCommand(-0.5, kVx));
}

TEST(CommandEnvelope, StickIsMonotonicAndNeverLeavesTheEnvelope)
{
  for (int axis = 0; axis < 3; ++axis) {
    float previous = -std::numeric_limits<float>::infinity();
    // Deliberately runs past +/-1: a mis-scaled remote must saturate, not extrapolate.
    for (int i = -150; i <= 150; ++i) {
      const float command = stickToCommand(i / 100.0, axis);
      EXPECT_GE(command, previous) << "axis " << axis << " stick " << i / 100.0;
      EXPECT_GE(command, static_cast<float>(policy::kCommandLower[axis]));
      EXPECT_LE(command, static_cast<float>(policy::kCommandUpper[axis]));
      previous = command;
    }
  }
}

TEST(CommandEnvelope, ClampUsesEachSidesOwnBound)
{
  for (int axis = 0; axis < 3; ++axis) {
    const auto lower = static_cast<float>(policy::kCommandLower[axis]);
    const auto upper = static_cast<float>(policy::kCommandUpper[axis]);
    EXPECT_FLOAT_EQ(clampCommand(100.0F, axis), upper);
    EXPECT_FLOAT_EQ(clampCommand(-100.0F, axis), lower);
    EXPECT_FLOAT_EQ(clampCommand(0.5F * upper, axis), 0.5F * upper);
    EXPECT_FLOAT_EQ(clampCommand(0.5F * lower, axis), 0.5F * lower);
  }
  // A /cmd_vel asking to reverse at the forward limit gets the reverse limit, not a
  // mirrored forward one.
  EXPECT_FLOAT_EQ(
    clampCommand(-static_cast<float>(policy::kCommandUpper[kVx]), kVx),
    static_cast<float>(policy::kCommandLower[kVx]));
}

// A NaN must not reach the observation: std::clamp passes it straight through, and the
// policy would then emit NaN actions and trip the non-finite watchdog into a collapse.
TEST(CommandEnvelope, NonFiniteInputMeansStop)
{
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float inf = std::numeric_limits<float>::infinity();
  for (int axis = 0; axis < 3; ++axis) {
    EXPECT_EQ(clampCommand(nan, axis), 0.0F);
    EXPECT_EQ(clampCommand(inf, axis), 0.0F);
    EXPECT_EQ(clampCommand(-inf, axis), 0.0F);
    EXPECT_EQ(stickToCommand(static_cast<double>(nan), axis), 0.0F);
    EXPECT_EQ(stickToCommand(static_cast<double>(inf), axis), 0.0F);
  }
}

}  // namespace
