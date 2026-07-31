// Copyright (c) 2026 Antonio Avila. MIT License.
//
// The joint remap between the policy's FL/FR/RL/RR order and the Unitree SDK's
// FR/FL/RR/RL order. Getting this wrong swaps the robot's left and right legs, which
// looks fine standing still.

#include <array>
#include <set>
#include <string>
#include <string_view>

#include <gtest/gtest.h>

#include "go2_rl_velocity/policy_constants.hpp"
#include "go2_rl_velocity/state_conversion.hpp"

namespace
{

namespace policy = go2_rl_velocity::policy;

/// The Go2 low-level motor index table, from the SDK's unitree_legged_const.py.
constexpr std::array<std::string_view, 12> kUnitreeMotorJoints{{
  "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",     // 0, 1, 2
  "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",     // 3, 4, 5
  "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",     // 6, 7, 8
  "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",     // 9, 10, 11
}};

TEST(JointMap, IsAPermutation)
{
  std::set<int> seen(policy::kPolicyToUnitree.begin(), policy::kPolicyToUnitree.end());
  EXPECT_EQ(seen.size(), 12u);
  EXPECT_EQ(*seen.begin(), 0);
  EXPECT_EQ(*seen.rbegin(), 11);
}

// The mapping is what makes the joint names line up; assert against the SDK table rather
// than against itself.
TEST(JointMap, MatchesUnitreeMotorTable)
{
  for (int i = 0; i < policy::kNumJoints; ++i) {
    EXPECT_EQ(policy::kJointNames[i], kUnitreeMotorJoints[policy::kPolicyToUnitree[i]])
      << "policy joint " << i << " (" << policy::kJointNames[i]
      << ") maps to Unitree motor " << policy::kPolicyToUnitree[i];
  }
}

TEST(JointMap, RoundTrips)
{
  std::array<int, 12> unitree{};
  for (int i = 0; i < 12; ++i) {
    unitree[i] = i * 10;
  }
  const auto policy_order = go2_rl_velocity::unitreeToPolicy(unitree);
  const auto back = go2_rl_velocity::policyToUnitree(policy_order);
  EXPECT_EQ(back, unitree);
}

// Documented in policy_constants.hpp and relied on by nothing, but if it ever stops being
// true the comment is a trap.
TEST(JointMap, IsItsOwnInverse)
{
  for (int i = 0; i < policy::kNumJoints; ++i) {
    EXPECT_EQ(policy::kPolicyToUnitree[policy::kPolicyToUnitree[i]], i);
  }
}

TEST(JointMap, SwapsTheLeftAndRightOfEachAxle)
{
  // Front legs occupy policy 0-5 and Unitree 0-5; a left/right swap within each pair.
  EXPECT_EQ(policy::kPolicyToUnitree[0], 3);   // policy FL -> unitree FL
  EXPECT_EQ(policy::kPolicyToUnitree[3], 0);   // policy FR -> unitree FR
  EXPECT_EQ(policy::kPolicyToUnitree[6], 9);   // policy RL -> unitree RL
  EXPECT_EQ(policy::kPolicyToUnitree[9], 6);   // policy RR -> unitree RR
}

TEST(Constants, DefaultPoseIsPerLegHipThighCalf)
{
  for (int leg = 0; leg < 4; ++leg) {
    EXPECT_DOUBLE_EQ(policy::kDefaultJointPos[leg * 3 + 0], 0.0);
    EXPECT_DOUBLE_EQ(policy::kDefaultJointPos[leg * 3 + 1], 0.9);
    EXPECT_DOUBLE_EQ(policy::kDefaultJointPos[leg * 3 + 2], -1.8);
  }
}

TEST(Constants, DefaultPoseIsWithinJointLimits)
{
  for (int i = 0; i < policy::kNumJoints; ++i) {
    EXPECT_GE(policy::kDefaultJointPos[i], policy::kJointPosLower[i]) << "joint " << i;
    EXPECT_LE(policy::kDefaultJointPos[i], policy::kJointPosUpper[i]) << "joint " << i;
  }
}

// The crouch waypoint is driven to blind, before the policy takes over; if it is outside
// the mechanical range the stand-up jams against the stops.
TEST(Constants, CrouchPoseIsWithinJointLimits)
{
  for (int i = 0; i < policy::kNumJoints; ++i) {
    EXPECT_GE(policy::kCrouchJointPos[i], policy::kJointPosLower[i]) << "joint " << i;
    EXPECT_LE(policy::kCrouchJointPos[i], policy::kJointPosUpper[i]) << "joint " << i;
  }
}

TEST(Constants, ActionScaleFollowsMjlabConvention)
{
  // 0.25 * effort_limit / kp, with derated efforts 23.5 / 45.0.
  for (int leg = 0; leg < 4; ++leg) {
    EXPECT_NEAR(policy::kActionScale[leg * 3 + 0], 0.25 * 23.5 / 20.0, 1e-12);
    EXPECT_NEAR(policy::kActionScale[leg * 3 + 1], 0.25 * 23.5 / 20.0, 1e-12);
    EXPECT_NEAR(policy::kActionScale[leg * 3 + 2], 0.25 * 45.0 / 40.0, 1e-12);
  }
}

TEST(StateConversion, ProjectedGravityIsMinusZWhenUpright)
{
  const auto g = go2_rl_velocity::projectedGravity({1.0F, 0.0F, 0.0F, 0.0F});
  EXPECT_NEAR(g[0], 0.0F, 1e-6F);
  EXPECT_NEAR(g[1], 0.0F, 1e-6F);
  EXPECT_NEAR(g[2], -1.0F, 1e-6F);
}

TEST(StateConversion, ProjectedGravityIsUnitLength)
{
  // Arbitrary un-normalised quaternion: the result must still be a unit vector.
  const auto g = go2_rl_velocity::projectedGravity({0.6F, 0.2F, -0.4F, 0.3F});
  const float norm2 = g[0] * g[0] + g[1] * g[1] + g[2] * g[2];
  EXPECT_NEAR(norm2, 1.0F, 1e-5F);
}

TEST(StateConversion, ProjectedGravityHandlesDegenerateQuaternion)
{
  const auto g = go2_rl_velocity::projectedGravity({0.0F, 0.0F, 0.0F, 0.0F});
  EXPECT_NEAR(g[2], -1.0F, 1e-6F);   // "upright", not NaN
}

TEST(StateConversion, ProjectedGravityFlipsWhenUpsideDown)
{
  // 180 degrees about x.
  const auto g = go2_rl_velocity::projectedGravity({0.0F, 1.0F, 0.0F, 0.0F});
  EXPECT_NEAR(g[2], 1.0F, 1e-6F);
}

TEST(StateConversion, ZeroActionGivesDefaultPose)
{
  const std::array<float, 12> zero{};
  const auto target = go2_rl_velocity::actionToTarget(zero);
  for (int i = 0; i < policy::kNumJoints; ++i) {
    EXPECT_NEAR(target[i], static_cast<float>(policy::kDefaultJointPos[i]), 1e-6F);
  }
}

TEST(StateConversion, ClampReportsWhetherItActed)
{
  auto target = go2_rl_velocity::actionToTarget(std::array<float, 12>{});
  EXPECT_FALSE(go2_rl_velocity::clampToJointLimits(target));

  target[0] = 99.0F;
  EXPECT_TRUE(go2_rl_velocity::clampToJointLimits(target));
  EXPECT_NEAR(target[0], static_cast<float>(policy::kJointPosUpper[0]), 1e-6F);
}

}  // namespace
