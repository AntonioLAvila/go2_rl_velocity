// Copyright (c) 2026 Antonio Avila. MIT License.
//
// Structural checks on the flat controller's constants and pure conversions.
//
// The parity test covers what these values do end to end; this file covers the things a
// trace cannot see -- internal consistency, and the specific numbers that must differ
// from the rugged controller's. Since the flat ONNX carries no metadata, nothing else
// asserts that this build's idea of the torque scale or the command envelope is the one
// the policy was trained with.

#include <array>
#include <cmath>

#include <gtest/gtest.h>

#include "go2_rl_velocity/flat/observation.hpp"
#include "go2_rl_velocity/flat/policy_constants.hpp"
#include "go2_rl_velocity/flat/state_conversion.hpp"

namespace
{

namespace flat = go2_rl_velocity::flat;

// ---------------------------------------------------------------------------
// Constants.
// ---------------------------------------------------------------------------

TEST(FlatConstants, ObservationIsASingleFortyFiveDimFrame)
{
  int total = 0;
  for (const auto & term : flat::kTerms) {
    total += term.width;
  }
  EXPECT_EQ(total, flat::kFrameDim);
  EXPECT_EQ(flat::kFrameDim, 45);
  // The defining difference from the rugged controller's 225: no history stacking.
  EXPECT_EQ(flat::kObsDim, flat::kFrameDim);
  EXPECT_EQ(flat::kActionDim, flat::kNumJoints);
}

TEST(FlatConstants, TermOffsetsFollowDeclarationOrder)
{
  EXPECT_EQ(flat::termOffset(0), 0);    // joint_pos
  EXPECT_EQ(flat::termOffset(1), 12);   // joint_vel
  EXPECT_EQ(flat::termOffset(2), 24);   // base_ang_vel
  EXPECT_EQ(flat::termOffset(3), 27);   // projected_gravity
  EXPECT_EQ(flat::termOffset(4), 30);   // last_action
  EXPECT_EQ(flat::termOffset(5), 42);   // command
  EXPECT_EQ(flat::termOffset(flat::kTerms.size()), flat::kObsDim);
}

TEST(FlatConstants, ControlRateIsTwoHundredHertz)
{
  // go2/constants.py SIM_TIMESTEP 0.0025 x DECIMATION 2.
  EXPECT_DOUBLE_EQ(flat::kControlDt, 0.005);
}

// The action scale IS the effort limit for mjlab's JointEffortActionCfg, so these two
// arrays must stay equal -- and must be the derated Unitree values, not the datasheet
// peaks (23.7 / 45.43).
TEST(FlatConstants, TorqueScaleIsTheEffortLimit)
{
  for (int i = 0; i < flat::kNumJoints; ++i) {
    const double expected = (i % 3 == 2) ? 45.0 : 23.5;   // hip, thigh, calf
    EXPECT_DOUBLE_EQ(flat::kTorqueScale[i], expected) << "joint " << i;
    EXPECT_DOUBLE_EQ(flat::kTorqueLimit[i], flat::kTorqueScale[i]) << "joint " << i;
  }
}

// kp must be zero -- there is no position loop -- and kd must be the passive joint
// damping the policy trained against, not zero. See the comment on kPolicyKd.
TEST(FlatConstants, RlGainsAreTorqueModeWithPassiveDamping)
{
  for (int i = 0; i < flat::kNumJoints; ++i) {
    EXPECT_DOUBLE_EQ(flat::kPolicyKp[i], 0.0) << "joint " << i;
    EXPECT_DOUBLE_EQ(flat::kPolicyKd[i], 2.0) << "joint " << i;
  }
}

// Tighter than the rugged policy's 2.0 / 1.0 / 1.5, and fixed: the flat task has no
// command curriculum.
TEST(FlatConstants, CommandEnvelopeIsTheFlatTasks)
{
  EXPECT_DOUBLE_EQ(flat::kCommandBounds[0], 1.5);
  EXPECT_DOUBLE_EQ(flat::kCommandBounds[1], 0.8);
  EXPECT_DOUBLE_EQ(flat::kCommandBounds[2], 1.2);
}

// bad_orientation(limit_angle = pi/4) in the flat task's terminations.
TEST(FlatConstants, TiltLimitIsFortyFiveDegrees)
{
  EXPECT_NEAR(flat::kTiltLimit, -std::cos(M_PI / 4.0), 1e-4);
}

TEST(FlatConstants, JointRemapIsAPermutationAndItsOwnInverse)
{
  std::array<int, flat::kNumJoints> seen{};
  for (const int idx : flat::kPolicyToUnitree) {
    ASSERT_GE(idx, 0);
    ASSERT_LT(idx, flat::kNumJoints);
    ++seen[idx];
  }
  for (const int count : seen) {
    EXPECT_EQ(count, 1);
  }
  for (int i = 0; i < flat::kNumJoints; ++i) {
    EXPECT_EQ(flat::kPolicyToUnitree[flat::kPolicyToUnitree[i]], i);
  }
}

// The soft band is what the excursion watchdog uses, so the poses the controller actually
// commands must sit inside it -- otherwise the watchdog fires the moment the policy
// engages, at the very pose the handover was designed to deliver.
TEST(FlatConstants, DefaultAndCrouchPosesAreInsideTheSoftLimits)
{
  for (int i = 0; i < flat::kNumJoints; ++i) {
    EXPECT_GT(flat::softJointPosLower(i), flat::kJointPosLower[i]) << "joint " << i;
    EXPECT_LT(flat::softJointPosUpper(i), flat::kJointPosUpper[i]) << "joint " << i;

    EXPECT_GT(flat::kDefaultJointPos[i], flat::softJointPosLower(i)) << "joint " << i;
    EXPECT_LT(flat::kDefaultJointPos[i], flat::softJointPosUpper(i)) << "joint " << i;

    EXPECT_GE(flat::kCrouchJointPos[i], flat::kJointPosLower[i]) << "joint " << i;
    EXPECT_LE(flat::kCrouchJointPos[i], flat::kJointPosUpper[i]) << "joint " << i;
  }
}

// ---------------------------------------------------------------------------
// Observation packing.
// ---------------------------------------------------------------------------

TEST(FlatObservation, FlattenFollowsTermOrder)
{
  flat::ObservationFrame frame;
  frame.joint_pos_rel.fill(1.0F);
  frame.joint_vel.fill(2.0F);
  frame.base_ang_vel.fill(3.0F);
  frame.projected_gravity.fill(4.0F);
  frame.last_action.fill(5.0F);
  frame.command.fill(6.0F);

  const auto flat_obs = frame.flatten();
  for (std::size_t t = 0; t < flat::kTerms.size(); ++t) {
    const int begin = flat::termOffset(t);
    for (int i = 0; i < flat::kTerms[t].width; ++i) {
      EXPECT_FLOAT_EQ(flat_obs[begin + i], static_cast<float>(t + 1))
        << "term '" << flat::kTerms[t].name << "', element " << i;
    }
  }
}

// ---------------------------------------------------------------------------
// Conversions.
// ---------------------------------------------------------------------------

TEST(FlatStateConversion, ActionMapsToTorqueWithNoOffset)
{
  std::array<float, flat::kActionDim> action{};
  action.fill(1.0F);
  const auto torque = flat::actionToTorque(action);
  for (int i = 0; i < flat::kNumJoints; ++i) {
    EXPECT_FLOAT_EQ(torque[i], static_cast<float>(flat::kTorqueScale[i]));
  }

  // Zero action is zero torque, NOT the default pose. This is the whole difference from
  // the rugged controller's actionToTarget, which offsets by q_default.
  action.fill(0.0F);
  for (const float t : flat::actionToTorque(action)) {
    EXPECT_FLOAT_EQ(t, 0.0F);
  }
}

TEST(FlatStateConversion, TorqueSaturatesAtTheEffortLimit)
{
  std::array<float, flat::kNumJoints> torque{};
  torque.fill(1000.0F);
  EXPECT_TRUE(flat::clampToTorqueLimits(torque));
  for (int i = 0; i < flat::kNumJoints; ++i) {
    EXPECT_FLOAT_EQ(torque[i], static_cast<float>(flat::kTorqueLimit[i]));
  }

  torque.fill(-1000.0F);
  EXPECT_TRUE(flat::clampToTorqueLimits(torque));
  for (int i = 0; i < flat::kNumJoints; ++i) {
    EXPECT_FLOAT_EQ(torque[i], static_cast<float>(-flat::kTorqueLimit[i]));
  }

  // An in-range torque must be reported as unclamped, or the saturation watchdog would
  // fault on a policy behaving perfectly.
  torque.fill(1.0F);
  EXPECT_FALSE(flat::clampToTorqueLimits(torque));
}

TEST(FlatStateConversion, JointRemapRoundTrips)
{
  std::array<float, flat::kNumJoints> unitree{};
  for (int i = 0; i < flat::kNumJoints; ++i) {
    unitree[i] = static_cast<float>(i);
  }
  EXPECT_EQ(flat::policyToUnitree(flat::unitreeToPolicy(unitree)), unitree);

  // Policy index 0 is FL_hip; the SDK puts FL at motors 3-5.
  const auto policy_order = flat::unitreeToPolicy(unitree);
  EXPECT_FLOAT_EQ(policy_order[0], 3.0F);
  EXPECT_FLOAT_EQ(policy_order[3], 0.0F);
}

TEST(FlatStateConversion, ProjectedGravityIsMinusZWhenUpright)
{
  const auto g = flat::projectedGravity({1.0F, 0.0F, 0.0F, 0.0F});
  EXPECT_NEAR(g[0], 0.0F, 1e-6F);
  EXPECT_NEAR(g[1], 0.0F, 1e-6F);
  EXPECT_NEAR(g[2], -1.0F, 1e-6F);
}

TEST(FlatStateConversion, ProjectedGravityIsUnitLengthAndSafeOnGarbage)
{
  // Deliberately non-unit quaternion: the IMU's is only approximately normalised.
  const auto g = flat::projectedGravity({0.9F, 0.1F, -0.2F, 0.3F});
  const float norm = std::sqrt(g[0] * g[0] + g[1] * g[1] + g[2] * g[2]);
  EXPECT_NEAR(norm, 1.0F, 1e-5F);

  // A zero quaternion must report upright rather than NaN -- the tilt watchdog cannot
  // fire on a NaN.
  const auto degenerate = flat::projectedGravity({0.0F, 0.0F, 0.0F, 0.0F});
  EXPECT_FLOAT_EQ(degenerate[2], -1.0F);
}

TEST(FlatStateConversion, SoftLimitExcursionIsDetected)
{
  std::array<float, flat::kNumJoints> pose{};
  for (int i = 0; i < flat::kNumJoints; ++i) {
    pose[i] = static_cast<float>(flat::kDefaultJointPos[i]);
  }
  EXPECT_FALSE(flat::outsideSoftJointLimits(pose));

  // Just outside the soft band but still inside the mechanical one: exactly the state the
  // rugged controller would have prevented by clamping its target, and that a torque
  // policy can reach.
  pose[0] = static_cast<float>(flat::softJointPosUpper(0)) + 1e-3F;
  EXPECT_LT(pose[0], static_cast<float>(flat::kJointPosUpper[0]));
  EXPECT_TRUE(flat::outsideSoftJointLimits(pose));
}

}  // namespace
