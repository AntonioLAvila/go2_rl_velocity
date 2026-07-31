// Copyright (c) 2026 Antonio Avila. MIT License.
//
// Replay the reference trace generated from the training repo and require this package's
// pipeline to reproduce it.
//
// This is the primary correctness guard for the whole package. Because there is no
// simulation backend here, it is what stands between a wiring mistake and a fall. It
// covers, in one pass:
//
//   * term-major vs time-major observation layout
//   * history direction (oldest-first) and the reset backfill
//   * the Unitree <-> policy joint remap, in both directions
//   * the quaternion -> projected_gravity sign convention
//   * the last_action off-by-one
//   * action -> position target
//   * that policy_constants.hpp still agrees with the trained checkpoint
//
// Expected values come from scripts/gen_reference_trace.py, which uses the training
// repo's own observation packer. See that file for why they are not written by hand.

#include <array>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "go2_rl_velocity/observation_builder.hpp"
#include "go2_rl_velocity/policy_constants.hpp"
#include "go2_rl_velocity/policy_runner.hpp"
#include "go2_rl_velocity/state_conversion.hpp"

namespace
{

using go2_rl_velocity::ObservationBuilder;
using go2_rl_velocity::ObservationFrame;
using go2_rl_velocity::PolicyRunner;
namespace policy = go2_rl_velocity::policy;

struct ReferenceRow
{
  int step{};
  bool reset{};
  std::array<float, 12> q_unitree{};
  std::array<float, 12> dq_unitree{};
  std::array<float, 3> gyro{};
  std::array<float, 4> quat{};      // w, x, y, z
  std::array<float, 3> command{};
  std::array<float, policy::kObsDim> obs{};
  std::array<float, 12> action{};
  std::array<float, 12> target_unitree{};
};

std::vector<ReferenceRow> loadTrace(const std::string & path)
{
  std::ifstream file(path);
  EXPECT_TRUE(file.is_open()) << "cannot open reference trace: " << path;

  std::string line;
  std::getline(file, line);   // header

  std::vector<ReferenceRow> rows;
  while (std::getline(file, line)) {
    if (line.empty()) {
      continue;
    }
    std::vector<double> values;
    values.reserve(285);
    std::stringstream ss(line);
    std::string cell;
    while (std::getline(ss, cell, ',')) {
      values.push_back(std::strtod(cell.c_str(), nullptr));
    }
    EXPECT_EQ(values.size(), 285u) << "unexpected column count on row " << rows.size();

    ReferenceRow row;
    std::size_t i = 0;
    row.step = static_cast<int>(values[i++]);
    row.reset = values[i++] != 0.0;
    for (auto & v : row.q_unitree) {v = static_cast<float>(values[i++]);}
    for (auto & v : row.dq_unitree) {v = static_cast<float>(values[i++]);}
    for (auto & v : row.gyro) {v = static_cast<float>(values[i++]);}
    for (auto & v : row.quat) {v = static_cast<float>(values[i++]);}
    for (auto & v : row.command) {v = static_cast<float>(values[i++]);}
    for (auto & v : row.obs) {v = static_cast<float>(values[i++]);}
    for (auto & v : row.action) {v = static_cast<float>(values[i++]);}
    for (auto & v : row.target_unitree) {v = static_cast<float>(values[i++]);}
    rows.push_back(row);
  }
  return rows;
}

/// Build the 45-dim frame from a row exactly as the controller node does.
ObservationFrame frameFromRow(
  const ReferenceRow & row, const std::array<float, 12> & last_action)
{
  ObservationFrame frame;

  const auto q_policy = go2_rl_velocity::unitreeToPolicy(row.q_unitree);
  const auto dq_policy = go2_rl_velocity::unitreeToPolicy(row.dq_unitree);
  for (int i = 0; i < policy::kNumJoints; ++i) {
    frame.joint_pos_rel[i] =
      q_policy[i] - static_cast<float>(policy::kDefaultJointPos[i]);
    frame.joint_vel[i] = dq_policy[i];
  }
  frame.base_ang_vel = row.gyro;
  frame.projected_gravity = go2_rl_velocity::projectedGravity(row.quat);
  frame.last_action = last_action;
  frame.command = row.command;
  return frame;
}

TEST(PolicyParity, ReproducesTrainingRepoTrace)
{
  const auto rows = loadTrace(GO2_RL_REFERENCE_CSV);
  ASSERT_FALSE(rows.empty());

  PolicyRunner runner(GO2_RL_POLICY_ONNX, /*warmup_iterations=*/8);
  ObservationBuilder builder;
  std::array<float, 12> last_action{};

  for (const auto & row : rows) {
    if (row.reset) {
      last_action.fill(0.0F);
    }
    const auto frame = frameFromRow(row, last_action);

    if (row.reset) {
      builder.reset(frame);
    } else {
      builder.push(frame);
    }

    const auto & obs = builder.pack();
    for (int i = 0; i < policy::kObsDim; ++i) {
      ASSERT_NEAR(obs[i], row.obs[i], 1e-5F)
        << "observation mismatch at step " << row.step << ", element " << i;
    }

    const auto & action = runner.run(obs);
    for (int i = 0; i < policy::kActionDim; ++i) {
      ASSERT_NEAR(action[i], row.action[i], 1e-4F)
        << "action mismatch at step " << row.step << ", joint " << i;
    }

    // The outgoing half of the pipeline: action -> target -> Unitree motor order.
    const auto target_policy = go2_rl_velocity::actionToTarget(action);
    const auto target_unitree = go2_rl_velocity::policyToUnitree(target_policy);
    for (int i = 0; i < policy::kNumJoints; ++i) {
      ASSERT_NEAR(target_unitree[i], row.target_unitree[i], 1e-4F)
        << "target mismatch at step " << row.step << ", unitree motor " << i;
    }

    last_action = action;
  }
}

// The generator emits an explicit reset flag, but on the robot the very first push after
// construction has to behave identically -- ObservationBuilder auto-backfills. If those
// two paths ever diverge, the policy's first four steps after handover see a transient.
TEST(PolicyParity, FirstPushMatchesExplicitReset)
{
  const auto rows = loadTrace(GO2_RL_REFERENCE_CSV);
  ASSERT_FALSE(rows.empty());

  const std::array<float, 12> zero_action{};
  const auto frame = frameFromRow(rows.front(), zero_action);

  ObservationBuilder via_reset;
  via_reset.reset(frame);
  const auto packed_reset = via_reset.pack();

  ObservationBuilder via_push;
  via_push.push(frame);
  const auto & packed_push = via_push.pack();

  EXPECT_EQ(packed_reset, packed_push);
  for (int i = 0; i < policy::kObsDim; ++i) {
    ASSERT_NEAR(packed_push[i], rows.front().obs[i], 1e-5F) << "element " << i;
  }
}

// ---------------------------------------------------------------------------
// Discrimination tests.
//
// A parity test that passes is only meaningful if it would have failed. Each of the
// following reproduces one of the specific mistakes this package exists to avoid, and
// asserts the reference trace rejects it. If any of these ever starts passing, the trace
// has stopped constraining the thing it was written to constrain.
// ---------------------------------------------------------------------------

/// Pack the same history time-major (frame by frame) instead of term-major -- the
/// legged_gym convention, and the single most likely way to get this wrong.
std::array<float, policy::kObsDim> packTimeMajor(
  const std::array<std::array<float, policy::kFrameDim>, policy::kHistoryLength> & frames)
{
  std::array<float, policy::kObsDim> out{};
  std::size_t k = 0;
  for (const auto & frame : frames) {
    for (const float v : frame) {
      out[k++] = v;
    }
  }
  return out;
}

TEST(PolicyParityDiscrimination, TimeMajorLayoutDoesNotMatch)
{
  const auto rows = loadTrace(GO2_RL_REFERENCE_CSV);
  ASSERT_GE(rows.size(), 6u);

  // Five distinct frames, so term-major and time-major genuinely differ. (After a reset
  // all five slots are equal and the two layouts coincide -- which is exactly why this
  // has to be checked away from a reset boundary.)
  std::array<std::array<float, policy::kFrameDim>, policy::kHistoryLength> frames{};
  std::array<float, 12> last_action{};
  for (std::size_t i = 0; i < policy::kHistoryLength; ++i) {
    frames[i] = frameFromRow(rows[i + 1], last_action).flatten();
    last_action = rows[i + 1].action;
  }

  const auto time_major = packTimeMajor(frames);
  EXPECT_NE(time_major, rows[policy::kHistoryLength].obs)
    << "time-major packing matched the reference -- the trace cannot tell the two "
       "layouts apart and is not constraining the layout";
}

TEST(PolicyParityDiscrimination, ReversedHistoryDoesNotMatch)
{
  const auto rows = loadTrace(GO2_RL_REFERENCE_CSV);
  ASSERT_GE(rows.size(), 8u);

  ObservationBuilder forward;
  std::array<float, 12> last_action{};
  for (std::size_t i = 1; i <= 6; ++i) {
    forward.push(frameFromRow(rows[i], last_action));
    last_action = rows[i].action;
  }
  const auto correct = forward.pack();

  // Same frames, newest first.
  ObservationBuilder reversed;
  last_action = rows[6].action;
  for (std::size_t i = 6; i >= 1; --i) {
    reversed.push(frameFromRow(rows[i], last_action));
    last_action = rows[i - 1].action;
  }
  EXPECT_NE(correct, reversed.pack()) << "history direction is unconstrained";
}

TEST(PolicyParityDiscrimination, SwappedLegsDoNotMatch)
{
  const auto rows = loadTrace(GO2_RL_REFERENCE_CSV);
  ASSERT_FALSE(rows.empty());
  const auto & row = rows.front();

  // Feed the joint arrays straight through in Unitree order, i.e. forget the remap. This
  // is the failure that leaves the robot standing happily and then falling when it moves.
  ObservationFrame unmapped;
  for (int i = 0; i < policy::kNumJoints; ++i) {
    unmapped.joint_pos_rel[i] =
      row.q_unitree[i] - static_cast<float>(policy::kDefaultJointPos[i]);
    unmapped.joint_vel[i] = row.dq_unitree[i];
  }
  unmapped.base_ang_vel = row.gyro;
  unmapped.projected_gravity = go2_rl_velocity::projectedGravity(row.quat);
  unmapped.command = row.command;

  ObservationBuilder builder;
  builder.reset(unmapped);
  EXPECT_NE(builder.pack(), row.obs) << "joint remap is unconstrained";
}

TEST(PolicyParityDiscrimination, FlippedGravitySignDoesNotMatch)
{
  const auto rows = loadTrace(GO2_RL_REFERENCE_CSV);
  ASSERT_FALSE(rows.empty());
  const auto & row = rows.front();

  const std::array<float, 12> zero{};
  auto frame = frameFromRow(row, zero);
  for (auto & g : frame.projected_gravity) {
    g = -g;
  }

  ObservationBuilder builder;
  builder.reset(frame);
  EXPECT_NE(builder.pack(), row.obs) << "gravity sign convention is unconstrained";
}

// A policy trained against different constants must not load at all.
TEST(PolicyContract, MetadataMatchesConstants)
{
  EXPECT_NO_THROW({PolicyRunner runner(GO2_RL_POLICY_ONNX, 1);});
}

TEST(PolicyContract, RejectsMissingFile)
{
  EXPECT_ANY_THROW({PolicyRunner runner("/nonexistent/policy.onnx", 1);});
}

}  // namespace
