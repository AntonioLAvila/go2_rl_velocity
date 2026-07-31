// Copyright (c) 2026 Antonio Avila. MIT License.
//
// Replay the flat-ground reference trace generated from the training repo and require
// this package's pipeline to reproduce it.
//
// This carries more weight than its rugged counterpart. The rugged policy's ONNX embeds a
// metadata map that FlatPolicyRunner's sibling cross-checks against policy_constants.hpp
// at startup, so a drifted constant fails before the robot moves. Flat exports carry NO
// metadata -- mjlab's exporter is hard-wired to a "joint_pos" action term and silently
// gives up on a torque task -- so the startup check is reduced to the graph's width, and
// this trace is the ONLY thing pinning:
//
//   * the observation term order (the whole layout contract, since there is no history)
//   * the Unitree <-> policy joint remap, in both directions
//   * the quaternion -> projected_gravity sign convention
//   * the last_action off-by-one
//   * action -> TORQUE, and its saturation
//   * that flat/policy_constants.hpp still agrees with the trained checkpoint
//
// Expected values come from scripts/gen_flat_reference_trace.py, which asserts the layout
// against the checkpoint's own frozen params/env.yaml. See that file for why they are not
// written by hand.

#include <array>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "go2_rl_velocity/flat/observation.hpp"
#include "go2_rl_velocity/flat/policy_constants.hpp"
#include "go2_rl_velocity/flat/policy_runner.hpp"
#include "go2_rl_velocity/flat/state_conversion.hpp"

namespace
{

namespace flat = go2_rl_velocity::flat;
using flat::ObservationFrame;
using flat::PolicyRunner;

/// 2 + 12 + 12 + 3 + 4 + 3 + 45 + 12 + 12. Deliberately not shared with the rugged
/// loader, whose trace is 285 columns wide.
constexpr std::size_t kColumns = 105;

struct ReferenceRow
{
  int step{};
  bool reset{};
  std::array<float, 12> q_unitree{};
  std::array<float, 12> dq_unitree{};
  std::array<float, 3> gyro{};
  std::array<float, 4> quat{};      // w, x, y, z
  std::array<float, 3> command{};
  std::array<float, flat::kObsDim> obs{};
  std::array<float, 12> action{};
  std::array<float, 12> torque_unitree{};
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
    values.reserve(kColumns);
    std::stringstream ss(line);
    std::string cell;
    while (std::getline(ss, cell, ',')) {
      values.push_back(std::strtod(cell.c_str(), nullptr));
    }
    EXPECT_EQ(values.size(), kColumns) << "unexpected column count on row " << rows.size();

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
    for (auto & v : row.torque_unitree) {v = static_cast<float>(values[i++]);}
    rows.push_back(row);
  }
  return rows;
}

/// Build the observation from a row exactly as the controller node does.
ObservationFrame frameFromRow(
  const ReferenceRow & row, const std::array<float, 12> & last_action)
{
  ObservationFrame frame;

  const auto q_policy = flat::unitreeToPolicy(row.q_unitree);
  const auto dq_policy = flat::unitreeToPolicy(row.dq_unitree);
  for (int i = 0; i < flat::kNumJoints; ++i) {
    frame.joint_pos_rel[i] = q_policy[i] - static_cast<float>(flat::kDefaultJointPos[i]);
    frame.joint_vel[i] = dq_policy[i];
  }
  frame.base_ang_vel = row.gyro;
  frame.projected_gravity = flat::projectedGravity(row.quat);
  frame.last_action = last_action;
  frame.command = row.command;
  return frame;
}

TEST(FlatPolicyParity, ReproducesTrainingRepoTrace)
{
  const auto rows = loadTrace(GO2_RL_FLAT_REFERENCE_CSV);
  ASSERT_FALSE(rows.empty());

  PolicyRunner runner(GO2_RL_FLAT_POLICY_ONNX, /*warmup_iterations=*/8);
  std::array<float, 12> last_action{};

  for (const auto & row : rows) {
    if (row.reset) {
      last_action.fill(0.0F);
    }
    const auto obs = frameFromRow(row, last_action).flatten();

    for (int i = 0; i < flat::kObsDim; ++i) {
      ASSERT_NEAR(obs[i], row.obs[i], 1e-5F)
        << "observation mismatch at step " << row.step << ", element " << i;
    }

    const auto & action = runner.run(obs);
    for (int i = 0; i < flat::kActionDim; ++i) {
      ASSERT_NEAR(action[i], row.action[i], 1e-4F)
        << "action mismatch at step " << row.step << ", joint " << i;
    }

    // The outgoing half of the pipeline: action -> torque -> saturation -> motor order.
    auto torque_policy = flat::actionToTorque(action);
    flat::clampToTorqueLimits(torque_policy);
    const auto torque_unitree = flat::policyToUnitree(torque_policy);
    for (int i = 0; i < flat::kNumJoints; ++i) {
      // Absolute tolerance against N*m, which run to 45 -- roughly the same relative
      // precision as the 1e-4 applied to the dimensionless action.
      ASSERT_NEAR(torque_unitree[i], row.torque_unitree[i], 2e-3F)
        << "torque mismatch at step " << row.step << ", unitree motor " << i;
    }

    last_action = action;
  }
}

// ---------------------------------------------------------------------------
// Discrimination tests.
//
// A parity test that passes is only meaningful if it would have failed. Each of the
// following reproduces one of the specific mistakes this package exists to avoid, and
// asserts the reference trace rejects it. If any of these ever starts passing, the trace
// has stopped constraining the thing it was written to constrain.
//
// The rugged suite's history discriminators (time-major packing, reversed history) have
// no analogue here -- there is no history to order. Their place is taken by checks on the
// term layout and on the control law, the latter mattering specifically because a
// position controller for the same robot sits alongside this one in the same package.
// ---------------------------------------------------------------------------

TEST(FlatPolicyParityDiscrimination, SwappedLegsDoNotMatch)
{
  const auto rows = loadTrace(GO2_RL_FLAT_REFERENCE_CSV);
  ASSERT_FALSE(rows.empty());
  const auto & row = rows.front();

  // Feed the joint arrays straight through in Unitree order, i.e. forget the remap. This
  // is the failure that leaves the robot standing happily and then falling when it moves.
  ObservationFrame unmapped;
  for (int i = 0; i < flat::kNumJoints; ++i) {
    unmapped.joint_pos_rel[i] =
      row.q_unitree[i] - static_cast<float>(flat::kDefaultJointPos[i]);
    unmapped.joint_vel[i] = row.dq_unitree[i];
  }
  unmapped.base_ang_vel = row.gyro;
  unmapped.projected_gravity = flat::projectedGravity(row.quat);
  unmapped.command = row.command;

  EXPECT_NE(unmapped.flatten(), row.obs) << "joint remap is unconstrained";
}

TEST(FlatPolicyParityDiscrimination, FlippedGravitySignDoesNotMatch)
{
  const auto rows = loadTrace(GO2_RL_FLAT_REFERENCE_CSV);
  ASSERT_FALSE(rows.empty());
  const auto & row = rows.front();

  const std::array<float, 12> zero{};
  auto frame = frameFromRow(row, zero);
  for (auto & g : frame.projected_gravity) {
    g = -g;
  }
  EXPECT_NE(frame.flatten(), row.obs) << "gravity sign convention is unconstrained";
}

// joint_pos is `q - q_default`, not `q`. With the default pose being [0, 0.9, -1.8] the
// difference is large, but it is also exactly the kind of thing that survives a code
// review because the raw reading looks like a perfectly reasonable observation.
TEST(FlatPolicyParityDiscrimination, RawJointPositionDoesNotMatch)
{
  const auto rows = loadTrace(GO2_RL_FLAT_REFERENCE_CSV);
  ASSERT_FALSE(rows.empty());
  const auto & row = rows.front();

  const std::array<float, 12> zero{};
  auto frame = frameFromRow(row, zero);
  frame.joint_pos_rel = flat::unitreeToPolicy(row.q_unitree);
  EXPECT_NE(frame.flatten(), row.obs) << "the default-pose offset is unconstrained";
}

// Term order is the entire layout contract for a single-frame observation. Swapping two
// same-shaped terms (base_ang_vel and projected_gravity, both 3-wide) keeps the vector
// the right length and every element in range.
TEST(FlatPolicyParityDiscrimination, SwappedGyroAndGravityDoNotMatch)
{
  const auto rows = loadTrace(GO2_RL_FLAT_REFERENCE_CSV);
  ASSERT_FALSE(rows.empty());
  const auto & row = rows.front();

  const std::array<float, 12> zero{};
  auto frame = frameFromRow(row, zero);
  std::swap(frame.base_ang_vel, frame.projected_gravity);
  EXPECT_NE(frame.flatten(), row.obs) << "observation term order is unconstrained";
}

// The rugged controller lives in the same package and its action law is
// `a * 0.29375 + q_default`, a joint POSITION. Applying it here would send the motor
// boards a number near 0.9 where a torque near 20 N*m was meant. Prove the trace rejects
// it, so the two controllers cannot be crossed without a test failing.
TEST(FlatPolicyParityDiscrimination, PositionActionLawDoesNotMatch)
{
  const auto rows = loadTrace(GO2_RL_FLAT_REFERENCE_CSV);
  ASSERT_FALSE(rows.empty());
  const auto & row = rows.front();

  // The rugged scale: 0.25 * effort_limit / kp.
  constexpr std::array<double, 12> kPositionScale{{
    0.29375, 0.29375, 0.28125, 0.29375, 0.29375, 0.28125,
    0.29375, 0.29375, 0.28125, 0.29375, 0.29375, 0.28125,
  }};

  std::array<float, 12> as_position{};
  for (int i = 0; i < flat::kNumJoints; ++i) {
    as_position[i] = static_cast<float>(
      static_cast<double>(row.action[i]) * kPositionScale[i] + flat::kDefaultJointPos[i]);
  }
  const auto as_position_unitree = flat::policyToUnitree(as_position);

  EXPECT_NE(as_position_unitree, row.torque_unitree)
    << "the position action law reproduced the torque trace -- the control law is "
       "unconstrained";
}

// ---------------------------------------------------------------------------
// Contract.
// ---------------------------------------------------------------------------

TEST(FlatPolicyContract, LoadsTheFlatCheckpoint)
{
  EXPECT_NO_THROW({PolicyRunner runner(GO2_RL_FLAT_POLICY_ONNX, 1);});
}

// Pins the shape-only validation decision. Flat exports carry an empty metadata map;
// the runner must load them anyway rather than regressing into a throw, and must report
// the absence so the node can warn about it.
TEST(FlatPolicyContract, MetadataAbsenceIsToleratedAndReported)
{
  PolicyRunner runner(GO2_RL_FLAT_POLICY_ONNX, 1);
  EXPECT_FALSE(runner.has_metadata())
    << "this checkpoint now carries metadata; the runner could validate more than shape";
  EXPECT_EQ(runner.run_path(), "unknown (flat exports carry no metadata)");
}

// The one contract error the shape check can still catch, and the most plausible operator
// mistake: pointing the flat node at the rugged checkpoint installed beside it.
TEST(FlatPolicyContract, RejectsTheRuggedCheckpoint)
{
  EXPECT_THROW(
    {PolicyRunner runner(GO2_RL_RUGGED_POLICY_ONNX, 1);},
    flat::PolicyContractError);
}

TEST(FlatPolicyContract, RejectsMissingFile)
{
  EXPECT_ANY_THROW({PolicyRunner runner("/nonexistent/policy.onnx", 1);});
}

}  // namespace
