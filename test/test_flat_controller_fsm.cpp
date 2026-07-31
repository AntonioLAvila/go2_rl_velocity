// Copyright (c) 2026 Antonio Avila. MIT License.
//
// Behavioural tests for the flat controller's state machine.
//
// The rugged controller's FSM lives in its node target and so is only exercised on a
// robot. The flat one is in the core library specifically so this file can exist: the
// position-to-torque handover and the two torque-mode watchdogs are new logic with no
// counterpart to copy from, and both are things whose first failure would be a fall.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <string>

#include <gtest/gtest.h>

#include "go2_rl_velocity/flat/controller_fsm.hpp"
#include "go2_rl_velocity/flat/policy_constants.hpp"
#include "go2_rl_velocity/flat/policy_runner.hpp"

namespace
{

namespace flat = go2_rl_velocity::flat;
using flat::ControllerState;

/// A robot standing quietly at the policy's default pose, upright, with no operator
/// input. Every test starts from here and perturbs one thing.
flat::RobotObservation defaultObs()
{
  flat::RobotObservation obs;
  for (int i = 0; i < flat::kNumJoints; ++i) {
    obs.joint_pos[i] = static_cast<float>(flat::kDefaultJointPos[i]);
  }
  obs.imu_quaternion = {1.0F, 0.0F, 0.0F, 0.0F};   // upright
  obs.lowstate_fresh = true;
  return obs;
}

/// Holds the runner alive for the FSM's reference and keeps declaration order right.
struct Harness
{
  explicit Harness(const flat::ControllerFsm::Config & config)
  : runner(GO2_RL_FLAT_POLICY_ONNX, /*warmup_iterations=*/1), fsm(config, runner),
    config(config) {}

  flat::PolicyRunner runner;
  flat::ControllerFsm fsm;
  flat::ControllerFsm::Config config;

  /// Stand, hand over, and let the gain ramp complete. Returns the time at which the
  /// policy engaged; the caller's next step() is the first one the policy drives.
  double driveToEngagedRl(const flat::RobotObservation & obs, double t0 = 0.0)
  {
    std::string reason;
    EXPECT_TRUE(fsm.requestState(ControllerState::FixStand, t0, reason)) << reason;
    fsm.step(obs, t0);                                    // captures the start pose
    const double stood = t0 + 2.0 * config.stand_segment_s;
    fsm.step(obs, stood);                                 // stand complete
    EXPECT_TRUE(fsm.requestState(ControllerState::Rl, stood, reason)) << reason;
    fsm.step(obs, stood);                                 // blend, alpha = 0
    const double engaged = stood + config.gain_ramp_s;
    fsm.step(obs, engaged);                               // blend, alpha = 1 -> engaged
    EXPECT_EQ(fsm.state(), ControllerState::Rl);
    return engaged;
  }
};

float maxAbs(const std::array<float, flat::kNumJoints> & values)
{
  float peak = 0.0F;
  for (const float v : values) {
    peak = std::max(peak, std::fabs(v));
  }
  return peak;
}

// ---------------------------------------------------------------------------
// Transitions.
// ---------------------------------------------------------------------------

TEST(FlatControllerFsm, StartsPassiveAndLimp)
{
  Harness h{{}};
  EXPECT_EQ(h.fsm.state(), ControllerState::Passive);
  h.fsm.step(defaultObs(), 0.0);

  for (int i = 0; i < flat::kNumJoints; ++i) {
    EXPECT_FLOAT_EQ(h.fsm.command().kp[i], 0.0F);
    EXPECT_FLOAT_EQ(h.fsm.command().kd[i], static_cast<float>(flat::kDampingKd));
    EXPECT_FLOAT_EQ(h.fsm.command().tau[i], 0.0F);
  }
}

// The policy's observations are relative to its default pose, and the flat task
// terminated on a base height 2.5 cm below nominal -- so it has never seen a crouch and
// cannot recover from one. RL must start from a stand.
TEST(FlatControllerFsm, RlIsOnlyReachableFromFixStand)
{
  Harness h{{}};
  std::string reason;

  EXPECT_FALSE(h.fsm.requestState(ControllerState::Rl, 0.0, reason));
  EXPECT_FALSE(reason.empty());
  EXPECT_EQ(h.fsm.state(), ControllerState::Passive);

  ASSERT_TRUE(h.fsm.requestState(ControllerState::FixStand, 0.0, reason));
  EXPECT_TRUE(h.fsm.requestState(ControllerState::Rl, 0.0, reason));
  EXPECT_EQ(h.fsm.state(), ControllerState::Rl);
}

TEST(FlatControllerFsm, StandIsPositionControlledWithNoTorque)
{
  Harness h{{}};
  std::string reason;
  ASSERT_TRUE(h.fsm.requestState(ControllerState::FixStand, 0.0, reason));

  auto obs = defaultObs();
  h.fsm.step(obs, 0.0);
  h.fsm.step(obs, 1.5);

  for (int i = 0; i < flat::kNumJoints; ++i) {
    EXPECT_FLOAT_EQ(h.fsm.command().kp[i], static_cast<float>(flat::kStandKp[i]));
    EXPECT_FLOAT_EQ(h.fsm.command().kd[i], static_cast<float>(flat::kStandKd[i]));
    EXPECT_FLOAT_EQ(h.fsm.command().tau[i], 0.0F) << "joint " << i;
  }
}

// ---------------------------------------------------------------------------
// The handover.
// ---------------------------------------------------------------------------

// The blend moves authority from the stand's position loop to the policy's torque. kp
// goes to zero; kd goes to the passive damping the policy trained with, NOT to zero.
TEST(FlatControllerFsm, HandoverBlendsPositionAuthorityIntoTorqueMode)
{
  Harness h{{}};
  const auto obs = defaultObs();
  std::string reason;

  ASSERT_TRUE(h.fsm.requestState(ControllerState::FixStand, 0.0, reason));
  h.fsm.step(obs, 0.0);
  h.fsm.step(obs, 2.0);
  ASSERT_TRUE(h.fsm.requestState(ControllerState::Rl, 2.0, reason));

  // Start of the ramp: still the stand's gains, so nothing changes at the instant of the
  // transition.
  h.fsm.step(obs, 2.0);
  for (int i = 0; i < flat::kNumJoints; ++i) {
    EXPECT_NEAR(h.fsm.command().kp[i], static_cast<float>(flat::kStandKp[i]), 1e-4F);
    EXPECT_NEAR(h.fsm.command().kd[i], static_cast<float>(flat::kStandKd[i]), 1e-4F);
    EXPECT_FLOAT_EQ(h.fsm.command().tau[i], 0.0F);
    EXPECT_NEAR(
      h.fsm.command().joint_target[i], static_cast<float>(flat::kDefaultJointPos[i]), 1e-6F);
  }

  // Halfway: strictly between the two, and no torque yet.
  h.fsm.step(obs, 2.0 + 0.5 * h.config.gain_ramp_s);
  EXPECT_LT(h.fsm.command().kp[1], static_cast<float>(flat::kStandKp[1]));
  EXPECT_GT(h.fsm.command().kp[1], 0.0F);
  EXPECT_FLOAT_EQ(h.fsm.command().tau[1], 0.0F);

  // End of the ramp: torque mode.
  h.fsm.step(obs, 2.0 + h.config.gain_ramp_s);
  for (int i = 0; i < flat::kNumJoints; ++i) {
    EXPECT_NEAR(h.fsm.command().kp[i], 0.0F, 1e-5F) << "joint " << i;
    EXPECT_NEAR(h.fsm.command().kd[i], 2.0F, 1e-5F) << "joint " << i;
    EXPECT_FLOAT_EQ(h.fsm.command().tau[i], 0.0F) << "joint " << i;
  }
}

TEST(FlatControllerFsm, PolicyDrivesTorqueOnceEngaged)
{
  Harness h{{}};
  const auto obs = defaultObs();
  const double engaged = h.driveToEngagedRl(obs);

  // Well past the torque ramp, so the policy has its full trained authority.
  h.fsm.step(obs, engaged + 1.0);

  EXPECT_EQ(h.fsm.state(), ControllerState::Rl);
  EXPECT_GT(maxAbs(h.fsm.command().tau), 0.0F) << "the policy is commanding no torque";
  for (int i = 0; i < flat::kNumJoints; ++i) {
    EXPECT_FLOAT_EQ(h.fsm.command().kp[i], 0.0F);
    EXPECT_FLOAT_EQ(h.fsm.command().kd[i], 2.0F);
    EXPECT_LE(std::fabs(h.fsm.command().tau[i]), static_cast<float>(flat::kTorqueLimit[i]))
      << "joint " << i << " exceeded its effort limit";
  }
}

// The gain ramp hands authority over; this one keeps the first torque from arriving as a
// step. Compared against the same configuration with the ramp disabled, at the same
// instant, so the two see identical observations and identical actions.
TEST(FlatControllerFsm, TorqueRampsInFromZero)
{
  flat::ControllerFsm::Config ramped;
  ramped.torque_ramp_s = 0.2;
  flat::ControllerFsm::Config instant;
  instant.torque_ramp_s = 0.0;

  Harness a{ramped};
  Harness b{instant};
  const auto obs = defaultObs();

  const double engaged = a.driveToEngagedRl(obs);
  ASSERT_DOUBLE_EQ(engaged, b.driveToEngagedRl(obs));

  const double first = engaged + flat::kControlDt;
  a.fsm.step(obs, first);
  b.fsm.step(obs, first);

  const float unramped = maxAbs(b.fsm.command().tau);
  ASSERT_GT(unramped, 0.0F);
  const float ramped_peak = maxAbs(a.fsm.command().tau);

  EXPECT_GT(ramped_peak, 0.0F);
  // smoothStep(0.005 / 0.2) is about 0.0018, so one control step in the ramp has let
  // through well under a percent of the policy's authority.
  EXPECT_LT(ramped_peak, 0.01F * unramped);
}

// The default must be the contract value, and the parameter must reach both ends of the
// handover -- the blend target as well as the engaged gains.
TEST(FlatControllerFsm, PassiveDampingKdDefaultsToTheTrainedValueAndIsHonoured)
{
  EXPECT_DOUBLE_EQ(flat::ControllerFsm::Config{}.passive_damping_kd, flat::kPolicyKd[0]);

  flat::ControllerFsm::Config config;
  config.passive_damping_kd = 0.5;
  Harness h{config};
  const auto obs = defaultObs();
  const double engaged = h.driveToEngagedRl(obs);

  for (int i = 0; i < flat::kNumJoints; ++i) {
    EXPECT_NEAR(h.fsm.command().kd[i], 0.5F, 1e-5F) << "blend endpoint, joint " << i;
  }
  h.fsm.step(obs, engaged + 1.0);
  for (int i = 0; i < flat::kNumJoints; ++i) {
    EXPECT_NEAR(h.fsm.command().kd[i], 0.5F, 1e-5F) << "engaged, joint " << i;
  }
}

TEST(FlatControllerFsm, TorqueScaleDeratesTheWholeCommand)
{
  flat::ControllerFsm::Config full;
  flat::ControllerFsm::Config derated;
  derated.torque_scale = 0.5;

  Harness a{full};
  Harness b{derated};
  const auto obs = defaultObs();

  const double engaged = a.driveToEngagedRl(obs);
  ASSERT_DOUBLE_EQ(engaged, b.driveToEngagedRl(obs));

  const double t = engaged + 1.0;
  a.fsm.step(obs, t);
  b.fsm.step(obs, t);

  ASSERT_GT(maxAbs(a.fsm.command().tau), 0.0F);
  for (int i = 0; i < flat::kNumJoints; ++i) {
    EXPECT_NEAR(b.fsm.command().tau[i], 0.5F * a.fsm.command().tau[i], 1e-4F)
      << "joint " << i;
  }
}

// ---------------------------------------------------------------------------
// Watchdogs. All of them end in DAMPING with kp = 0, kd = 3 and no torque.
// ---------------------------------------------------------------------------

void expectLimpAndFaulted(const flat::ControllerFsm & fsm)
{
  EXPECT_EQ(fsm.state(), ControllerState::Damping);
  EXPECT_FALSE(fsm.faultReason().empty());
  for (int i = 0; i < flat::kNumJoints; ++i) {
    EXPECT_FLOAT_EQ(fsm.command().kp[i], 0.0F);
    EXPECT_FLOAT_EQ(fsm.command().kd[i], static_cast<float>(flat::kDampingKd));
    EXPECT_FLOAT_EQ(fsm.command().tau[i], 0.0F) << "joint " << i << " kept a stale torque";
  }
}

TEST(FlatControllerFsm, StaleLowStateFaults)
{
  Harness h{{}};
  auto obs = defaultObs();
  obs.lowstate_fresh = false;
  h.fsm.step(obs, 0.0);
  expectLimpAndFaulted(h.fsm);
}

TEST(FlatControllerFsm, TiltPastFortyFiveDegreesFaults)
{
  Harness h{{}};
  std::string reason;
  ASSERT_TRUE(h.fsm.requestState(ControllerState::FixStand, 0.0, reason));

  auto obs = defaultObs();
  // 60 degrees of roll: projected gravity z becomes -0.5, above the -0.7071 limit.
  obs.imu_quaternion = {0.8660254F, 0.5F, 0.0F, 0.0F};
  h.fsm.step(obs, 0.0);
  expectLimpAndFaulted(h.fsm);
}

TEST(FlatControllerFsm, EStopChordFaults)
{
  Harness h{{}};
  auto obs = defaultObs();
  obs.remote.buttons =
    static_cast<std::uint16_t>((1u << 5) | (1u << 9));   // L2 + B
  h.fsm.step(obs, 0.0);
  expectLimpAndFaulted(h.fsm);
}

// The torque-mode replacement for the rugged controller's joint-target clamp watchdog.
// Nothing prevents a torque policy driving a joint into its stop, so the excursion is
// detected on the measured position instead.
TEST(FlatControllerFsm, SustainedJointExcursionFaults)
{
  flat::ControllerFsm::Config config;
  config.joint_excursion_fault_steps = 3;
  Harness h{config};

  const double engaged = h.driveToEngagedRl(defaultObs());

  auto excursion = defaultObs();
  // Outside mjlab's soft band but still inside the mechanical limit: reachable, and
  // exactly what the rugged controller's target clamp would have prevented.
  excursion.joint_pos[0] = static_cast<float>(flat::softJointPosUpper(0)) + 0.01F;
  ASSERT_LT(excursion.joint_pos[0], static_cast<float>(flat::kJointPosUpper[0]));

  double t = engaged;
  for (int step = 0; step < config.joint_excursion_fault_steps - 1; ++step) {
    t += flat::kControlDt;
    h.fsm.step(excursion, t);
    EXPECT_EQ(h.fsm.state(), ControllerState::Rl) << "faulted after only " << step + 1
                                                  << " excursion steps";
  }
  t += flat::kControlDt;
  h.fsm.step(excursion, t);
  expectLimpAndFaulted(h.fsm);
}

// A single excursion is a policy leaning on a limit, which happens; the streak must reset
// so it does not accumulate across an entire run.
TEST(FlatControllerFsm, IsolatedJointExcursionsDoNotFault)
{
  flat::ControllerFsm::Config config;
  config.joint_excursion_fault_steps = 3;
  Harness h{config};

  const auto ok = defaultObs();
  auto excursion = ok;
  excursion.joint_pos[0] = static_cast<float>(flat::softJointPosUpper(0)) + 0.01F;

  double t = h.driveToEngagedRl(ok);
  for (int i = 0; i < 12; ++i) {
    t += flat::kControlDt;
    h.fsm.step((i % 2 == 0) ? excursion : ok, t);
  }
  EXPECT_EQ(h.fsm.state(), ControllerState::Rl);
}

TEST(FlatControllerFsm, DampingSettlesBackToPassive)
{
  Harness h{{}};
  const auto obs = defaultObs();

  h.fsm.fault("test", 0.0);
  ASSERT_EQ(h.fsm.state(), ControllerState::Damping);

  h.fsm.step(obs, h.config.damping_settle_s * 0.5);
  EXPECT_EQ(h.fsm.state(), ControllerState::Damping);

  h.fsm.step(obs, h.config.damping_settle_s);
  EXPECT_EQ(h.fsm.state(), ControllerState::Passive);
}

// A held torque is republished at 500 Hz by the node, so a fault that left one in place
// would keep driving the motors all the way through the collapse.
TEST(FlatControllerFsm, FaultingOutOfRlDropsTheHeldTorque)
{
  Harness h{{}};
  const auto obs = defaultObs();
  const double engaged = h.driveToEngagedRl(obs);

  h.fsm.step(obs, engaged + 1.0);
  ASSERT_GT(maxAbs(h.fsm.command().tau), 0.0F);

  h.fsm.fault("e-stop", engaged + 1.0);
  expectLimpAndFaulted(h.fsm);
}

// last_action is an observation term, so a value carried across a fault into the next RL
// session is a phantom the policy reacts to on its first step. Compare a re-entry against
// a first-ever entry driven by identical observations: they must agree.
TEST(FlatControllerFsm, ReEnteringRlDoesNotInheritTheLastAction)
{
  const auto obs = defaultObs();

  Harness reference{{}};
  const double ref_engaged = reference.driveToEngagedRl(obs);
  reference.fsm.step(obs, ref_engaged + 1.0);
  const auto expected = reference.fsm.command().tau;
  ASSERT_GT(maxAbs(expected), 0.0F);

  Harness h{{}};
  double t = h.driveToEngagedRl(obs);
  for (int i = 0; i < 10; ++i) {   // accumulate a non-zero last_action
    t += flat::kControlDt;
    h.fsm.step(obs, t);
  }
  h.fsm.fault("test", t);
  t += h.config.damping_settle_s;
  h.fsm.step(obs, t);
  ASSERT_EQ(h.fsm.state(), ControllerState::Passive);

  const double re_engaged = h.driveToEngagedRl(obs, t);
  h.fsm.step(obs, re_engaged + 1.0);

  for (int i = 0; i < flat::kNumJoints; ++i) {
    EXPECT_NEAR(h.fsm.command().tau[i], expected[i], 1e-4F) << "joint " << i;
  }
}

}  // namespace
