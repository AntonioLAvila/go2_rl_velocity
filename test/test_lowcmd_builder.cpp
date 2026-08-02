// Copyright (c) 2026 Antonio Avila. MIT License.
//
// Verifies the actual compiled fillLowCmd() output against an independent,
// hand-packed-struct CRC reference mirroring Unitree's own get_crc()/crc32_core()
// (unitree_ros2/example/src/{include,src}/common/motor_crc.{h,cpp}), rather than by
// hand-transcribing captured wire values -- that transcription process produced three
// separate false mismatches during debugging (see CLAUDE.md, "The open blocker") before
// being traced to `ros2 topic echo`'s text rendering, not a real bug. This test exists so
// that check never has to be redone by hand again.

#include <array>
#include <cstdint>
#include <cstring>

#include <gtest/gtest.h>

#include <unitree_go/msg/low_cmd.hpp>

#include "go2_rl_velocity/controller_fsm.hpp"
#include "go2_rl_velocity/lowcmd_builder.hpp"
#include "go2_rl_velocity/policy_constants.hpp"

namespace
{

namespace policy = go2_rl_velocity::policy;
using go2_rl_velocity::MotorCommand;

// --- Independent oracle: Unitree's own hand-packed wire struct and CRC algorithm,
// reproduced verbatim from motor_crc.h/motor_crc.cpp. Deliberately not shared code with
// go2_rl_velocity::crc32Core or the ROS message type -- the whole point is to check the
// real fillLowCmd() output against a from-scratch reimplementation of the reference.

struct OracleMotorCmd
{
  std::uint8_t mode;
  float q;
  float dq;
  float tau;
  float kp;
  float kd;
  std::array<std::uint32_t, 3> reserve;
};

struct OracleBmsCmd
{
  std::uint8_t off;
  std::array<std::uint8_t, 3> reserve;
};

struct OracleLowCmd
{
  std::array<std::uint8_t, 2> head;
  std::uint8_t level_flag;
  std::uint8_t frame_reserve;
  std::array<std::uint32_t, 2> sn;
  std::array<std::uint32_t, 2> version;
  std::uint16_t bandwidth;
  std::array<OracleMotorCmd, 20> motor_cmd;
  OracleBmsCmd bms;
  std::array<std::uint8_t, 40> wireless_remote;
  std::array<std::uint8_t, 12> led;
  std::array<std::uint8_t, 2> fan;
  std::uint8_t gpio;
  std::uint32_t reserve;
  std::uint32_t crc;
};

std::uint32_t oracleCrc32Core(const std::uint32_t * ptr, std::uint32_t len)
{
  std::uint32_t crc = 0xFFFFFFFFu;
  constexpr std::uint32_t kPoly = 0x04c11db7u;
  for (std::uint32_t i = 0; i < len; i++) {
    std::uint32_t xbit = 1u << 31;
    const std::uint32_t data = ptr[i];
    for (int bits = 0; bits < 32; bits++) {
      if (crc & 0x80000000u) {
        crc = (crc << 1) ^ kPoly;
      } else {
        crc <<= 1;
      }
      if (data & xbit) {
        crc ^= kPoly;
      }
      xbit >>= 1;
    }
  }
  return crc;
}

/// Independently recompute what the CRC *should* be for a filled LowCmd, by repacking
/// its fields into OracleLowCmd (mirroring Unitree's get_crc()) and running the CRC over
/// that struct's own memory layout -- entirely separate code from lowcmd_builder.cpp.
std::uint32_t oracleCrc(const unitree_go::msg::LowCmd & msg)
{
  OracleLowCmd raw{};
  std::memcpy(&raw.head[0], &msg.head[0], 2);
  raw.level_flag = msg.level_flag;
  raw.frame_reserve = msg.frame_reserve;
  std::memcpy(&raw.sn[0], &msg.sn[0], 8);
  std::memcpy(&raw.version[0], &msg.version[0], 8);
  raw.bandwidth = msg.bandwidth;
  for (int i = 0; i < 20; i++) {
    raw.motor_cmd[i].mode = msg.motor_cmd[i].mode;
    raw.motor_cmd[i].q = msg.motor_cmd[i].q;
    raw.motor_cmd[i].dq = msg.motor_cmd[i].dq;
    raw.motor_cmd[i].tau = msg.motor_cmd[i].tau;
    raw.motor_cmd[i].kp = msg.motor_cmd[i].kp;
    raw.motor_cmd[i].kd = msg.motor_cmd[i].kd;
    std::memcpy(&raw.motor_cmd[i].reserve[0], &msg.motor_cmd[i].reserve[0], 12);
  }
  raw.bms.off = msg.bms_cmd.off;
  std::memcpy(&raw.bms.reserve[0], &msg.bms_cmd.reserve[0], 3);
  std::memcpy(&raw.wireless_remote[0], &msg.wireless_remote[0], 40);
  std::memcpy(&raw.led[0], &msg.led[0], 12);
  std::memcpy(&raw.fan[0], &msg.fan[0], 2);
  raw.gpio = msg.gpio;
  raw.reserve = msg.reserve;

  static_assert(sizeof(OracleLowCmd) == 812, "oracle struct layout drifted from the wire format");
  return oracleCrc32Core(
    reinterpret_cast<const std::uint32_t *>(&raw), (sizeof(OracleLowCmd) >> 2) - 1);
}

}  // namespace

TEST(LowCmdBuilder, FixStandTargetProducesCorrectHeaderAndCrc)
{
  MotorCommand command;
  for (int i = 0; i < policy::kNumJoints; ++i) {
    command.joint_target[i] = static_cast<float>(policy::kDefaultJointPos[i]);
    command.kp[i] = static_cast<float>(policy::kStandKp[i]);
    command.kd[i] = static_cast<float>(policy::kStandKd[i]);
  }

  unitree_go::msg::LowCmd cmd{};
  go2_rl_velocity::fillLowCmd(cmd, command);

  EXPECT_EQ(cmd.head[0], policy::kLowCmdHead0);
  EXPECT_EQ(cmd.head[1], policy::kLowCmdHead1);
  EXPECT_EQ(cmd.level_flag, policy::kLowCmdLevelFlag);
  EXPECT_EQ(cmd.gpio, 0);

  // The independent, from-scratch oracle must agree with the real compiled output.
  EXPECT_EQ(cmd.crc, oracleCrc(cmd))
    << "fillLowCmd()'s CRC does not match an independent reimplementation of Unitree's "
       "own get_crc()/crc32_core() over the same field values -- the Go2 would silently "
       "drop this message.";
}

TEST(LowCmdBuilder, ActiveSlotsCarryStandGainsInUnitreeOrder)
{
  MotorCommand command;
  for (int i = 0; i < policy::kNumJoints; ++i) {
    command.joint_target[i] = static_cast<float>(policy::kDefaultJointPos[i]);
    command.kp[i] = static_cast<float>(policy::kStandKp[i]);
    command.kd[i] = static_cast<float>(policy::kStandKd[i]);
  }

  unitree_go::msg::LowCmd cmd{};
  go2_rl_velocity::fillLowCmd(cmd, command);

  // kStandKp/kStandKd/kDefaultJointPos all repeat the same [hip, thigh, calf] triplet for
  // every leg, so this holds regardless of which leg is which -- it does NOT exercise the
  // remap direction (see the next test for that), only that stand gains/targets reached
  // the active slots at all, with the right per-joint-type value.
  constexpr std::array<float, 3> kExpectedQ{0.0F, 0.9F, -1.8F};
  constexpr std::array<float, 3> kExpectedKp{60.0F, 80.0F, 80.0F};
  constexpr std::array<float, 3> kExpectedKd{5.0F, 4.0F, 4.0F};
  for (int i = 0; i < 12; ++i) {
    const auto & motor = cmd.motor_cmd[i];
    EXPECT_EQ(motor.mode, policy::kMotorModeServo) << "slot " << i;
    EXPECT_NEAR(motor.q, kExpectedQ[i % 3], 1e-5F) << "slot " << i;
    EXPECT_FLOAT_EQ(motor.dq, 0.0F) << "slot " << i;
    EXPECT_FLOAT_EQ(motor.tau, 0.0F) << "slot " << i;
    EXPECT_FLOAT_EQ(motor.kp, kExpectedKp[i % 3]) << "slot " << i;
    EXPECT_FLOAT_EQ(motor.kd, kExpectedKd[i % 3]) << "slot " << i;
  }
}

TEST(LowCmdBuilder, IdleSlotsUseServoModeWithStopSentinels)
{
  MotorCommand command{};  // All zero is fine; only slots 12-19 are checked here.

  unitree_go::msg::LowCmd cmd{};
  go2_rl_velocity::fillLowCmd(cmd, command);

  for (int i = 12; i < 20; ++i) {
    const auto & motor = cmd.motor_cmd[i];
    EXPECT_EQ(motor.mode, policy::kMotorModeServo) << "slot " << i;
    EXPECT_FLOAT_EQ(motor.q, policy::kPosStopF) << "slot " << i;
    EXPECT_FLOAT_EQ(motor.dq, policy::kVelStopF) << "slot " << i;
    EXPECT_FLOAT_EQ(motor.kp, 0.0F) << "slot " << i;
    EXPECT_FLOAT_EQ(motor.kd, 0.0F) << "slot " << i;
    EXPECT_FLOAT_EQ(motor.tau, 0.0F) << "slot " << i;
  }
}

TEST(LowCmdBuilder, RemapsEachJointToItsOwnUnitreeSlot)
{
  // Distinct per-joint values (unlike the symmetric FIXSTAND pose above) so a leg-swap or
  // scrambled remap would show up as a mismatched value at a specific slot, not just a
  // differently-labelled but still-plausible-looking one.
  MotorCommand command;
  for (int i = 0; i < policy::kNumJoints; ++i) {
    command.joint_target[i] = static_cast<float>(i);
    command.kp[i] = static_cast<float>(100 + i);
    command.kd[i] = static_cast<float>(200 + i);
  }

  unitree_go::msg::LowCmd cmd{};
  go2_rl_velocity::fillLowCmd(cmd, command);

  for (int policy_index = 0; policy_index < policy::kNumJoints; ++policy_index) {
    const int unitree_slot = policy::kPolicyToUnitree[policy_index];
    const auto & motor = cmd.motor_cmd[unitree_slot];
    EXPECT_FLOAT_EQ(motor.q, static_cast<float>(policy_index))
      << "policy joint " << policy_index << " -> Unitree slot " << unitree_slot;
    EXPECT_FLOAT_EQ(motor.kp, static_cast<float>(100 + policy_index))
      << "policy joint " << policy_index << " -> Unitree slot " << unitree_slot;
    EXPECT_FLOAT_EQ(motor.kd, static_cast<float>(200 + policy_index))
      << "policy joint " << policy_index << " -> Unitree slot " << unitree_slot;
  }
}
