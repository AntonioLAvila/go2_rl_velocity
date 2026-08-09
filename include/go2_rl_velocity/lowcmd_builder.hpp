// Copyright (c) 2026 Antonio Avila. MIT License.

#ifndef GO2_RL_VELOCITY__LOWCMD_BUILDER_HPP_
#define GO2_RL_VELOCITY__LOWCMD_BUILDER_HPP_

#include <array>
#include <cstddef>
#include <cstdint>

#include <unitree_go/msg/low_cmd.hpp>

namespace go2_rl_velocity
{

/// Active motors on a Go2, and the motor slots a LowCmd actually carries.
inline constexpr std::size_t kGo2Joints = 12;
inline constexpr std::size_t kMotorSlots = 20;

/// Number of 32-bit words the LowCmd checksum covers: the whole message except its own
/// trailing crc field. Exposed so a test can independently verify it, not just the node.
inline constexpr std::size_t kLowCmdCrcWords = (sizeof(unitree_go::msg::LowCmd) >> 2) - 1;

/// Unitree MotorCmd.mode for servo (position/velocity/torque) control on the Go2.
inline constexpr std::uint8_t kMotorModeServo = 0x01;

/// Sentinel q/dq values Unitree's own examples use on unused motor slots (12-19) --
/// "no target", not zero. Those slots must still carry kMotorModeServo with zero gains;
/// go2_stand_example (~/unitree_ros2/example/src/src/go2/go2_stand_example.cpp) leaves
/// mode 0x01 on all 20 slots and never changes it. Setting mode 0x00 there instead made
/// the Go2 silently ignore the entire command, not just the absent motors -- confirmed
/// 2026-08-01 by diffing against that example, which stood the robot under
/// otherwise-identical conditions while this node's LowCmd did nothing.
inline constexpr float kPosStopF = 2.146E+9F;
inline constexpr float kVelStopF = 16000.0F;

/// LowCmd header bytes and level flag, as the Go2 expects them.
inline constexpr std::uint8_t kLowCmdHead0 = 0xFE;
inline constexpr std::uint8_t kLowCmdHead1 = 0xEF;
inline constexpr std::uint8_t kLowCmdLevelFlag = 0xFF;

/// One control step's joint command, already remapped into **Unitree motor order**
/// (FR, FL, RR, RL). Each controller applies its own remap before filling this in, so the
/// joint map stays duplicated per policy while the wire format does not.
///
/// The motor board computes `kp*(q - q_meas) + kd*(dq - dq_meas) + tau` with dq fixed at
/// zero here, so all four fields are live at once: the rugged controller leaves `tau`
/// zeroed and drives through kp/kd, the flat controller zeroes kp and drives through tau.
struct LowCmdJoints
{
  std::array<float, kGo2Joints> q{};
  std::array<float, kGo2Joints> kp{};
  std::array<float, kGo2Joints> kd{};
  std::array<float, kGo2Joints> tau{};
};

/// Fill a LowCmd: header, the active slots, the idle-slot sentinels on slots 12-19, and
/// the CRC. Policy-agnostic -- the Unitree wire format has nothing to do with which policy
/// is driving, so both controllers share this rather than carrying a copy each.
///
/// Free of any node/ROS state so a test can exercise the exact bytes a real build would
/// publish, rather than a hand reconstruction of them.
void fillLowCmd(unitree_go::msg::LowCmd & cmd, const LowCmdJoints & joints);

}  // namespace go2_rl_velocity

#endif  // GO2_RL_VELOCITY__LOWCMD_BUILDER_HPP_
