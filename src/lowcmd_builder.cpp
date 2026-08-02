// Copyright (c) 2026 Antonio Avila. MIT License.

#include "go2_rl_velocity/lowcmd_builder.hpp"

#include <cstddef>
#include <cstring>

#include "go2_rl_velocity/crc32.hpp"
#include "go2_rl_velocity/state_conversion.hpp"

namespace go2_rl_velocity
{

namespace
{

/// LowCmd/LowState carry 20 motor slots; the Go2 uses the first 12. The rest must be
/// explicitly idled, not just left default-constructed.
constexpr std::size_t kMotorSlots = 20;

// The word-wise CRC is only correct if the generated message is densely packed with the
// crc as its final word -- which is what makes it byte-identical to the IDL struct the
// Go2's firmware checksums. With the stock unitree_ros2 messages this is 812 bytes with
// crc at offset 808. If a future unitree_go changes the layout, fail here rather than
// silently emitting commands the robot discards.
static_assert(
  sizeof(unitree_go::msg::LowCmd) % 4 == 0,
  "LowCmd is not a whole number of 32-bit words; the Unitree CRC cannot be computed "
  "over it. Check the unitree_go message definitions.");
static_assert(
  offsetof(unitree_go::msg::LowCmd, crc) == kLowCmdCrcWords * 4,
  "LowCmd.crc is not the final word of the message; the CRC would cover the wrong "
  "bytes and the Go2 would silently discard every command.");

}  // namespace

void fillLowCmd(unitree_go::msg::LowCmd & cmd, const MotorCommand & command)
{
  // LowCmd_'s constructor zeros every *named* field individually (std::fill / direct
  // assignment) but never touches the compiler-inserted padding between them (e.g. 3
  // bytes after MotorCmd::mode, before the next 4-byte-aligned float). crc32Core below
  // reads raw struct memory, padding included, so uninitialised padding silently
  // corrupts the CRC we compute -- the message still looks entirely correct field by
  // field, and the Go2 discards it without a trace. Unitree's own get_crc() sidesteps
  // this by rebuilding into a fresh value-initialised struct before checksumming;
  // zeroing here first gets the same guarantee without a second struct type. LowCmd_ has
  // no virtual members and every field is a fixed-size array of a trivial type, so this
  // is well-defined; routed through void* to sidestep -Wclass-memaccess, which flags any
  // memset on a type with a user-provided constructor regardless of whether it is
  // actually unsafe.
  void * raw_cmd = &cmd;
  std::memset(raw_cmd, 0, sizeof(cmd));

  cmd.head[0] = policy::kLowCmdHead0;
  cmd.head[1] = policy::kLowCmdHead1;
  cmd.level_flag = policy::kLowCmdLevelFlag;
  cmd.gpio = 0;

  const auto target = policyToUnitree(command.joint_target);
  const auto kp = policyToUnitree(command.kp);
  const auto kd = policyToUnitree(command.kd);

  for (std::size_t i = 0; i < kMotorSlots; ++i) {
    auto & motor = cmd.motor_cmd[i];
    motor.mode = policy::kMotorModeServo;
    if (i < static_cast<std::size_t>(policy::kNumJoints)) {
      motor.q = target[i];
      motor.dq = 0.0F;
      motor.kp = kp[i];
      motor.kd = kd[i];
    } else {
      // Slots 12-19 are unused on the Go2. Zero gains keep a stray command from reaching
      // a motor that is not there; mode must still be servo (see kPosStopF/kVelStopF).
      motor.q = policy::kPosStopF;
      motor.dq = policy::kVelStopF;
      motor.kp = 0.0F;
      motor.kd = 0.0F;
    }
    motor.tau = 0.0F;
  }

  // The Go2 silently drops any LowCmd with a bad CRC, so this must be last.
  cmd.crc = crc32Core(&cmd, kLowCmdCrcWords);
}

}  // namespace go2_rl_velocity
