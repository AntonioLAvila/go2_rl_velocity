// Copyright (c) 2026 Antonio Avila. MIT License.

#include "go2_rl_velocity/lowcmd_builder.hpp"

#include <cstddef>
#include <cstring>

#include "go2_rl_velocity/crc32.hpp"

namespace go2_rl_velocity
{

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

void fillLowCmd(unitree_go::msg::LowCmd & cmd, const LowCmdJoints & joints)
{
  // LowCmd_'s constructor zeros every *named* field but never the compiler-inserted
  // padding between them. crc32Core below reads raw struct memory, padding included, so
  // uninitialised padding silently corrupts the CRC -- the message still looks correct
  // field by field, and the Go2 discards it without a trace. Routed through void* to
  // sidestep -Wclass-memaccess; LowCmd_ has no virtual members and every field is a
  // fixed-size array of a trivial type, so this is well-defined.
  void * raw_cmd = &cmd;
  std::memset(raw_cmd, 0, sizeof(cmd));

  cmd.head[0] = kLowCmdHead0;
  cmd.head[1] = kLowCmdHead1;
  cmd.level_flag = kLowCmdLevelFlag;
  cmd.gpio = 0;

  for (std::size_t i = 0; i < kMotorSlots; ++i) {
    auto & motor = cmd.motor_cmd[i];
    motor.mode = kMotorModeServo;
    if (i < kGo2Joints) {
      motor.q = joints.q[i];
      motor.dq = 0.0F;
      motor.kp = joints.kp[i];
      motor.kd = joints.kd[i];
      motor.tau = joints.tau[i];
    } else {
      // Slots 12-19 are unused on the Go2. Zero gains keep a stray command from reaching
      // a motor that is not there; mode must still be servo (see kPosStopF/kVelStopF).
      motor.q = kPosStopF;
      motor.dq = kVelStopF;
      motor.kp = 0.0F;
      motor.kd = 0.0F;
      motor.tau = 0.0F;
    }
  }

  // The Go2 silently drops any LowCmd with a bad CRC, so this must be last.
  cmd.crc = crc32Core(&cmd, kLowCmdCrcWords);
}

}  // namespace go2_rl_velocity
