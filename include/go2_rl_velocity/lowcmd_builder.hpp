// Copyright (c) 2026 Antonio Avila. MIT License.

#ifndef GO2_RL_VELOCITY__LOWCMD_BUILDER_HPP_
#define GO2_RL_VELOCITY__LOWCMD_BUILDER_HPP_

#include <cstddef>

#include <unitree_go/msg/low_cmd.hpp>

#include "go2_rl_velocity/controller_fsm.hpp"

namespace go2_rl_velocity
{

/// Number of 32-bit words the LowCmd checksum covers: the whole message except its own
/// trailing crc field. Exposed so a test can independently verify it, not just the node.
inline constexpr std::size_t kLowCmdCrcWords = (sizeof(unitree_go::msg::LowCmd) >> 2) - 1;

/// Fill a LowCmd from a policy-order MotorCommand: header, the joint remap into Unitree
/// motor order, the idle-slot sentinel values on slots 12-19, and the CRC.
///
/// Free of any node/ROS state so a test can exercise the exact bytes a real build would
/// publish, rather than a hand reconstruction of them.
void fillLowCmd(unitree_go::msg::LowCmd & cmd, const MotorCommand & command);

}  // namespace go2_rl_velocity

#endif  // GO2_RL_VELOCITY__LOWCMD_BUILDER_HPP_
