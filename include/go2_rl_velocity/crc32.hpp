// Copyright (c) 2026 Antonio Avila. MIT License.

#ifndef GO2_RL_VELOCITY__CRC32_HPP_
#define GO2_RL_VELOCITY__CRC32_HPP_

#include <cstddef>
#include <cstdint>

namespace go2_rl_velocity
{

/// Unitree's LowCmd checksum (MPEG-2 CRC-32, poly 0x04C11DB7, init 0xFFFFFFFF, no final
/// XOR, no reflection).
///
/// The Go2 silently discards any LowCmd whose crc field does not match, so a controller
/// that forgets this looks exactly like a controller whose messages never arrive.
///
/// `data` must point at the message struct itself and `word_count` must be
/// (sizeof(message) / 4) - 1, i.e. every word except the trailing crc field. That works
/// only because the unitree_ros2 LowCmd message is plain fixed-size POD with the same
/// layout as the underlying IDL struct; the caller is expected to static_assert its size.
std::uint32_t crc32Core(const void * data, std::size_t word_count);

}  // namespace go2_rl_velocity

#endif  // GO2_RL_VELOCITY__CRC32_HPP_
