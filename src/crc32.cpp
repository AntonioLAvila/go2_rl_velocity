// Copyright (c) 2026 Antonio Avila. MIT License.

#include "go2_rl_velocity/crc32.hpp"

#include <cstring>

namespace go2_rl_velocity
{

std::uint32_t crc32Core(const void * data, std::size_t word_count)
{
  constexpr std::uint32_t kPolynomial = 0x04C11DB7u;
  std::uint32_t crc = 0xFFFFFFFFu;

  const auto * bytes = static_cast<const std::uint8_t *>(data);
  for (std::size_t i = 0; i < word_count; ++i) {
    // memcpy rather than a reinterpret_cast: the message struct gives no alignment
    // guarantee for a uint32_t view, and this compiles to the same load.
    std::uint32_t word = 0;
    std::memcpy(&word, bytes + i * sizeof(std::uint32_t), sizeof(word));

    std::uint32_t xbit = 1u << 31;
    for (int bit = 0; bit < 32; ++bit) {
      if (crc & 0x80000000u) {
        crc = (crc << 1) ^ kPolynomial;
      } else {
        crc <<= 1;
      }
      if (word & xbit) {
        crc ^= kPolynomial;
      }
      xbit >>= 1;
    }
  }
  return crc;
}

}  // namespace go2_rl_velocity
