// Copyright (c) 2026 Antonio Avila. MIT License.

#include "go2_rl_velocity/unitree_remote.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <sstream>

namespace go2_rl_velocity
{

namespace
{

float readFloat(const std::array<std::uint8_t, 40> & raw, std::size_t offset)
{
  float value = 0.0F;
  std::memcpy(&value, raw.data() + offset, sizeof(value));
  // A disconnected or garbled remote can present as NaN, which would propagate straight
  // into the velocity command and then into the observation.
  if (!std::isfinite(value)) {
    return 0.0F;
  }
  return std::clamp(value, -1.0F, 1.0F);
}

}  // namespace

RemoteState UnitreeRemote::decode(const std::array<std::uint8_t, 40> & raw)
{
  RemoteState state;
  std::memcpy(&state.buttons, raw.data() + 2, sizeof(state.buttons));
  state.lx = readFloat(raw, 4);
  state.rx = readFloat(raw, 8);
  state.ry = readFloat(raw, 12);
  state.l2 = readFloat(raw, 16);
  state.ly = readFloat(raw, 20);
  return state;
}

void UnitreeRemote::update(const RemoteState & state)
{
  previous_ = current_;
  current_ = state;
}

std::string RemoteState::describe() const
{
  static constexpr std::array<const char *, 16> kNames{{
    "R1", "L1", "start", "select", "R2", "L2", "F1", "F2",
    "A", "B", "X", "Y", "up", "right", "down", "left",
  }};

  std::ostringstream oss;
  oss << "buttons=0x" << std::hex << buttons << std::dec << " [";
  bool first = true;
  for (std::uint16_t bit = 0; bit < 16; ++bit) {
    if ((buttons >> bit) & 1u) {
      if (!first) {
        oss << ' ';
      }
      oss << kNames[bit];
      first = false;
    }
  }
  oss << "] lx=" << lx << " ly=" << ly << " rx=" << rx << " ry=" << ry << " L2=" << l2;
  return oss.str();
}

}  // namespace go2_rl_velocity
