// Copyright (c) 2026 Antonio Avila. MIT License.

#ifndef GO2_RL_VELOCITY__UNITREE_REMOTE_HPP_
#define GO2_RL_VELOCITY__UNITREE_REMOTE_HPP_

#include <array>
#include <cstdint>
#include <string>

namespace go2_rl_velocity
{

/// Buttons on the Unitree wireless remote, as bit positions in the 16-bit field.
///
/// VERIFY THIS EMPIRICALLY before trusting a transition to it. The layout below is the
/// one every Unitree example uses, but it is undocumented and the node exposes a
/// `remote_debug` parameter that logs the decoded struct precisely so it can be checked
/// against the physical remote.
enum class RemoteButton : std::uint16_t
{
  R1 = 0, L1 = 1, Start = 2, Select = 3,
  R2 = 4, L2 = 5, F1 = 6, F2 = 7,
  A = 8, B = 9, X = 10, Y = 11,
  Up = 12, Right = 13, Down = 14, Left = 15,
};

/// Decoded state of the remote.
struct RemoteState
{
  std::uint16_t buttons{0};
  float lx{0.0F};   ///< Left stick, horizontal.
  float rx{0.0F};   ///< Right stick, horizontal.
  float ry{0.0F};   ///< Right stick, vertical.
  float l2{0.0F};   ///< L2 analogue trigger.
  float ly{0.0F};   ///< Left stick, vertical.

  bool held(RemoteButton b) const
  {
    return (buttons >> static_cast<std::uint16_t>(b)) & 1u;
  }

  /// True when every button in the combination is held.
  template<typename ... Buttons>
  bool chord(Buttons ... buttons_held) const
  {
    return (held(buttons_held) && ...);
  }

  std::string describe() const;
};

/// Decodes LowState.wireless_remote and tracks edges.
///
/// The Go2 publishes the remote state inside every LowState at ~500 Hz, so a level check
/// on a chord would fire hundreds of times per press. Transitions use `chordPressed`,
/// which is true only on the cycle the chord becomes complete.
class UnitreeRemote
{
public:
  /// Decode one 40-byte wireless_remote blob:
  ///
  ///   offset  0  uint8[2] head
  ///   offset  2  uint16   button bitfield
  ///   offset  4  float    lx
  ///   offset  8  float    rx
  ///   offset 12  float    ry
  ///   offset 16  float    L2
  ///   offset 20  float    ly
  ///   offset 24  uint8[16] idle
  ///
  /// Sticks are clamped to [-1, 1]; the robot has been seen to report slightly out-of-range
  /// values at the stops, and those feed straight into a velocity command.
  static RemoteState decode(const std::array<std::uint8_t, 40> & raw);

  /// Feed a freshly decoded state. Must be called once per LowState.
  void update(const RemoteState & state);

  const RemoteState & state() const {return current_;}

  /// True on the first update where all of `buttons` are held, having not all been held
  /// on the previous update.
  template<typename ... Buttons>
  bool chordPressed(Buttons ... buttons) const
  {
    return current_.chord(buttons ...) && !previous_.chord(buttons ...);
  }

private:
  RemoteState current_{};
  RemoteState previous_{};
};

}  // namespace go2_rl_velocity

#endif  // GO2_RL_VELOCITY__UNITREE_REMOTE_HPP_
