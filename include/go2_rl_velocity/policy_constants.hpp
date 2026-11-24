// Copyright (c) 2026 Antonio Avila. MIT License.
//
// The deployment-side mirror of the trained policy's contract.
//
// Every number here is copied from the training repo and is load-bearing. Nothing
// numeric belongs anywhere else in this package -- if a value is needed elsewhere, it is
// referenced from here. Upstream sources (relative to the quadruped_walking repo):
//
//   go2/constants.py           joint names, default pose, actuator table
//   go2/rugged/constants.py    history length, term widths, action scale, timing, bounds
//   scripts/verify_rugged.py   the reference deployment implementation
//
// PolicyRunner cross-checks most of this against the metadata embedded in the ONNX file
// at load time and refuses to start on any mismatch, so drift here is loud, not silent.

#ifndef GO2_RL_VELOCITY__POLICY_CONSTANTS_HPP_
#define GO2_RL_VELOCITY__POLICY_CONSTANTS_HPP_

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace go2_rl_velocity::policy
{

/// Joints driven by the policy. The Go2 LowState/LowCmd motor arrays are 20 long; the
/// remaining 8 entries are unused and must be explicitly zeroed.
inline constexpr int kNumJoints = 12;

/// Number of stacked observation frames.
///
/// go2/rugged/constants.py:50 -- "5 frames @ 50 Hz = 100 ms". Retraining with a different
/// history changes the ONNX input width; PolicyRunner asserts kObsDim against the actual
/// graph input, so a stale value here fails at startup rather than on the robot.
inline constexpr int kHistoryLength = 5;

/// One observation term: its name as mjlab declares it, and its per-frame width.
struct Term
{
  std::string_view name;
  int width;
};

/// THE LAYOUT CONTRACT (go2/rugged/constants.py:52-68).
///
/// mjlab stacks history *per term* and concatenates afterwards, so the flat observation
/// is TERM-MAJOR, oldest to newest:
///
///   [joint_pos t-4..t | joint_vel t-4..t | ang_vel | gravity | last_action | command]
///
/// This is the opposite of legged_gym's time-major convention. Getting it wrong produces
/// a well-formed vector of the right length that the policy silently misreads.
inline constexpr std::array<Term, 6> kTerms{{
  {"joint_pos", 12},          // q - kDefaultJointPos
  {"joint_vel", 12},          // qd
  {"base_ang_vel", 3},        // IMU gyro, body frame
  {"projected_gravity", 3},   // R^T * (0, 0, -1)
  {"last_action", 12},        // RAW pre-scale a_{t-1}
  {"command", 3},             // [vx, vy, wz]
}};

inline constexpr int kFrameDim = 45;   // sum of kTerms widths
inline constexpr int kObsDim = kFrameDim * kHistoryLength;   // 225
inline constexpr int kActionDim = kNumJoints;

/// Byte offset of a term within one 45-dim frame.
inline constexpr int termFrameOffset(std::size_t term_index)
{
  int offset = 0;
  for (std::size_t i = 0; i < term_index; ++i) {
    offset += kTerms[i].width;
  }
  return offset;
}

/// Control period. go2/rugged/constants.py:36, ControlTiming(0.005, decimation=4) = 50 Hz.
inline constexpr double kControlDt = 0.02;

/// Joint order the policy was trained in: FL, FR, RL, RR x hip, thigh, calf.
/// go2/constants.py:32,59,66. Also embedded in the ONNX metadata.
inline constexpr std::array<std::string_view, kNumJoints> kJointNames{{
  "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
  "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
  "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
  "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
}};

/// Policy joint index -> Unitree LowState/LowCmd motor index.
///
/// The Go2 low-level SDK orders motors FR(0-2), FL(3-5), RR(6-8), RL(9-11) while the
/// policy uses FL, FR, RL, RR. Omitting this remap swaps left and right legs: the robot
/// still stands, and then falls the moment it is asked to move.
///
/// (This permutation swaps two pairs of blocks, so it is its own inverse -- the same
/// table maps Unitree indices back to policy indices. test_joint_map asserts that.)
inline constexpr std::array<int, kNumJoints> kPolicyToUnitree{{
  3, 4, 5,      // FL -> Unitree FL
  0, 1, 2,      // FR -> Unitree FR
  9, 10, 11,    // RL -> Unitree RL
  6, 7, 8,      // RR -> Unitree RR
}};

/// Home keyframe pose, per leg [hip=0, thigh=0.9, calf=-1.8]. go2/constants.py:62.
/// joint_pos observations are relative to this, and position targets are offset from it.
inline constexpr std::array<double, kNumJoints> kDefaultJointPos{{
  0.0, 0.9, -1.8, 0.0, 0.9, -1.8, 0.0, 0.9, -1.8, 0.0, 0.9, -1.8,
}};

/// Action scale: 0.25 * effort_limit / kp (go2/rugged/constants.py:115-123).
///   hip/thigh: 0.25 * 23.5 / 20 = 0.29375      calf: 0.25 * 45.0 / 40 = 0.28125
///
/// Use these, NOT the ONNX metadata string -- mjlab's exporter rounds it to 3 decimals
/// (0.294 / 0.281), which is why the metadata check below tolerates 2e-3.
inline constexpr std::array<double, kNumJoints> kActionScale{{
  0.29375, 0.29375, 0.28125, 0.29375, 0.29375, 0.28125,
  0.29375, 0.29375, 0.28125, 0.29375, 0.29375, 0.28125,
}};

/// PD gains the policy was trained against (go2/constants.py:117-127). These go straight
/// into MotorCmd.kp/.kd -- the motor boards close the loop, we never compute torque.
inline constexpr std::array<double, kNumJoints> kPolicyKp{{
  20.0, 20.0, 40.0, 20.0, 20.0, 40.0, 20.0, 20.0, 40.0, 20.0, 20.0, 40.0,
}};
inline constexpr std::array<double, kNumJoints> kPolicyKd{{
  1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 1.0, 1.0, 2.0,
}};

/// Gains for the scripted stand-up. Stiffer than the policy gains because the move is a
/// slow position interpolation, not a dynamic gait. From Unitree's own deploy config.
inline constexpr std::array<double, kNumJoints> kStandKp{{
  60.0, 80.0, 80.0, 60.0, 80.0, 80.0, 60.0, 80.0, 80.0, 60.0, 80.0, 80.0,
}};
inline constexpr std::array<double, kNumJoints> kStandKd{{
  5.0, 4.0, 4.0, 5.0, 4.0, 4.0, 5.0, 4.0, 4.0, 5.0, 4.0, 4.0,
}};

/// Damping-only gains for PASSIVE and DAMPING. kp = 0, so the legs go limp but do not
/// free-fall.
inline constexpr double kDampingKd = 3.0;

/// Crouch waypoint on the way up, then the policy's default pose. Note the second
/// waypoint is kDefaultJointPos and NOT Unitree's stock [0, 0.8, -1.5] stand: the policy's
/// observations are relative to its own default pose, so that is where it must start.
inline constexpr std::array<double, kNumJoints> kCrouchJointPos{{
  0.0, 1.36, -2.65, 0.0, 1.36, -2.65, 0.0, 1.36, -2.65, 0.0, 1.36, -2.65,
}};

/// Go2 joint position limits, policy order. Targets are clamped to these.
inline constexpr std::array<double, kNumJoints> kJointPosLower{{
  -1.0472, -1.5708, -2.7227, -1.0472, -1.5708, -2.7227,
  -1.0472, -0.5236, -2.7227, -1.0472, -0.5236, -2.7227,
}};
inline constexpr std::array<double, kNumJoints> kJointPosUpper{{
  1.0472, 3.4907, -0.83776, 1.0472, 3.4907, -0.83776,
  1.0472, 4.5379, -0.83776, 1.0472, 4.5379, -0.83776,
}};

/// The training run every number in this file was last checked against. It is also the
/// ONNX metadata's `run_path`, which is how the node tells whether the checkpoint it was
/// handed is the one this header describes -- see the command envelope below for why that
/// matters.
inline constexpr std::string_view kTrainedRunPath{"2026-11-13_14-08-41_rugged_3ms"};

/// Velocity command envelope (vx m/s, vy m/s, wz rad/s), as [lower, upper] per axis.
/// go2/rugged/constants.py:143-178 -- REVERSE_SPEED, TOP_SPEED, LIN_VEL_Y, ANG_VEL_Z.
///
/// ASYMMETRIC in vx: the policy runs forward at up to 3.0 m/s but was only ever asked to
/// back up at 1.0. The forward limit is the last stage of the training ladder
/// (1.5 -> 2.5 -> 3.0, reached at iteration 2000 of 6000), so the checkpoint spent two
/// thirds of training with the whole range open.
///
/// THE ONE PART OF THE CONTRACT NOTHING CHECKS AT STARTUP. mjlab's exporter writes joint
/// order, gains, default pose, action scale and term names into the ONNX metadata, and
/// PolicyRunner validates all of them -- but not the command ranges. A checkpoint with a
/// different envelope loads cleanly and is then driven outside what it trained on. The
/// reference trace pins both corners of the envelope (test_policy_parity,
/// CommandEnvelopeMatchesTrace), and the node warns when `run_path` is not
/// kTrainedRunPath.
inline constexpr std::array<double, 3> kCommandLower{{-1.0, -0.5, -1.0}};
inline constexpr std::array<double, 3> kCommandUpper{{3.0, 0.5, 1.0}};

/// Unitree MotorCmd.mode for servo (position/velocity/torque) control on the Go2.
inline constexpr std::uint8_t kMotorModeServo = 0x01;

/// Sentinel q/dq values Unitree's own examples use on unused motor slots (12-19) --
/// "no target", not zero. Slots 12-19 must still carry kMotorModeServo with zero gains;
/// go2_stand_example (~/unitree_ros2/example/src/src/go2/go2_stand_example.cpp) leaves
/// mode 0x01 on all 20 slots and never changes it. Setting mode 0x00 there instead (as
/// this node previously did) made the Go2 silently ignore the entire command, not just
/// the absent motors -- confirmed 2026-08-01 by diffing against that example, which
/// stood the robot under otherwise-identical conditions while this node's LowCmd did
/// nothing. See ~/ros2_ws/CLAUDE.md, "The open blocker".
inline constexpr float kPosStopF = 2.146E+9F;
inline constexpr float kVelStopF = 16000.0F;

/// LowCmd header bytes and level flag, as the Go2 expects them.
inline constexpr std::uint8_t kLowCmdHead0 = 0xFE;
inline constexpr std::uint8_t kLowCmdHead1 = 0xEF;
inline constexpr std::uint8_t kLowCmdLevelFlag = 0xFF;

}  // namespace go2_rl_velocity::policy

#endif  // GO2_RL_VELOCITY__POLICY_CONSTANTS_HPP_
