// Copyright (c) 2026 Antonio Avila. MIT License.
//
// The deployment-side mirror of the FLAT-GROUND policy's contract.
//
// This is the sibling of ../policy_constants.hpp, which serves the rugged policy. The two
// are kept apart deliberately, values duplicated rather than shared, so that a number
// changed for one policy cannot silently move the other. Upstream sources (relative to
// the quadruped_walking repo):
//
//   go2/constants.py           joint names, default pose, actuator table, GO2_ACTION_SCALE
//   go2/flat/env_cfg.py        observation terms, action term, command bounds, timing
//   scripts/verify_flat.py     the reference deployment implementation (Drake)
//
// There is no go2/flat/constants.py -- the flat task's numbers live partly in the shared
// go2/constants.py and partly as literals in go2/flat/env_cfg.py. That is why this header
// cites two files where the rugged one cites one.
//
// UNLIKE the rugged policy, the flat exports carry NO ONNX metadata at all: mjlab's
// get_base_metadata() asserts a "joint_pos" action term, the flat task's is
// "joint_effort", and the resulting exception is swallowed by the runner's save(). So
// FlatPolicyRunner can only check the graph's shape. Everything else in this header is
// guarded solely by test_flat_policy_parity and the reference trace it replays --
// regenerate that trace after ANY retrain.

#ifndef GO2_RL_VELOCITY__FLAT__POLICY_CONSTANTS_HPP_
#define GO2_RL_VELOCITY__FLAT__POLICY_CONSTANTS_HPP_

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace go2_rl_velocity::flat
{

/// Joints driven by the policy. The Go2 LowState/LowCmd motor arrays are 20 long; the
/// remaining 8 entries are unused and must be explicitly zeroed.
inline constexpr int kNumJoints = 12;

/// One observation term: its name as mjlab declares it, and its width.
struct Term
{
  std::string_view name;
  int width;
};

/// THE LAYOUT CONTRACT (go2/flat/env_cfg.py:90-108).
///
/// The same six terms, in the same order, as the rugged policy's per-frame layout -- but
/// the flat task stacks NO history (`history_length: 0` on every term in the checkpoint's
/// frozen env.yaml), so this 45-dim frame *is* the whole observation:
///
///   [joint_pos | joint_vel | ang_vel | gravity | last_action | command]
///
/// Base *linear* velocity is deliberately absent: it is not reliably observable on the
/// real Go2, so the actor was trained blind to it and only the critic ever saw it.
inline constexpr std::array<Term, 6> kTerms{{
  {"joint_pos", 12},          // q - kDefaultJointPos
  {"joint_vel", 12},          // qd
  {"base_ang_vel", 3},        // IMU gyro, body frame
  {"projected_gravity", 3},   // R^T * (0, 0, -1)
  {"last_action", 12},        // RAW pre-scale a_{t-1}
  {"command", 3},             // [vx, vy, wz]
}};

inline constexpr int kFrameDim = 45;   // sum of kTerms widths
inline constexpr int kObsDim = kFrameDim;   // no history: HISTORY_LENGTH == 1
inline constexpr int kActionDim = kNumJoints;

/// Offset of a term within the observation vector.
inline constexpr int termOffset(std::size_t term_index)
{
  int offset = 0;
  for (std::size_t i = 0; i < term_index; ++i) {
    offset += kTerms[i].width;
  }
  return offset;
}

/// Control period. go2/constants.py:131-133, SIM_TIMESTEP 0.0025 x DECIMATION 2 = 200 Hz.
///
/// Four times faster than the rugged policy. The inference itself is nowhere near the
/// budget, but at 5 ms the whole chain -- LowState delivery, DDS, scheduler jitter -- has
/// to fit inside one period.
inline constexpr double kControlDt = 0.005;

/// Joint order the policy was trained in: FL, FR, RL, RR x hip, thigh, calf.
/// go2/constants.py:25,50,57. Identical to the rugged policy's.
inline constexpr std::array<std::string_view, kNumJoints> kJointNames{{
  "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
  "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
  "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
  "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
}};

/// Policy joint index -> Unitree LowState/LowCmd motor index.
///
/// The Go2 low-level SDK orders motors FR(0-2), FL(3-5), RR(6-8), RL(9-11) while the
/// policy uses FL, FR, RL, RR. Omitting this remap swaps left and right legs. (The
/// permutation swaps two pairs of blocks, so it is its own inverse.)
inline constexpr std::array<int, kNumJoints> kPolicyToUnitree{{
  3, 4, 5,      // FL -> Unitree FL
  0, 1, 2,      // FR -> Unitree FR
  9, 10, 11,    // RL -> Unitree RL
  6, 7, 8,      // RR -> Unitree RR
}};

/// Home keyframe pose, per leg [hip=0, thigh=0.9, calf=-1.8]. go2/constants.py:53.
/// joint_pos observations are relative to this.
inline constexpr std::array<double, kNumJoints> kDefaultJointPos{{
  0.0, 0.9, -1.8, 0.0, 0.9, -1.8, 0.0, 0.9, -1.8, 0.0, 0.9, -1.8,
}};

/// THE CONTROL LAW, and the deepest difference from the rugged policy.
///
/// mjlab's JointEffortActionCfg with scale = GO2_ACTION_SCALE and offset 0, i.e.
/// go2/constants.py:123-125: "action ~[-1, 1] maps to +/- the effort limit, so the scale
/// *is* the effort limit". The policy's output is a JOINT TORQUE in N*m, not a position.
///
/// Nothing here closes a position loop. `tau` goes into MotorCmd.tau with kp = 0.
inline constexpr std::array<double, kNumJoints> kTorqueScale{{
  23.5, 23.5, 45.0, 23.5, 23.5, 45.0,
  23.5, 23.5, 45.0, 23.5, 23.5, 45.0,
}};

/// Torque saturation, standing in for MuJoCo's <motor ctrlrange> (go2/robot.py:73-76) and
/// for Drake's Saturation block in scripts/verify_flat.py. Same numbers as the scale,
/// because the scale is the effort limit -- an action outside [-1, 1] is what saturates.
inline constexpr std::array<double, kNumJoints> kTorqueLimit = kTorqueScale;

/// Gains for the RL state.
///
/// kp is zero: there is no position loop, the policy commands torque directly.
///
/// kd is NOT zero, and this is load-bearing rather than cosmetic. Every joint in training
/// carried GO2_ACTUATORS[*].damping = 2.0 N*m*s/rad of passive viscous damping
/// (go2/constants.py:101-111), which the real Go2 does not have. The Go2 motor board
/// computes tau_total = kp*(q_des - q) + kd*(dq_des - dq) + tau_ff, so sending kd = 2.0
/// with dq_des = 0 reproduces sim's -2.0*qd exactly. At a 10 rad/s swing velocity that is
/// ~20 N*m, comparable to the effort limit; the policy was trained leaning on it, and
/// omitting it lets the same torques produce far faster joint motion than it ever saw.
inline constexpr std::array<double, kNumJoints> kPolicyKp{{
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
}};
inline constexpr std::array<double, kNumJoints> kPolicyKd{{
  2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
}};

/// Gains for the scripted stand-up. Standing is still POSITION controlled -- getting off
/// the ground is not the policy's job in either task -- so these are unchanged from the
/// rugged controller, and come from Unitree's own deploy config.
inline constexpr std::array<double, kNumJoints> kStandKp{{
  60.0, 80.0, 80.0, 60.0, 80.0, 80.0, 60.0, 80.0, 80.0, 60.0, 80.0, 80.0,
}};
inline constexpr std::array<double, kNumJoints> kStandKd{{
  5.0, 4.0, 4.0, 5.0, 4.0, 4.0, 5.0, 4.0, 4.0, 5.0, 4.0, 4.0,
}};

/// Damping-only gains for PASSIVE and DAMPING. kp = 0 and tau = 0, so the legs go limp
/// but do not free-fall.
inline constexpr double kDampingKd = 3.0;

/// Crouch waypoint on the way up, then the policy's default pose. The second waypoint is
/// kDefaultJointPos and NOT Unitree's stock [0, 0.8, -1.5]: the policy's observations are
/// relative to its own default pose, so that is where it must start.
inline constexpr std::array<double, kNumJoints> kCrouchJointPos{{
  0.0, 1.36, -2.65, 0.0, 1.36, -2.65, 0.0, 1.36, -2.65, 0.0, 1.36, -2.65,
}};

/// Go2 joint position limits, policy order.
inline constexpr std::array<double, kNumJoints> kJointPosLower{{
  -1.0472, -1.5708, -2.7227, -1.0472, -1.5708, -2.7227,
  -1.0472, -0.5236, -2.7227, -1.0472, -0.5236, -2.7227,
}};
inline constexpr std::array<double, kNumJoints> kJointPosUpper{{
  1.0472, 3.4907, -0.83776, 1.0472, 3.4907, -0.83776,
  1.0472, 4.5379, -0.83776, 1.0472, 4.5379, -0.83776,
}};

/// go2/robot.py:91, soft_joint_pos_limit_factor. mjlab shrinks the range about its
/// midpoint by this factor and penalises the policy for leaving it, so in training the
/// joints effectively lived inside the soft band.
///
/// The rugged controller enforces limits by clamping the position TARGET. A torque policy
/// has no target to clamp, so the flat controller instead watches the measured position
/// against this band and faults if the policy sits outside it -- see FlatControllerFsm.
inline constexpr double kSoftJointLimitFactor = 0.95;

inline constexpr double softJointPosLower(int i)
{
  const double mid = 0.5 * (kJointPosLower[i] + kJointPosUpper[i]);
  return mid - 0.5 * (kJointPosUpper[i] - kJointPosLower[i]) * kSoftJointLimitFactor;
}
inline constexpr double softJointPosUpper(int i)
{
  const double mid = 0.5 * (kJointPosLower[i] + kJointPosUpper[i]);
  return mid + 0.5 * (kJointPosUpper[i] - kJointPosLower[i]) * kSoftJointLimitFactor;
}

/// Velocity command envelope (vx m/s, vy m/s, wz rad/s). go2/flat/env_cfg.py:176-181,
/// Go2VelocityCommandCfg(bounds=(1.5, 0.8, 1.2)), fixed -- the flat task has no command
/// curriculum. Tighter than the rugged policy's 2.0/1.0/1.5.
inline constexpr std::array<double, 3> kCommandBounds{{1.5, 0.8, 1.2}};

/// Upper bound on projected_gravity z before the tilt watchdog fires.
///
/// go2/flat/env_cfg.py terminates on bad_orientation(limit_angle = pi/4), i.e. the policy
/// was killed in training the moment it passed 45 degrees: cos(pi/4) = 0.7071, and
/// gravity z is -cos(tilt). Tighter than the rugged controller's -0.6 (~53 degrees),
/// because the flat policy has no experience past this angle.
inline constexpr double kTiltLimit = -0.7071;

}  // namespace go2_rl_velocity::flat

#endif  // GO2_RL_VELOCITY__FLAT__POLICY_CONSTANTS_HPP_
