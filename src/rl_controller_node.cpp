// Copyright (c) 2026 Antonio Avila. MIT License.

#include "go2_rl_velocity/rl_controller_node.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <utility>
#include <vector>

#include <ament_index_cpp/get_package_share_directory.hpp>

#include "go2_rl_velocity/crc32.hpp"
#include "go2_rl_velocity/state_conversion.hpp"

namespace go2_rl_velocity
{

namespace
{

/// LowCmd/LowState carry 20 motor slots; the Go2 uses the first 12. The rest must be
/// explicitly idled, not just left default-constructed.
constexpr std::size_t kMotorSlots = 20;

/// Number of 32-bit words the LowCmd checksum covers: the whole message except its own
/// trailing crc field.
constexpr std::size_t kLowCmdCrcWords = (sizeof(unitree_go::msg::LowCmd) >> 2) - 1;

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

RlControllerNode::RlControllerNode()
: rclcpp::Node("go2_rl_velocity")
{
  declareParameters();

  const auto policy_path = resolvePolicyPath();
  RCLCPP_INFO(get_logger(), "loading policy: %s", policy_path.c_str());

  // Any contract mismatch throws here, before a single LowCmd goes out.
  policy_ = std::make_unique<PolicyRunner>(
    policy_path, static_cast<int>(get_parameter("warmup_iterations").as_int()));
  RCLCPP_INFO(
    get_logger(), "policy OK: run '%s', %d-dim observation (%d x H%d), CPU execution",
    policy_->run_path().c_str(), policy::kObsDim, policy::kFrameDim,
    policy::kHistoryLength);

  ControllerFsm::Config fsm_config;
  fsm_config.stand_segment_s = get_parameter("stand_segment_s").as_double();
  fsm_config.gain_ramp_s = get_parameter("gain_ramp_s").as_double();
  fsm_config.command_warmup_s = get_parameter("command_warmup_s").as_double();
  fsm_config.tilt_limit = get_parameter("tilt_limit").as_double();
  fsm_config.joint_clamp_fault_steps =
    static_cast<int>(get_parameter("joint_clamp_fault_steps").as_int());
  fsm_config.inference_warn_s = get_parameter("inference_warn_s").as_double();
  fsm_config.inference_overrun_fault_steps =
    static_cast<int>(get_parameter("inference_overrun_fault_steps").as_int());
  fsm_config.damping_settle_s = get_parameter("damping_settle_s").as_double();
  fsm_config.use_joystick_command = get_parameter("use_joystick_command").as_bool();

  const auto accel = get_parameter("command_accel_limit").as_double_array();
  const auto signs = get_parameter("joystick_sign").as_double_array();
  if (accel.size() == 3) {
    fsm_config.command_accel_limit = {accel[0], accel[1], accel[2]};
  }
  if (signs.size() == 3) {
    fsm_config.joystick_sign = {signs[0], signs[1], signs[2]};
  }

  fsm_ = std::make_unique<ControllerFsm>(fsm_config, *policy_);

  lowcmd_rate_hz_ = get_parameter("lowcmd_rate_hz").as_double();
  lowstate_timeout_s_ = get_parameter("lowstate_timeout_s").as_double();
  cmd_vel_timeout_s_ = get_parameter("cmd_vel_timeout_s").as_double();
  remote_debug_ = get_parameter("remote_debug").as_bool();
  dry_run_ = get_parameter("dry_run").as_bool();

  if (dry_run_) {
    RCLCPP_WARN(
      get_logger(),
      "dry_run is set: the controller runs normally but publishes NO LowCmd");
  }

  // Newest-wins. A queued LowState backlog would feed the policy stale joint angles,
  // which is worse than dropping frames.
  rclcpp::QoS state_qos(rclcpp::KeepLast(1));
  state_qos.best_effort();

  lowstate_sub_ = create_subscription<unitree_go::msg::LowState>(
    "lowstate", state_qos,
    std::bind(&RlControllerNode::onLowState, this, std::placeholders::_1));
  cmd_vel_sub_ = create_subscription<geometry_msgs::msg::Twist>(
    "cmd_vel", rclcpp::QoS(rclcpp::KeepLast(1)),
    std::bind(&RlControllerNode::onCmdVel, this, std::placeholders::_1));

  rclcpp::QoS cmd_qos(rclcpp::KeepLast(1));
  cmd_qos.reliable();
  lowcmd_pub_ = create_publisher<unitree_go::msg::LowCmd>("lowcmd", cmd_qos);
  status_pub_ = create_publisher<std_msgs::msg::String>("~/status", rclcpp::QoS(10));

  using namespace std::chrono_literals;
  const auto policy_period = std::chrono::duration<double>(policy::kControlDt);
  const auto publish_period = std::chrono::duration<double>(1.0 / lowcmd_rate_hz_);

  policy_timer_ = create_wall_timer(
    std::chrono::duration_cast<std::chrono::nanoseconds>(policy_period),
    std::bind(&RlControllerNode::onPolicyTimer, this));
  publish_timer_ = create_wall_timer(
    std::chrono::duration_cast<std::chrono::nanoseconds>(publish_period),
    std::bind(&RlControllerNode::onPublishTimer, this));
  status_timer_ = create_wall_timer(
    500ms, std::bind(&RlControllerNode::onStatusTimer, this));

  estop_srv_ = create_service<std_srvs::srv::Trigger>(
    "~/estop",
    std::bind(
      &RlControllerNode::handleEstop, this, std::placeholders::_1, std::placeholders::_2));

  auto make_state_service = [this](const std::string & name, ControllerState target) {
      return create_service<std_srvs::srv::Trigger>(
        name,
        [this, target](
          const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
          std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
          handleSetState(request, response, target);
        });
    };
  passive_srv_ = make_state_service("~/passive", ControllerState::Passive);
  stand_srv_ = make_state_service("~/stand", ControllerState::FixStand);
  start_policy_srv_ = make_state_service("~/start_policy", ControllerState::Rl);

  RCLCPP_INFO(
    get_logger(),
    "ready. LowCmd at %.0f Hz, policy at %.0f Hz. "
    "Press L2+UP to stand, then R2+A to hand over to the policy. L2+B stops.",
    lowcmd_rate_hz_, 1.0 / policy::kControlDt);
}

void RlControllerNode::declareParameters()
{
  declare_parameter<std::string>("policy_path", "");
  declare_parameter<int>("warmup_iterations", 200);
  declare_parameter<double>("lowcmd_rate_hz", 500.0);
  declare_parameter<double>("lowstate_timeout_s", 0.1);
  declare_parameter<double>("cmd_vel_timeout_s", 0.5);
  declare_parameter<double>("stand_segment_s", 1.0);
  declare_parameter<double>("gain_ramp_s", 0.5);
  declare_parameter<double>("command_warmup_s", 0.5);
  declare_parameter<double>("tilt_limit", -0.6);
  declare_parameter<int>("joint_clamp_fault_steps", 25);
  declare_parameter<double>("inference_warn_s", 0.010);
  declare_parameter<int>("inference_overrun_fault_steps", 10);
  declare_parameter<double>("damping_settle_s", 1.5);
  declare_parameter<bool>("use_joystick_command", true);
  declare_parameter<std::vector<double>>("command_accel_limit", {3.0, 3.0, 6.0});
  declare_parameter<std::vector<double>>("joystick_sign", {1.0, -1.0, -1.0});
  declare_parameter<bool>("remote_debug", false);
  declare_parameter<bool>("dry_run", false);
}

std::string RlControllerNode::resolvePolicyPath() const
{
  auto path = get_parameter("policy_path").as_string();
  if (path.empty()) {
    path = ament_index_cpp::get_package_share_directory("go2_rl_velocity") +
      "/config/policy/rugged-v2.onnx";
  }
  if (!std::filesystem::exists(path)) {
    throw std::runtime_error("policy file does not exist: " + path);
  }
  return path;
}

double RlControllerNode::now() const
{
  return static_cast<double>(get_clock()->now().nanoseconds()) * 1e-9;
}

void RlControllerNode::onLowState(unitree_go::msg::LowState::ConstSharedPtr msg)
{
  std::lock_guard<std::mutex> lock(state_mutex_);
  latest_lowstate_ = std::move(msg);
  latest_lowstate_time_s_ = now();
}

void RlControllerNode::onCmdVel(geometry_msgs::msg::Twist::ConstSharedPtr msg)
{
  std::lock_guard<std::mutex> lock(state_mutex_);
  latest_cmd_vel_ = {
    static_cast<float>(msg->linear.x),
    static_cast<float>(msg->linear.y),
    static_cast<float>(msg->angular.z),
  };
  latest_cmd_vel_time_s_ = now();
}

void RlControllerNode::onPolicyTimer()
{
  unitree_go::msg::LowState::ConstSharedPtr state;
  double state_age = 0.0;
  std::array<float, 3> cmd_vel{};
  bool cmd_vel_fresh = false;

  const double t = now();
  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    state = latest_lowstate_;
    state_age = t - latest_lowstate_time_s_;
    cmd_vel = latest_cmd_vel_;
    cmd_vel_fresh = (t - latest_cmd_vel_time_s_) < cmd_vel_timeout_s_;
  }

  if (!state) {
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 2000, "no LowState received yet; is the bridge running?");
    return;
  }

  RobotObservation obs;
  obs.lowstate_fresh = state_age < lowstate_timeout_s_;

  // Unitree motor order -> policy order.
  std::array<float, policy::kNumJoints> q_unitree{};
  std::array<float, policy::kNumJoints> dq_unitree{};
  for (int i = 0; i < policy::kNumJoints; ++i) {
    q_unitree[i] = state->motor_state[i].q;
    dq_unitree[i] = state->motor_state[i].dq;
  }
  obs.joint_pos = unitreeToPolicy(q_unitree);
  obs.joint_vel = unitreeToPolicy(dq_unitree);

  for (int i = 0; i < 3; ++i) {
    obs.base_ang_vel[i] = state->imu_state.gyroscope[i];
  }
  for (int i = 0; i < 4; ++i) {
    obs.imu_quaternion[i] = state->imu_state.quaternion[i];
  }

  std::array<std::uint8_t, 40> remote_raw{};
  std::copy_n(state->wireless_remote.begin(), remote_raw.size(), remote_raw.begin());
  obs.remote = UnitreeRemote::decode(remote_raw);

  obs.external_command = cmd_vel;
  obs.external_command_fresh = cmd_vel_fresh;

  if (remote_debug_) {
    RCLCPP_INFO_THROTTLE(
      get_logger(), *get_clock(), 250, "remote: %s", obs.remote.describe().c_str());
  }

  const auto previous_state = fsm_->state();
  fsm_->step(obs, t);

  if (fsm_->state() != previous_state) {
    if (fsm_->state() == ControllerState::Damping && !fsm_->faultReason().empty()) {
      RCLCPP_ERROR(
        get_logger(), "%s -> DAMPING: %s", toString(previous_state),
        fsm_->faultReason().c_str());
    } else {
      RCLCPP_INFO(get_logger(), "%s -> %s", toString(previous_state), toString(fsm_->state()));
    }
  }

  {
    std::lock_guard<std::mutex> lock(command_mutex_);
    held_command_ = fsm_->command();
    have_command_ = true;
  }
}

void RlControllerNode::fillLowCmd(unitree_go::msg::LowCmd & cmd) const
{
  cmd.head[0] = policy::kLowCmdHead0;
  cmd.head[1] = policy::kLowCmdHead1;
  cmd.level_flag = policy::kLowCmdLevelFlag;
  cmd.gpio = 0;

  MotorCommand command;
  {
    std::lock_guard<std::mutex> lock(command_mutex_);
    command = held_command_;
  }

  const auto target = policyToUnitree(command.joint_target);
  const auto kp = policyToUnitree(command.kp);
  const auto kd = policyToUnitree(command.kd);

  for (std::size_t i = 0; i < kMotorSlots; ++i) {
    auto & motor = cmd.motor_cmd[i];
    if (i < static_cast<std::size_t>(policy::kNumJoints)) {
      motor.mode = policy::kMotorModeServo;
      motor.q = target[i];
      motor.kp = kp[i];
      motor.kd = kd[i];
    } else {
      // Slots 12-19 are unused on the Go2. Leaving them at a non-idle mode with non-zero
      // gains is how a stray command reaches a motor that is not there.
      motor.mode = policy::kMotorModeIdle;
      motor.q = 0.0F;
      motor.kp = 0.0F;
      motor.kd = 0.0F;
    }
    motor.dq = 0.0F;
    motor.tau = 0.0F;
  }

  // The Go2 silently drops any LowCmd with a bad CRC, so this must be last.
  cmd.crc = crc32Core(&cmd, kLowCmdCrcWords);
}

void RlControllerNode::onPublishTimer()
{
  {
    std::lock_guard<std::mutex> lock(command_mutex_);
    if (!have_command_) {
      // Nothing computed yet: stay silent rather than publish zero gains against a
      // zero target, which the motor boards would accept.
      return;
    }
  }
  if (dry_run_) {
    return;
  }

  unitree_go::msg::LowCmd cmd;
  fillLowCmd(cmd);
  lowcmd_pub_->publish(cmd);
}

void RlControllerNode::onStatusTimer()
{
  const auto state = fsm_->state();
  const auto & cmd = fsm_->activeCommand();

  std_msgs::msg::String msg;
  msg.data = std::string(toString(state)) +
    " cmd=[" + std::to_string(cmd[0]) + ", " + std::to_string(cmd[1]) + ", " +
    std::to_string(cmd[2]) + "]" +
    " inference=" + std::to_string(policy_->last_inference_time_s() * 1e6) + "us";
  if (!fsm_->faultReason().empty()) {
    msg.data += " last_fault='" + fsm_->faultReason() + "'";
  }
  status_pub_->publish(msg);
}

void RlControllerNode::handleSetState(
  const std::shared_ptr<std_srvs::srv::Trigger::Request>,
  std::shared_ptr<std_srvs::srv::Trigger::Response> response,
  ControllerState target)
{
  std::string reason;
  if (fsm_->requestState(target, now(), reason)) {
    response->success = true;
    response->message = std::string("entering ") + toString(target);
    RCLCPP_INFO(get_logger(), "service request: %s", response->message.c_str());
  } else {
    response->success = false;
    response->message = reason;
    RCLCPP_WARN(get_logger(), "service request rejected: %s", reason.c_str());
  }
}

void RlControllerNode::handleEstop(
  const std::shared_ptr<std_srvs::srv::Trigger::Request>,
  std::shared_ptr<std_srvs::srv::Trigger::Response> response)
{
  fsm_->fault("e-stop service called", now());
  response->success = true;
  response->message = "damping";
  RCLCPP_ERROR(get_logger(), "e-stop service called");
}

}  // namespace go2_rl_velocity
