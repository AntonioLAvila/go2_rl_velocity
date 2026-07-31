// Copyright (c) 2026 Antonio Avila. MIT License.

#ifndef GO2_RL_VELOCITY__RL_CONTROLLER_NODE_HPP_
#define GO2_RL_VELOCITY__RL_CONTROLLER_NODE_HPP_

#include <memory>
#include <mutex>
#include <string>

#include <geometry_msgs/msg/twist.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <unitree_go/msg/low_cmd.hpp>
#include <unitree_go/msg/low_state.hpp>

#include "go2_rl_velocity/controller_fsm.hpp"
#include "go2_rl_velocity/policy_runner.hpp"

namespace go2_rl_velocity
{

/// Runs the trained policy on a Go2 through the unitree_ros2 low-level bridge.
///
/// Two rates, mirroring Unitree's own deploy stack:
///
///   * the LowCmd publisher runs at `lowcmd_rate_hz` (500 Hz by default) and always sends
///     the currently held target, whatever state the controller is in. The motor boards
///     close the PD loop at ~1 kHz off the last command they received, and want feeding
///     continuously.
///   * the policy runs at 50 Hz, matching the training decimation, and only recomputes
///     the target. Running inference at the publish rate would be four times faster than
///     anything the policy saw in training.
///
/// LowState arrives at ~500 Hz and is latched, newest-wins, rather than queued.
class RlControllerNode : public rclcpp::Node
{
public:
  RlControllerNode();

private:
  void declareParameters();
  std::string resolvePolicyPath() const;

  void onLowState(unitree_go::msg::LowState::ConstSharedPtr msg);
  void onCmdVel(geometry_msgs::msg::Twist::ConstSharedPtr msg);
  void onPolicyTimer();
  void onPublishTimer();
  void onStatusTimer();

  void handleSetState(
    const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
    std::shared_ptr<std_srvs::srv::Trigger::Response> response,
    ControllerState target);
  void handleEstop(
    const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
    std::shared_ptr<std_srvs::srv::Trigger::Response> response);

  /// Populate a LowCmd from the FSM's held command, including the joint remap and CRC.
  void fillLowCmd(unitree_go::msg::LowCmd & cmd) const;

  double now() const;

  // Configuration.
  double lowcmd_rate_hz_{500.0};
  double lowstate_timeout_s_{0.1};
  double cmd_vel_timeout_s_{0.5};
  bool remote_debug_{false};
  bool dry_run_{false};

  // Latched inputs.
  mutable std::mutex state_mutex_;
  unitree_go::msg::LowState::ConstSharedPtr latest_lowstate_;
  double latest_lowstate_time_s_{0.0};
  std::array<float, 3> latest_cmd_vel_{};
  double latest_cmd_vel_time_s_{0.0};

  // Held output, written by the policy timer and read by the publish timer.
  mutable std::mutex command_mutex_;
  MotorCommand held_command_{};
  bool have_command_{false};

  std::unique_ptr<PolicyRunner> policy_;
  std::unique_ptr<ControllerFsm> fsm_;

  rclcpp::Subscription<unitree_go::msg::LowState>::SharedPtr lowstate_sub_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_sub_;
  rclcpp::Publisher<unitree_go::msg::LowCmd>::SharedPtr lowcmd_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr status_pub_;

  rclcpp::TimerBase::SharedPtr policy_timer_;
  rclcpp::TimerBase::SharedPtr publish_timer_;
  rclcpp::TimerBase::SharedPtr status_timer_;

  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr estop_srv_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr stand_srv_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr start_policy_srv_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr passive_srv_;
};

}  // namespace go2_rl_velocity

#endif  // GO2_RL_VELOCITY__RL_CONTROLLER_NODE_HPP_
