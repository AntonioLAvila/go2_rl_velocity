// Copyright (c) 2026 Antonio Avila. MIT License.

#include <sys/mman.h>

#include <cerrno>
#include <cstring>
#include <memory>

#include <rclcpp/rclcpp.hpp>

#include "go2_rl_velocity/rl_controller_node.hpp"

namespace
{

/// Pin the process's pages in RAM.
///
/// A major page fault inside the 20 ms control step is a missed policy tick, and on a
/// Jetson under memory pressure that is a real possibility. Failure is logged rather
/// than fatal: it needs RLIMIT_MEMLOCK headroom, which a default install may not grant,
/// and running unlocked is better than not running.
void tryLockMemory(const rclcpp::Logger & logger)
{
  if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
    RCLCPP_WARN(
      logger,
      "mlockall failed (%s); continuing without locked memory. Raise RLIMIT_MEMLOCK "
      "(see README.md) to remove page-fault jitter from the control loop.",
      std::strerror(errno));
  } else {
    RCLCPP_INFO(logger, "memory locked");
  }
}

}  // namespace

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  const auto logger = rclcpp::get_logger("go2_rl_velocity");
  tryLockMemory(logger);

  int exit_code = 0;
  try {
    auto node = std::make_shared<go2_rl_velocity::RlControllerNode>();
    // Single-threaded: the two timers and the LowState callback are all short, and a
    // single thread removes any question of what the FSM's state is being mutated from.
    rclcpp::spin(node);
  } catch (const go2_rl_velocity::PolicyContractError & error) {
    RCLCPP_FATAL(
      logger,
      "the policy does not match this build's constants: %s\n"
      "Refusing to start. Either the ONNX is from a different training run, or "
      "policy_constants.hpp has drifted from the training repo.",
      error.what());
    exit_code = 1;
  } catch (const std::exception & error) {
    RCLCPP_FATAL(logger, "fatal: %s", error.what());
    exit_code = 1;
  }

  rclcpp::shutdown();
  return exit_code;
}
