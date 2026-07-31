// Copyright (c) 2026 Antonio Avila. MIT License.

#ifndef GO2_RL_VELOCITY__POLICY_RUNNER_HPP_
#define GO2_RL_VELOCITY__POLICY_RUNNER_HPP_

#include <array>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

#include "go2_rl_velocity/policy_constants.hpp"

namespace go2_rl_velocity
{

/// Thrown when the ONNX file does not match this build's constants.
class PolicyContractError : public std::runtime_error
{
public:
  explicit PolicyContractError(const std::string & what)
  : std::runtime_error(what) {}
};

/// The exported actor, evaluated on CPU.
///
/// Execution provider is CPU with a single intra-op thread, deliberately. The graph is
/// Sub -> Div -> (Gemm -> Elu) x3 -> Gemm, about 280k MACs, which measures ~13 us
/// single-threaded on x86 and well under 100 us on a Jetson Orin Nano core -- against a
/// 20 ms budget. A CUDA provider would add per-call host/device transfer and kernel launch
/// overhead exceeding the compute itself, plus scheduling jitter, for no gain.
///
/// Observation normalisation is part of the graph (the leading Sub/Div hold the training
/// run's empirical mean and std), so observations are fed in raw and unnormalised.
class PolicyRunner
{
public:
  /// Loads `model_path`, validates it against policy_constants.hpp, and warms it up.
  /// Throws PolicyContractError on any mismatch, Ort::Exception on a load failure.
  explicit PolicyRunner(const std::string & model_path, int warmup_iterations = 200);

  /// Run one inference. Returns the raw, pre-scale action.
  const std::array<float, policy::kActionDim> & run(
    const std::array<float, policy::kObsDim> & obs);

  /// Wall-clock duration of the most recent run(), in seconds.
  double last_inference_time_s() const {return last_inference_time_s_;}

  /// Metadata read out of the ONNX, for logging.
  const std::string & run_path() const {return run_path_;}

private:
  void validateContract();

  Ort::Env env_;
  Ort::SessionOptions session_options_;
  Ort::MemoryInfo memory_info_;
  std::unique_ptr<Ort::Session> session_;

  std::string input_name_;
  std::string output_name_;
  std::array<const char *, 1> input_names_{};
  std::array<const char *, 1> output_names_{};

  // Reused across calls so a control step allocates nothing.
  std::array<float, policy::kObsDim> input_buffer_{};
  std::array<float, policy::kActionDim> output_buffer_{};
  Ort::Value input_tensor_{nullptr};
  Ort::Value output_tensor_{nullptr};

  std::string run_path_{"unknown"};
  double last_inference_time_s_{0.0};
};

}  // namespace go2_rl_velocity

#endif  // GO2_RL_VELOCITY__POLICY_RUNNER_HPP_
