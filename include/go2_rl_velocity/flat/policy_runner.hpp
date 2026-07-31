// Copyright (c) 2026 Antonio Avila. MIT License.

#ifndef GO2_RL_VELOCITY__FLAT__POLICY_RUNNER_HPP_
#define GO2_RL_VELOCITY__FLAT__POLICY_RUNNER_HPP_

#include <array>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

#include "go2_rl_velocity/flat/policy_constants.hpp"

namespace go2_rl_velocity::flat
{

/// Thrown when the ONNX file does not match this build's constants.
class PolicyContractError : public std::runtime_error
{
public:
  explicit PolicyContractError(const std::string & what)
  : std::runtime_error(what) {}
};

/// The exported flat-ground actor, evaluated on CPU.
///
/// Execution provider is CPU with a single intra-op thread, deliberately. The graph is
/// Sub -> Div -> (Gemm -> Elu) x3 -> Gemm -- architecturally the same MLP as the rugged
/// policy, just with a 45-wide first layer instead of 225 -- which measures ~13 us
/// single-threaded on x86, against a 5 ms budget. A CUDA provider would add per-call
/// host/device transfer and kernel launch overhead exceeding the compute itself.
///
/// Observation normalisation is part of the graph (the leading Sub/Div hold the training
/// run's empirical mean and std), so observations are fed in raw and unnormalised.
///
/// VALIDATION IS SHAPE-ONLY, and that is a deliberate, documented limitation. mjlab's
/// exporter attaches its metadata map inside a try/except that first calls
/// `env.action_manager.get_term("joint_pos")`; the flat task's action term is named
/// "joint_effort", so the lookup raises, the exception is swallowed, and the ONNX is
/// written with an empty metadata map. There is nothing to cross-check joint order,
/// action scale or observation term names against. The rugged runner treats missing
/// metadata as "skip that check"; here every check would be skipped, so the guard on the
/// rest of the contract is test_flat_policy_parity replaying the reference trace.
class PolicyRunner
{
public:
  /// Loads `model_path`, checks its graph shape against flat/policy_constants.hpp, and
  /// warms it up. Throws PolicyContractError on a mismatch, Ort::Exception on a load
  /// failure.
  explicit PolicyRunner(const std::string & model_path, int warmup_iterations = 200);

  /// Run one inference. Returns the raw, pre-scale action (a torque coefficient).
  const std::array<float, kActionDim> & run(const std::array<float, kObsDim> & obs);

  /// Wall-clock duration of the most recent run(), in seconds.
  double last_inference_time_s() const {return last_inference_time_s_;}

  /// Training run name, if the ONNX happens to carry one. Flat exports do not, so this
  /// normally reads "unknown (flat exports carry no metadata)".
  const std::string & run_path() const {return run_path_;}

  /// True if the model carried any custom metadata at all. False for every flat export
  /// mjlab has produced so far; the node logs it so the operator knows the contract is
  /// resting entirely on the parity test.
  bool has_metadata() const {return has_metadata_;}

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
  std::array<float, kObsDim> input_buffer_{};
  std::array<float, kActionDim> output_buffer_{};
  Ort::Value input_tensor_{nullptr};
  Ort::Value output_tensor_{nullptr};

  std::string run_path_{"unknown (flat exports carry no metadata)"};
  bool has_metadata_{false};
  double last_inference_time_s_{0.0};
};

}  // namespace go2_rl_velocity::flat

#endif  // GO2_RL_VELOCITY__FLAT__POLICY_RUNNER_HPP_
