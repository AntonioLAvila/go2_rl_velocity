// Copyright (c) 2026 Antonio Avila. MIT License.

#include "go2_rl_velocity/policy_runner.hpp"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <sstream>

namespace go2_rl_velocity
{

namespace
{

std::vector<std::string> splitCsv(const std::string & s)
{
  std::vector<std::string> out;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ',')) {
    out.push_back(item);
  }
  return out;
}

std::vector<double> parseFloatCsv(const std::string & s)
{
  std::vector<double> out;
  for (const auto & tok : splitCsv(s)) {
    out.push_back(std::strtod(tok.c_str(), nullptr));
  }
  return out;
}

}  // namespace

PolicyRunner::PolicyRunner(const std::string & model_path, int warmup_iterations)
: env_(ORT_LOGGING_LEVEL_WARNING, "go2_rl_velocity"),
  memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
{
  // Single-threaded and sequential: the graph is far too small for threading to pay for
  // itself, and a thread pool only adds jitter to a hard 50 Hz loop.
  session_options_.SetIntraOpNumThreads(1);
  session_options_.SetInterOpNumThreads(1);
  session_options_.SetExecutionMode(ORT_SEQUENTIAL);
  session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

  session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options_);

  Ort::AllocatorWithDefaultOptions allocator;
  input_name_ = session_->GetInputNameAllocated(0, allocator).get();
  output_name_ = session_->GetOutputNameAllocated(0, allocator).get();
  input_names_[0] = input_name_.c_str();
  output_names_[0] = output_name_.c_str();

  validateContract();

  // Bind tensors to the member buffers once; run() then reuses them.
  const std::array<int64_t, 2> input_shape{1, policy::kObsDim};
  const std::array<int64_t, 2> output_shape{1, policy::kActionDim};
  input_tensor_ = Ort::Value::CreateTensor<float>(
    memory_info_, input_buffer_.data(), input_buffer_.size(),
    input_shape.data(), input_shape.size());
  output_tensor_ = Ort::Value::CreateTensor<float>(
    memory_info_, output_buffer_.data(), output_buffer_.size(),
    output_shape.data(), output_shape.size());

  // Warm up so the first inference on the robot does not pay for lazy allocation, page
  // faults or arena growth. The robot is standing still at this point; a slow first step
  // once it is walking is a fall.
  const std::array<float, policy::kObsDim> zeros{};
  for (int i = 0; i < warmup_iterations; ++i) {
    run(zeros);
  }
}

void PolicyRunner::validateContract()
{
  if (session_->GetInputCount() != 1 || session_->GetOutputCount() != 1) {
    throw PolicyContractError("expected exactly one input and one output");
  }

  const auto in_shape =
    session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
  const auto out_shape =
    session_->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();

  if (in_shape.empty() || in_shape.back() != policy::kObsDim) {
    std::ostringstream oss;
    oss << "policy expects " << (in_shape.empty() ? -1 : in_shape.back())
        << " observations but this build packs " << policy::kObsDim
        << " (" << policy::kFrameDim << " x H" << policy::kHistoryLength
        << ") -- was HISTORY_LENGTH changed after training?";
    throw PolicyContractError(oss.str());
  }
  if (out_shape.empty() || out_shape.back() != policy::kActionDim) {
    throw PolicyContractError("policy action dimension is not 12");
  }

  Ort::AllocatorWithDefaultOptions allocator;
  const auto metadata = session_->GetModelMetadata();

  auto lookup = [&](const char * key) -> std::string {
      auto value = metadata.LookupCustomMetadataMapAllocated(key, allocator);
      return value ? std::string(value.get()) : std::string{};
    };

  const auto run_path = lookup("run_path");
  if (!run_path.empty()) {
    run_path_ = run_path;
  }

  // Observation terms, in order. This is the check that catches a retrain with a
  // different observation group -- the one failure the dimension check cannot see.
  const auto obs_names = lookup("observation_names");
  if (!obs_names.empty()) {
    const auto trained = splitCsv(obs_names);
    if (trained.size() != policy::kTerms.size()) {
      throw PolicyContractError(
              "policy has " + std::to_string(trained.size()) + " observation terms, expected " +
              std::to_string(policy::kTerms.size()));
    }
    for (std::size_t i = 0; i < trained.size(); ++i) {
      if (trained[i] != policy::kTerms[i].name) {
        throw PolicyContractError(
                "observation term " + std::to_string(i) + " is '" + trained[i] +
                "', expected '" + std::string(policy::kTerms[i].name) + "'");
      }
    }
  }

  const auto joint_names = lookup("joint_names");
  if (!joint_names.empty()) {
    const auto trained = splitCsv(joint_names);
    if (trained.size() != policy::kJointNames.size()) {
      throw PolicyContractError("policy joint count is not 12");
    }
    for (std::size_t i = 0; i < trained.size(); ++i) {
      if (trained[i] != policy::kJointNames[i]) {
        throw PolicyContractError(
                "joint order drifted at index " + std::to_string(i) + ": policy has '" +
                trained[i] + "', this build assumes '" +
                std::string(policy::kJointNames[i]) + "'");
      }
    }
  }

  auto checkFloats = [&](const char * key, const std::array<double, policy::kNumJoints> & ours,
      double tolerance) {
      const auto raw = lookup(key);
      if (raw.empty()) {
        return;
      }
      const auto trained = parseFloatCsv(raw);
      if (trained.size() != ours.size()) {
        throw PolicyContractError(std::string(key) + " has the wrong length");
      }
      for (std::size_t i = 0; i < ours.size(); ++i) {
        if (std::fabs(trained[i] - ours[i]) > tolerance) {
          std::ostringstream oss;
          oss << key << " drifted at index " << i << ": policy has " << trained[i]
              << ", this build assumes " << ours[i];
          throw PolicyContractError(oss.str());
        }
      }
    };

  checkFloats("default_joint_pos", policy::kDefaultJointPos, 1e-3);
  checkFloats("joint_stiffness", policy::kPolicyKp, 1e-3);
  checkFloats("joint_damping", policy::kPolicyKd, 1e-3);
  // mjlab's exporter formats the action scale with 3 decimals (0.294 vs 0.29375), so this
  // one gets a looser tolerance. kActionScale carries the exact value.
  checkFloats("action_scale", policy::kActionScale, 2e-3);
}

const std::array<float, policy::kActionDim> & PolicyRunner::run(
  const std::array<float, policy::kObsDim> & obs)
{
  input_buffer_ = obs;

  const auto start = std::chrono::steady_clock::now();
  session_->Run(
    Ort::RunOptions{nullptr},
    input_names_.data(), &input_tensor_, 1,
    output_names_.data(), &output_tensor_, 1);
  last_inference_time_s_ =
    std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

  return output_buffer_;
}

}  // namespace go2_rl_velocity
