// Copyright (c) 2026 Antonio Avila. MIT License.

#include "go2_rl_velocity/flat/policy_runner.hpp"

#include <chrono>
#include <sstream>

namespace go2_rl_velocity::flat
{

PolicyRunner::PolicyRunner(const std::string & model_path, int warmup_iterations)
: env_(ORT_LOGGING_LEVEL_WARNING, "go2_rl_flat"),
  memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
{
  // Single-threaded and sequential: the graph is far too small for threading to pay for
  // itself, and a thread pool only adds jitter to a hard 200 Hz loop.
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
  const std::array<int64_t, 2> input_shape{1, kObsDim};
  const std::array<int64_t, 2> output_shape{1, kActionDim};
  input_tensor_ = Ort::Value::CreateTensor<float>(
    memory_info_, input_buffer_.data(), input_buffer_.size(),
    input_shape.data(), input_shape.size());
  output_tensor_ = Ort::Value::CreateTensor<float>(
    memory_info_, output_buffer_.data(), output_buffer_.size(),
    output_shape.data(), output_shape.size());

  // Warm up so the first inference on the robot does not pay for lazy allocation, page
  // faults or arena growth. The robot is standing still at this point; a slow first step
  // once it is walking is a fall.
  const std::array<float, kObsDim> zeros{};
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

  if (in_shape.empty() || in_shape.back() != kObsDim) {
    std::ostringstream oss;
    oss << "policy expects " << (in_shape.empty() ? -1 : in_shape.back())
        << " observations but this build packs " << kObsDim
        << " (a single " << kFrameDim << "-dim frame, no history) -- is this a rugged "
        << "checkpoint (225) rather than a flat one?";
    throw PolicyContractError(oss.str());
  }
  if (out_shape.empty() || out_shape.back() != kActionDim) {
    throw PolicyContractError("policy action dimension is not 12");
  }

  // Everything below is informational only. See the class comment: flat exports carry an
  // empty metadata map because mjlab's exporter is hard-wired to position control, so
  // there is nothing here to validate against. Read it if it happens to be present -- a
  // hand-stamped or future export would populate it -- but never fault on its absence,
  // and never treat its presence as a check that ran.
  Ort::AllocatorWithDefaultOptions allocator;
  const auto metadata = session_->GetModelMetadata();
  const auto keys = metadata.GetCustomMetadataMapKeysAllocated(allocator);
  has_metadata_ = !keys.empty();

  if (has_metadata_) {
    auto value = metadata.LookupCustomMetadataMapAllocated("run_path", allocator);
    if (value) {
      run_path_ = std::string(value.get());
    }
  }
}

const std::array<float, kActionDim> & PolicyRunner::run(
  const std::array<float, kObsDim> & obs)
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

}  // namespace go2_rl_velocity::flat
