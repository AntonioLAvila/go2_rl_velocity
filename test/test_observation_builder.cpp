// Copyright (c) 2026 Antonio Avila. MIT License.
//
// Structural tests for the history buffer. The parity test proves the layout matches the
// training environment; these pin down the behaviour that makes it easy to reason about.

#include <array>

#include <gtest/gtest.h>

#include "go2_rl_velocity/observation_builder.hpp"
#include "go2_rl_velocity/policy_constants.hpp"

namespace
{

using go2_rl_velocity::ObservationBuilder;
using go2_rl_velocity::ObservationFrame;
namespace policy = go2_rl_velocity::policy;

/// A frame whose every element equals `tag`, so a packed vector reads back as the history
/// index that produced each slot.
ObservationFrame uniformFrame(float tag)
{
  ObservationFrame frame;
  frame.joint_pos_rel.fill(tag);
  frame.joint_vel.fill(tag);
  frame.base_ang_vel.fill(tag);
  frame.projected_gravity.fill(tag);
  frame.last_action.fill(tag);
  frame.command.fill(tag);
  return frame;
}

/// A frame whose elements are 0..44, so a packed vector reveals the term layout.
ObservationFrame rampFrame()
{
  ObservationFrame frame;
  float v = 0.0F;
  for (auto & x : frame.joint_pos_rel) {x = v++;}
  for (auto & x : frame.joint_vel) {x = v++;}
  for (auto & x : frame.base_ang_vel) {x = v++;}
  for (auto & x : frame.projected_gravity) {x = v++;}
  for (auto & x : frame.last_action) {x = v++;}
  for (auto & x : frame.command) {x = v++;}
  return frame;
}

TEST(ObservationFrameTest, FlattensInTermOrder)
{
  const auto flat = rampFrame().flatten();
  for (int i = 0; i < policy::kFrameDim; ++i) {
    EXPECT_FLOAT_EQ(flat[i], static_cast<float>(i)) << "element " << i;
  }
}

TEST(ObservationBuilderTest, DimensionsAreSelfConsistent)
{
  int sum = 0;
  for (const auto & term : policy::kTerms) {
    sum += term.width;
  }
  EXPECT_EQ(sum, policy::kFrameDim);
  EXPECT_EQ(policy::kObsDim, policy::kFrameDim * policy::kHistoryLength);
}

TEST(ObservationBuilderTest, ResetBackfillsEverySlot)
{
  ObservationBuilder builder;
  EXPECT_FALSE(builder.initialised());
  builder.reset(uniformFrame(7.0F));
  EXPECT_TRUE(builder.initialised());

  for (const float v : builder.pack()) {
    EXPECT_FLOAT_EQ(v, 7.0F);
  }
}

// The layout that breaks silently: term-major, oldest first. After pushing frames tagged
// 1..5, term `t` occupies [offset*H, offset*H + width*H) and within it the tags ascend.
TEST(ObservationBuilderTest, PacksTermMajorOldestFirst)
{
  ObservationBuilder builder;
  for (int i = 1; i <= policy::kHistoryLength; ++i) {
    builder.push(uniformFrame(static_cast<float>(i)));
  }
  const auto & obs = builder.pack();

  int frame_offset = 0;
  for (std::size_t t = 0; t < policy::kTerms.size(); ++t) {
    const int width = policy::kTerms[t].width;
    const int base = frame_offset * policy::kHistoryLength;
    for (int h = 0; h < policy::kHistoryLength; ++h) {
      for (int w = 0; w < width; ++w) {
        EXPECT_FLOAT_EQ(obs[base + h * width + w], static_cast<float>(h + 1))
          << "term " << policy::kTerms[t].name << ", history " << h << ", element " << w;
      }
    }
    frame_offset += width;
  }
  EXPECT_EQ(frame_offset, policy::kFrameDim);
}

TEST(ObservationBuilderTest, TermBlocksLandAtTheDocumentedOffsets)
{
  // The offsets quoted throughout the package docs and in CLAUDE.md.
  const std::array<std::pair<int, int>, 6> expected{{
    {0, 60}, {60, 120}, {120, 135}, {135, 150}, {150, 210}, {210, 225},
  }};

  int frame_offset = 0;
  for (std::size_t t = 0; t < policy::kTerms.size(); ++t) {
    const int begin = frame_offset * policy::kHistoryLength;
    const int end = begin + policy::kTerms[t].width * policy::kHistoryLength;
    EXPECT_EQ(begin, expected[t].first) << policy::kTerms[t].name;
    EXPECT_EQ(end, expected[t].second) << policy::kTerms[t].name;
    frame_offset += policy::kTerms[t].width;
  }
}

TEST(ObservationBuilderTest, EvictsOldestOnWrap)
{
  ObservationBuilder builder;
  // Push more than the history length; only the last kHistoryLength must survive.
  for (int i = 1; i <= 12; ++i) {
    builder.push(uniformFrame(static_cast<float>(i)));
  }
  const auto & obs = builder.pack();

  // joint_pos occupies [0, 60): five blocks of 12, tags 8..12.
  for (int h = 0; h < policy::kHistoryLength; ++h) {
    for (int w = 0; w < 12; ++w) {
      EXPECT_FLOAT_EQ(obs[h * 12 + w], static_cast<float>(8 + h))
        << "history " << h;
    }
  }
}

TEST(ObservationBuilderTest, PushBeforeResetBackfills)
{
  ObservationBuilder pushed;
  pushed.push(uniformFrame(3.0F));

  ObservationBuilder reset;
  reset.reset(uniformFrame(3.0F));

  EXPECT_EQ(pushed.pack(), reset.pack());
}

TEST(ObservationBuilderTest, ResetClearsPriorHistory)
{
  ObservationBuilder builder;
  for (int i = 1; i <= 5; ++i) {
    builder.push(uniformFrame(static_cast<float>(i)));
  }
  builder.reset(uniformFrame(99.0F));
  for (const float v : builder.pack()) {
    EXPECT_FLOAT_EQ(v, 99.0F);
  }
}

}  // namespace
