// Tests for Conv1x1Fixed (templated implementation)

#include <Eigen/Dense>
#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "NAM/conv1x1_factory.h"
#include "NAM/conv1x1_fixed.h"
#include "NAM/dsp.h"
#include "allocation_tracking.h"

namespace test_conv1x1_fixed
{

// Test factory creation
void test_factory_create()
{
  auto conv = nam::Conv1x1Factory::create(4, 4, true, 1);
  assert(conv != nullptr);
  assert(conv->get_in_channels() == 4);
  assert(conv->get_out_channels() == 4);
}

// Test factory with groups
void test_factory_create_with_groups()
{
  auto conv = nam::Conv1x1Factory::create(8, 8, false, 2);
  assert(conv != nullptr);
  assert(conv->get_in_channels() == 8);
  assert(conv->get_out_channels() == 8);
}

// Test process gives same result as dynamic implementation
void test_numerical_equivalence()
{
  const int in_channels = 4;
  const int out_channels = 4;
  const bool do_bias = true;
  const int groups = 1;
  const int num_frames = 4;

  // Create both implementations
  nam::Conv1x1 conv_dynamic(in_channels, out_channels, do_bias, groups);
  auto conv_fixed = nam::Conv1x1Factory::create(in_channels, out_channels, do_bias, groups);

  // Same weights
  std::vector<float> weights;
  for (int i = 0; i < out_channels * in_channels; i++)
  {
    weights.push_back(static_cast<float>(i) * 0.1f);
  }
  for (int i = 0; i < out_channels; i++)
  {
    weights.push_back(static_cast<float>(i) * 0.5f);
  }

  auto it1 = weights.begin();
  auto it2 = weights.begin();
  conv_dynamic.set_weights_(it1);
  conv_fixed->set_weights_(it2);

  conv_dynamic.SetMaxBufferSize(num_frames);
  conv_fixed->SetMaxBufferSize(num_frames);

  // Same input
  Eigen::MatrixXf input(in_channels, num_frames);
  for (int i = 0; i < in_channels; i++)
    for (int j = 0; j < num_frames; j++)
      input(i, j) = static_cast<float>(i * num_frames + j);

  // Process both
  Eigen::MatrixXf output_dynamic = conv_dynamic.process(input, num_frames);
  Eigen::MatrixXf output_fixed = conv_fixed->process(input, num_frames);

  // Compare outputs
  assert(output_dynamic.rows() == output_fixed.rows());
  assert(output_dynamic.cols() == output_fixed.cols());

  for (int i = 0; i < output_dynamic.rows(); i++)
  {
    for (int j = 0; j < output_dynamic.cols(); j++)
    {
      float diff = std::abs(output_dynamic(i, j) - output_fixed(i, j));
      assert(diff < 1e-5f);
    }
  }
}

// Test grouped convolution numerical equivalence
void test_numerical_equivalence_grouped()
{
  const int in_channels = 8;
  const int out_channels = 8;
  const bool do_bias = true;
  const int groups = 2;
  const int num_frames = 4;

  // Create both implementations
  nam::Conv1x1 conv_dynamic(in_channels, out_channels, do_bias, groups);
  auto conv_fixed = nam::Conv1x1Factory::create(in_channels, out_channels, do_bias, groups);

  // Same weights (grouped layout)
  std::vector<float> weights;
  const int in_per_group = in_channels / groups;
  const int out_per_group = out_channels / groups;
  for (int g = 0; g < groups; g++)
  {
    for (int i = 0; i < out_per_group; i++)
    {
      for (int j = 0; j < in_per_group; j++)
      {
        weights.push_back(static_cast<float>(g * 10 + i * in_per_group + j) * 0.1f);
      }
    }
  }
  for (int i = 0; i < out_channels; i++)
  {
    weights.push_back(static_cast<float>(i) * 0.5f);
  }

  auto it1 = weights.begin();
  auto it2 = weights.begin();
  conv_dynamic.set_weights_(it1);
  conv_fixed->set_weights_(it2);

  conv_dynamic.SetMaxBufferSize(num_frames);
  conv_fixed->SetMaxBufferSize(num_frames);

  // Same input
  Eigen::MatrixXf input(in_channels, num_frames);
  for (int i = 0; i < in_channels; i++)
    for (int j = 0; j < num_frames; j++)
      input(i, j) = static_cast<float>(i * num_frames + j);

  // Process both
  Eigen::MatrixXf output_dynamic = conv_dynamic.process(input, num_frames);
  Eigen::MatrixXf output_fixed = conv_fixed->process(input, num_frames);

  // Compare outputs
  for (int i = 0; i < output_dynamic.rows(); i++)
  {
    for (int j = 0; j < output_dynamic.cols(); j++)
    {
      float diff = std::abs(output_dynamic(i, j) - output_fixed(i, j));
      assert(diff < 1e-5f);
    }
  }
}

// Test process_ is real-time safe (no allocations)
void test_process_realtime_safe()
{
  const int in_channels = 4;
  const int out_channels = 4;
  const bool do_bias = true;
  const int groups = 1;
  const int num_frames = 64;

  auto conv = nam::Conv1x1Factory::create(in_channels, out_channels, do_bias, groups);

  // Initialize weights
  std::vector<float> weights;
  for (int i = 0; i < out_channels * in_channels; i++)
  {
    weights.push_back(static_cast<float>(i) * 0.1f);
  }
  for (int i = 0; i < out_channels; i++)
  {
    weights.push_back(0.0f);
  }

  auto it = weights.begin();
  conv->set_weights_(it);
  conv->SetMaxBufferSize(num_frames);

  // Create input buffer
  Eigen::MatrixXf input(in_channels, num_frames);
  for (int i = 0; i < in_channels; i++)
    for (int j = 0; j < num_frames; j++)
      input(i, j) = static_cast<float>(i + j);

  // Run allocation test
  allocation_tracking::run_allocation_test_no_allocations(
    nullptr,
    [&]() {
      conv->process_(input, num_frames);
    },
    nullptr, "test_conv1x1_fixed_process_realtime_safe");
}

// Test process_ with groups is real-time safe
void test_process_grouped_realtime_safe()
{
  const int in_channels = 8;
  const int out_channels = 8;
  const bool do_bias = true;
  const int groups = 4;
  const int num_frames = 64;

  auto conv = nam::Conv1x1Factory::create(in_channels, out_channels, do_bias, groups);

  // Initialize weights (identity-like for each group)
  std::vector<float> weights;
  const int in_per_group = in_channels / groups;
  const int out_per_group = out_channels / groups;
  for (int g = 0; g < groups; g++)
  {
    for (int i = 0; i < out_per_group; i++)
    {
      for (int j = 0; j < in_per_group; j++)
      {
        weights.push_back(i == j ? 1.0f : 0.0f);
      }
    }
  }
  for (int i = 0; i < out_channels; i++)
  {
    weights.push_back(0.0f);
  }

  auto it = weights.begin();
  conv->set_weights_(it);
  conv->SetMaxBufferSize(num_frames);

  // Create input buffer
  Eigen::MatrixXf input(in_channels, num_frames);
  for (int i = 0; i < in_channels; i++)
    for (int j = 0; j < num_frames; j++)
      input(i, j) = static_cast<float>(i + j);

  // Run allocation test
  allocation_tracking::run_allocation_test_no_allocations(
    nullptr,
    [&]() {
      conv->process_(input, num_frames);
    },
    nullptr, "test_conv1x1_fixed_process_grouped_realtime_safe");
}

// Test SetMaxBufferSize
void test_set_max_buffer_size()
{
  auto conv = nam::Conv1x1Factory::create(4, 4, false, 1);
  conv->SetMaxBufferSize(128);
  auto& output = conv->GetOutput();
  assert(output.rows() == 4);
  assert(output.cols() == 128);
}

// Test multiple calls to process
void test_process_multiple_calls()
{
  const int channels = 4;
  const int num_frames = 2;

  auto conv = nam::Conv1x1Factory::create(channels, channels, false, 1);

  // Identity weights
  std::vector<float> weights;
  for (int i = 0; i < channels; i++)
  {
    for (int j = 0; j < channels; j++)
    {
      weights.push_back(i == j ? 1.0f : 0.0f);
    }
  }

  auto it = weights.begin();
  conv->set_weights_(it);
  conv->SetMaxBufferSize(num_frames);

  Eigen::MatrixXf input1(channels, num_frames);
  input1.setConstant(1.0f);

  auto output1 = conv->process(input1, num_frames);
  for (int i = 0; i < channels; i++)
    for (int j = 0; j < num_frames; j++)
      assert(std::abs(output1(i, j) - 1.0f) < 0.01f);

  Eigen::MatrixXf input2(channels, num_frames);
  input2.setConstant(2.0f);

  auto output2 = conv->process(input2, num_frames);
  for (int i = 0; i < channels; i++)
    for (int j = 0; j < num_frames; j++)
      assert(std::abs(output2(i, j) - 2.0f) < 0.01f);
}

// Test with bias disabled
void test_no_bias()
{
  const int channels = 4;
  const int num_frames = 2;

  auto conv = nam::Conv1x1Factory::create(channels, channels, false, 1);

  // Identity weights (no bias)
  std::vector<float> weights;
  for (int i = 0; i < channels; i++)
  {
    for (int j = 0; j < channels; j++)
    {
      weights.push_back(i == j ? 1.0f : 0.0f);
    }
  }

  auto it = weights.begin();
  conv->set_weights_(it);
  conv->SetMaxBufferSize(num_frames);

  Eigen::MatrixXf input(channels, num_frames);
  input.setConstant(5.0f);

  auto output = conv->process(input, num_frames);
  for (int i = 0; i < channels; i++)
    for (int j = 0; j < num_frames; j++)
      assert(std::abs(output(i, j) - 5.0f) < 0.01f);
}

} // namespace test_conv1x1_fixed
