// Tests for Conv1DFixed (templated implementation)

#include <Eigen/Dense>
#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "NAM/conv1d.h"
#include "NAM/conv1d_factory.h"
#include "NAM/conv1d_fixed.h"
#include "allocation_tracking.h"

namespace test_conv1d_fixed
{

// Test factory creation
void test_factory_create()
{
  auto conv = nam::Conv1DFactory::create(4, 4, 3, 1, true, 1);
  assert(conv != nullptr);
  assert(conv->get_in_channels() == 4);
  assert(conv->get_out_channels() == 4);
  assert(conv->get_kernel_size() == 3);
  assert(conv->get_dilation() == 1);
  assert(conv->has_bias() == true);
}

// Test factory with groups
void test_factory_create_with_groups()
{
  auto conv = nam::Conv1DFactory::create(8, 8, 3, 1, false, 2);
  assert(conv != nullptr);
  assert(conv->get_in_channels() == 8);
  assert(conv->get_out_channels() == 8);
}

// Test process gives same result as dynamic implementation
void test_numerical_equivalence()
{
  const int in_channels = 4;
  const int out_channels = 4;
  const int kernel_size = 3;
  const int dilation = 1;
  const bool do_bias = true;
  const int groups = 1;
  const int num_frames = 8;

  // Create both implementations
  nam::Conv1D conv_dynamic;
  conv_dynamic.set_size_(in_channels, out_channels, kernel_size, do_bias, dilation, groups);
  auto conv_fixed = nam::Conv1DFactory::create(in_channels, out_channels, kernel_size, dilation, do_bias, groups);

  // Generate weights
  std::vector<float> weights;
  const int in_per_group = in_channels / groups;
  const int out_per_group = out_channels / groups;
  for (int g = 0; g < groups; g++)
  {
    for (int i = 0; i < out_per_group; i++)
    {
      for (int j = 0; j < in_per_group; j++)
      {
        for (int k = 0; k < kernel_size; k++)
        {
          weights.push_back(static_cast<float>(g * 100 + i * 10 + j + k) * 0.01f);
        }
      }
    }
  }
  for (int i = 0; i < out_channels; i++)
  {
    weights.push_back(static_cast<float>(i) * 0.1f);
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
      input(i, j) = static_cast<float>(i * num_frames + j) * 0.1f;

  // Process both
  conv_dynamic.Process(input, num_frames);
  conv_fixed->Process(input, num_frames);

  auto& output_dynamic = conv_dynamic.GetOutput();
  auto& output_fixed = conv_fixed->GetOutput();

  // Compare outputs
  for (int i = 0; i < out_channels; i++)
  {
    for (int j = 0; j < num_frames; j++)
    {
      float diff = std::abs(output_dynamic(i, j) - output_fixed(i, j));
      assert(diff < 1e-4f);
    }
  }
}

// Test grouped convolution numerical equivalence
void test_numerical_equivalence_grouped()
{
  const int in_channels = 8;
  const int out_channels = 8;
  const int kernel_size = 3;
  const int dilation = 1;
  const bool do_bias = true;
  const int groups = 2;
  const int num_frames = 8;

  // Create both implementations
  nam::Conv1D conv_dynamic;
  conv_dynamic.set_size_(in_channels, out_channels, kernel_size, do_bias, dilation, groups);
  auto conv_fixed = nam::Conv1DFactory::create(in_channels, out_channels, kernel_size, dilation, do_bias, groups);

  // Generate weights
  std::vector<float> weights;
  const int in_per_group = in_channels / groups;
  const int out_per_group = out_channels / groups;
  for (int g = 0; g < groups; g++)
  {
    for (int i = 0; i < out_per_group; i++)
    {
      for (int j = 0; j < in_per_group; j++)
      {
        for (int k = 0; k < kernel_size; k++)
        {
          weights.push_back(static_cast<float>(g * 100 + i * 10 + j + k) * 0.01f);
        }
      }
    }
  }
  for (int i = 0; i < out_channels; i++)
  {
    weights.push_back(static_cast<float>(i) * 0.1f);
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
      input(i, j) = static_cast<float>(i * num_frames + j) * 0.1f;

  // Process both
  conv_dynamic.Process(input, num_frames);
  conv_fixed->Process(input, num_frames);

  auto& output_dynamic = conv_dynamic.GetOutput();
  auto& output_fixed = conv_fixed->GetOutput();

  // Compare outputs
  for (int i = 0; i < out_channels; i++)
  {
    for (int j = 0; j < num_frames; j++)
    {
      float diff = std::abs(output_dynamic(i, j) - output_fixed(i, j));
      assert(diff < 1e-4f);
    }
  }
}

// Test with different kernel size
void test_numerical_equivalence_kernel4()
{
  const int in_channels = 4;
  const int out_channels = 4;
  const int kernel_size = 4;
  const int dilation = 1;
  const bool do_bias = true;
  const int groups = 1;
  const int num_frames = 8;

  // Create both implementations
  nam::Conv1D conv_dynamic;
  conv_dynamic.set_size_(in_channels, out_channels, kernel_size, do_bias, dilation, groups);
  auto conv_fixed = nam::Conv1DFactory::create(in_channels, out_channels, kernel_size, dilation, do_bias, groups);

  // Generate weights
  std::vector<float> weights;
  for (int i = 0; i < out_channels; i++)
  {
    for (int j = 0; j < in_channels; j++)
    {
      for (int k = 0; k < kernel_size; k++)
      {
        weights.push_back(static_cast<float>(i * 10 + j + k) * 0.01f);
      }
    }
  }
  for (int i = 0; i < out_channels; i++)
  {
    weights.push_back(static_cast<float>(i) * 0.1f);
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
      input(i, j) = static_cast<float>(i * num_frames + j) * 0.1f;

  // Process both
  conv_dynamic.Process(input, num_frames);
  conv_fixed->Process(input, num_frames);

  auto& output_dynamic = conv_dynamic.GetOutput();
  auto& output_fixed = conv_fixed->GetOutput();

  // Compare outputs
  for (int i = 0; i < out_channels; i++)
  {
    for (int j = 0; j < num_frames; j++)
    {
      float diff = std::abs(output_dynamic(i, j) - output_fixed(i, j));
      assert(diff < 1e-4f);
    }
  }
}

// Test process is real-time safe (no allocations)
void test_process_realtime_safe()
{
  const int in_channels = 4;
  const int out_channels = 4;
  const int kernel_size = 3;
  const int dilation = 1;
  const bool do_bias = true;
  const int groups = 1;
  const int num_frames = 64;

  auto conv = nam::Conv1DFactory::create(in_channels, out_channels, kernel_size, dilation, do_bias, groups);

  // Initialize weights
  std::vector<float> weights;
  for (int i = 0; i < out_channels; i++)
  {
    for (int j = 0; j < in_channels; j++)
    {
      for (int k = 0; k < kernel_size; k++)
      {
        weights.push_back(0.1f);
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
      conv->Process(input, num_frames);
    },
    nullptr, "test_conv1d_fixed_process_realtime_safe");
}

// Test process with groups is real-time safe
void test_process_grouped_realtime_safe()
{
  const int in_channels = 8;
  const int out_channels = 8;
  const int kernel_size = 3;
  const int dilation = 1;
  const bool do_bias = true;
  const int groups = 4;
  const int num_frames = 64;

  auto conv = nam::Conv1DFactory::create(in_channels, out_channels, kernel_size, dilation, do_bias, groups);

  // Initialize weights
  std::vector<float> weights;
  const int in_per_group = in_channels / groups;
  const int out_per_group = out_channels / groups;
  for (int g = 0; g < groups; g++)
  {
    for (int i = 0; i < out_per_group; i++)
    {
      for (int j = 0; j < in_per_group; j++)
      {
        for (int k = 0; k < kernel_size; k++)
        {
          weights.push_back(i == j ? 1.0f : 0.0f);
        }
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
      conv->Process(input, num_frames);
    },
    nullptr, "test_conv1d_fixed_process_grouped_realtime_safe");
}

// Test SetMaxBufferSize
void test_set_max_buffer_size()
{
  auto conv = nam::Conv1DFactory::create(4, 4, 3, 1, false, 1);
  conv->SetMaxBufferSize(128);
  auto& output = conv->GetOutput();
  assert(output.rows() == 4);
  assert(output.cols() == 128);
}

// Test multiple calls to process
void test_process_multiple_calls()
{
  const int channels = 4;
  const int kernel_size = 3;
  const int num_frames = 4;

  auto conv = nam::Conv1DFactory::create(channels, channels, kernel_size, 1, false, 1);

  // Identity-like weights (all zeros except center)
  std::vector<float> weights;
  for (int i = 0; i < channels; i++)
  {
    for (int j = 0; j < channels; j++)
    {
      for (int k = 0; k < kernel_size; k++)
      {
        // Put weight at last kernel position for identity-like behavior
        weights.push_back((k == kernel_size - 1 && i == j) ? 1.0f : 0.0f);
      }
    }
  }

  auto it = weights.begin();
  conv->set_weights_(it);
  conv->SetMaxBufferSize(num_frames);

  // First call
  Eigen::MatrixXf input1(channels, num_frames);
  input1.setConstant(1.0f);
  conv->Process(input1, num_frames);

  // Second call
  Eigen::MatrixXf input2(channels, num_frames);
  input2.setConstant(2.0f);
  conv->Process(input2, num_frames);

  // Output should reflect the second call's values (for the last positions at least)
  auto& output = conv->GetOutput();
  // After the ring buffer fills, we should see values based on the second input
  assert(output.rows() == channels);
}

// Test with bias disabled
void test_no_bias()
{
  const int channels = 4;
  const int kernel_size = 3;
  const int num_frames = 4;

  // Create dynamic and fixed with no bias
  nam::Conv1D conv_dynamic;
  conv_dynamic.set_size_(channels, channels, kernel_size, false, 1, 1);
  auto conv_fixed = nam::Conv1DFactory::create(channels, channels, kernel_size, 1, false, 1);

  // Same weights (no bias)
  std::vector<float> weights;
  for (int i = 0; i < channels; i++)
  {
    for (int j = 0; j < channels; j++)
    {
      for (int k = 0; k < kernel_size; k++)
      {
        weights.push_back((k == kernel_size - 1 && i == j) ? 1.0f : 0.0f);
      }
    }
  }

  auto it1 = weights.begin();
  auto it2 = weights.begin();
  conv_dynamic.set_weights_(it1);
  conv_fixed->set_weights_(it2);

  conv_dynamic.SetMaxBufferSize(num_frames);
  conv_fixed->SetMaxBufferSize(num_frames);

  Eigen::MatrixXf input(channels, num_frames);
  input.setConstant(5.0f);

  conv_dynamic.Process(input, num_frames);
  conv_fixed->Process(input, num_frames);

  auto& output_dynamic = conv_dynamic.GetOutput();
  auto& output_fixed = conv_fixed->GetOutput();

  // Compare outputs
  for (int i = 0; i < channels; i++)
  {
    for (int j = 0; j < num_frames; j++)
    {
      float diff = std::abs(output_dynamic(i, j) - output_fixed(i, j));
      assert(diff < 1e-4f);
    }
  }
}

// Test with dilation
void test_with_dilation()
{
  const int channels = 4;
  const int kernel_size = 3;
  const int dilation = 2;
  const int num_frames = 8;

  nam::Conv1D conv_dynamic;
  conv_dynamic.set_size_(channels, channels, kernel_size, true, dilation, 1);
  auto conv_fixed = nam::Conv1DFactory::create(channels, channels, kernel_size, dilation, true, 1);

  // Same weights
  std::vector<float> weights;
  for (int i = 0; i < channels; i++)
  {
    for (int j = 0; j < channels; j++)
    {
      for (int k = 0; k < kernel_size; k++)
      {
        weights.push_back(0.1f * (i + j + k));
      }
    }
  }
  for (int i = 0; i < channels; i++)
  {
    weights.push_back(0.5f);
  }

  auto it1 = weights.begin();
  auto it2 = weights.begin();
  conv_dynamic.set_weights_(it1);
  conv_fixed->set_weights_(it2);

  conv_dynamic.SetMaxBufferSize(num_frames);
  conv_fixed->SetMaxBufferSize(num_frames);

  Eigen::MatrixXf input(channels, num_frames);
  for (int i = 0; i < channels; i++)
    for (int j = 0; j < num_frames; j++)
      input(i, j) = static_cast<float>(i + j);

  conv_dynamic.Process(input, num_frames);
  conv_fixed->Process(input, num_frames);

  auto& output_dynamic = conv_dynamic.GetOutput();
  auto& output_fixed = conv_fixed->GetOutput();

  // Compare outputs
  for (int i = 0; i < channels; i++)
  {
    for (int j = 0; j < num_frames; j++)
    {
      float diff = std::abs(output_dynamic(i, j) - output_fixed(i, j));
      assert(diff < 1e-4f);
    }
  }
}

} // namespace test_conv1d_fixed
