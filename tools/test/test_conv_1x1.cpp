// Tests for Conv1x1

#include <Eigen/Dense>
#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "NAM/dsp.h"
#include "allocation_tracking.h"

namespace test_conv_1x1
{
// Test basic construction
void test_construct()
{
  const int in_channels = 2;
  const int out_channels = 3;
  const bool do_bias = false;
  nam::Conv1x1 conv(in_channels, out_channels, do_bias);
  assert(conv.get_in_channels() == in_channels);
  assert(conv.get_out_channels() == out_channels);
}

// Test construction with groups (default should be 1)
void test_construct_with_groups()
{
  const int in_channels = 4;
  const int out_channels = 6;
  const bool do_bias = false;
  const int groups = 2;
  nam::Conv1x1 conv(in_channels, out_channels, do_bias, groups);
  assert(conv.get_in_channels() == in_channels);
  assert(conv.get_out_channels() == out_channels);
}

// Test construction validation - in_channels not divisible by groups
void test_construct_validation_in_channels()
{
  const int in_channels = 5;
  const int out_channels = 6;
  const bool do_bias = false;
  const int groups = 2; // 5 not divisible by 2
  bool threw = false;
  try
  {
    nam::Conv1x1 conv(in_channels, out_channels, do_bias, groups);
  }
  catch (const std::runtime_error&)
  {
    threw = true;
  }
  assert(threw);
}

// Test construction validation - out_channels not divisible by groups
void test_construct_validation_out_channels()
{
  const int in_channels = 4;
  const int out_channels = 5;
  const bool do_bias = false;
  const int groups = 2; // 5 not divisible by 2
  bool threw = false;
  try
  {
    nam::Conv1x1 conv(in_channels, out_channels, do_bias, groups);
  }
  catch (const std::runtime_error&)
  {
    threw = true;
  }
  assert(threw);
}

// Test basic process without groups
void test_process_basic()
{
  const int in_channels = 2;
  const int out_channels = 3;
  const bool do_bias = false;
  nam::Conv1x1 conv(in_channels, out_channels, do_bias);
  const int num_frames = 2;

  // Set weights: 3x2 matrix
  // [1.0, 2.0]
  // [3.0, 4.0]
  // [5.0, 6.0]
  std::vector<float> weights{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  auto it = weights.begin();
  conv.set_weights_(it);

  conv.SetMaxBufferSize(64);

  Eigen::MatrixXf input(in_channels, num_frames);
  input(0, 0) = 1.0f;
  input(1, 0) = 2.0f;
  input(0, 1) = 3.0f;
  input(1, 1) = 4.0f;

  Eigen::MatrixXf output = conv.process(input, num_frames);

  assert(output.rows() == out_channels);
  assert(output.cols() == num_frames);
  // Frame 0: [1.0, 2.0] * [1.0; 2.0] = [5.0, 11.0, 17.0]
  assert(std::abs(output(0, 0) - 5.0f) < 0.01f); // 1.0*1.0 + 2.0*2.0
  assert(std::abs(output(1, 0) - 11.0f) < 0.01f); // 3.0*1.0 + 4.0*2.0
  assert(std::abs(output(2, 0) - 17.0f) < 0.01f); // 5.0*1.0 + 6.0*2.0
  // Frame 1: [1.0, 2.0] * [3.0; 4.0] = [11.0, 25.0, 39.0]
  assert(std::abs(output(0, 1) - 11.0f) < 0.01f); // 1.0*3.0 + 2.0*4.0
  assert(std::abs(output(1, 1) - 25.0f) < 0.01f); // 3.0*3.0 + 4.0*4.0
  assert(std::abs(output(2, 1) - 39.0f) < 0.01f); // 5.0*3.0 + 6.0*4.0
}

// Test process with bias
void test_process_with_bias()
{
  const int in_channels = 2;
  const int out_channels = 2;
  const bool do_bias = true;
  nam::Conv1x1 conv(in_channels, out_channels, do_bias);
  const int num_frames = 1;

  // Set weights: 2x2 identity matrix
  // [1.0, 0.0]
  // [0.0, 1.0]
  // Bias: [10.0, 20.0]
  std::vector<float> weights{1.0f, 0.0f, 0.0f, 1.0f, 10.0f, 20.0f};
  auto it = weights.begin();
  conv.set_weights_(it);

  conv.SetMaxBufferSize(64);

  Eigen::MatrixXf input(in_channels, num_frames);
  input(0, 0) = 5.0f;
  input(1, 0) = 7.0f;

  Eigen::MatrixXf output = conv.process(input, num_frames);

  assert(output.rows() == out_channels);
  assert(output.cols() == num_frames);
  // Output should be input + bias (identity weights)
  assert(std::abs(output(0, 0) - 15.0f) < 0.01f); // 5.0 + 10.0
  assert(std::abs(output(1, 0) - 27.0f) < 0.01f); // 7.0 + 20.0
}

// Test process_ method (stores to internal buffer)
void test_process_underscore()
{
  const int in_channels = 2;
  const int out_channels = 2;
  const bool do_bias = false;
  nam::Conv1x1 conv(in_channels, out_channels, do_bias);
  const int num_frames = 1;

  // Set weights: 2x2 identity matrix
  std::vector<float> weights{1.0f, 0.0f, 0.0f, 1.0f};
  auto it = weights.begin();
  conv.set_weights_(it);

  conv.SetMaxBufferSize(64);

  Eigen::MatrixXf input(in_channels, num_frames);
  input(0, 0) = 3.0f;
  input(1, 0) = 4.0f;

  conv.process_(input, num_frames);
  auto output = conv.GetOutput().leftCols(num_frames);

  assert(output.rows() == out_channels);
  assert(output.cols() == num_frames);
  assert(std::abs(output(0, 0) - 3.0f) < 0.01f);
  assert(std::abs(output(1, 0) - 4.0f) < 0.01f);
}

// Test basic grouped convolution with 2 groups
void test_process_grouped_basic()
{
  const int in_channels = 4;
  const int out_channels = 4;
  const bool do_bias = false;
  const int groups = 2;
  nam::Conv1x1 conv(in_channels, out_channels, do_bias, groups);
  const int num_frames = 2;

  // For grouped convolution with 2 groups:
  // Group 0: processes in_channels[0:1] -> out_channels[0:1]
  // Group 1: processes in_channels[2:3] -> out_channels[2:3]
  // Each group has out_per_group=2, in_per_group=2
  // Weight layout: [group0, group1]
  // Group 0: identity matrix (2x2)
  // Group 1: scale by 2.0 (2x2)
  std::vector<float> weights;
  // Group 0: identity
  weights.push_back(1.0f); // out[0], in[0]
  weights.push_back(0.0f); // out[0], in[1]
  weights.push_back(0.0f); // out[1], in[0]
  weights.push_back(1.0f); // out[1], in[1]
  // Group 1: scale by 2.0
  weights.push_back(2.0f); // out[2], in[2]
  weights.push_back(0.0f); // out[2], in[3]
  weights.push_back(0.0f); // out[3], in[2]
  weights.push_back(2.0f); // out[3], in[3]

  auto it = weights.begin();
  conv.set_weights_(it);

  conv.SetMaxBufferSize(64);

  Eigen::MatrixXf input(in_channels, num_frames);
  input(0, 0) = 1.0f; // Group 0, channel 0
  input(1, 0) = 2.0f; // Group 0, channel 1
  input(2, 0) = 3.0f; // Group 1, channel 0
  input(3, 0) = 4.0f; // Group 1, channel 1
  input(0, 1) = 5.0f;
  input(1, 1) = 6.0f;
  input(2, 1) = 7.0f;
  input(3, 1) = 8.0f;

  Eigen::MatrixXf output = conv.process(input, num_frames);

  assert(output.rows() == out_channels);
  assert(output.cols() == num_frames);
  // Group 0: identity transformation
  assert(std::abs(output(0, 0) - 1.0f) < 0.01f); // out[0] = in[0]
  assert(std::abs(output(1, 0) - 2.0f) < 0.01f); // out[1] = in[1]
  // Group 1: double transformation
  assert(std::abs(output(2, 0) - 6.0f) < 0.01f); // out[2] = 2.0 * in[2]
  assert(std::abs(output(3, 0) - 8.0f) < 0.01f); // out[3] = 2.0 * in[3]
  // Frame 1
  assert(std::abs(output(0, 1) - 5.0f) < 0.01f);
  assert(std::abs(output(1, 1) - 6.0f) < 0.01f);
  assert(std::abs(output(2, 1) - 14.0f) < 0.01f); // 2.0 * 7.0
  assert(std::abs(output(3, 1) - 16.0f) < 0.01f); // 2.0 * 8.0
}

// Test grouped convolution with bias
void test_process_grouped_with_bias()
{
  const int in_channels = 4;
  const int out_channels = 4;
  const bool do_bias = true;
  const int groups = 2;
  nam::Conv1x1 conv(in_channels, out_channels, do_bias, groups);
  const int num_frames = 1;

  std::vector<float> weights;
  // Group 0 weights (2x2 identity)
  weights.push_back(1.0f);
  weights.push_back(0.0f);
  weights.push_back(0.0f);
  weights.push_back(1.0f);
  // Group 1 weights (2x2 identity)
  weights.push_back(1.0f);
  weights.push_back(0.0f);
  weights.push_back(0.0f);
  weights.push_back(1.0f);
  // Bias: [1.0, 2.0, 3.0, 4.0]
  weights.push_back(1.0f);
  weights.push_back(2.0f);
  weights.push_back(3.0f);
  weights.push_back(4.0f);

  auto it = weights.begin();
  conv.set_weights_(it);

  conv.SetMaxBufferSize(64);

  Eigen::MatrixXf input(in_channels, num_frames);
  input(0, 0) = 10.0f;
  input(1, 0) = 20.0f;
  input(2, 0) = 30.0f;
  input(3, 0) = 40.0f;

  Eigen::MatrixXf output = conv.process(input, num_frames);

  assert(output.rows() == out_channels);
  assert(output.cols() == num_frames);
  // Output should be input + bias (identity weights)
  assert(std::abs(output(0, 0) - 11.0f) < 0.01f); // 10.0 + 1.0
  assert(std::abs(output(1, 0) - 22.0f) < 0.01f); // 20.0 + 2.0
  assert(std::abs(output(2, 0) - 33.0f) < 0.01f); // 30.0 + 3.0
  assert(std::abs(output(3, 0) - 44.0f) < 0.01f); // 40.0 + 4.0
}

// Test grouped convolution with 4 groups
void test_process_grouped_multiple_groups()
{
  const int in_channels = 8;
  const int out_channels = 8;
  const bool do_bias = false;
  const int groups = 4;
  nam::Conv1x1 conv(in_channels, out_channels, do_bias, groups);
  const int num_frames = 1;

  // Each group processes 2 input channels -> 2 output channels
  std::vector<float> weights;
  // Group 0: scale by 1.0
  weights.push_back(1.0f);
  weights.push_back(0.0f);
  weights.push_back(0.0f);
  weights.push_back(1.0f);
  // Group 1: scale by 2.0
  weights.push_back(2.0f);
  weights.push_back(0.0f);
  weights.push_back(0.0f);
  weights.push_back(2.0f);
  // Group 2: scale by 3.0
  weights.push_back(3.0f);
  weights.push_back(0.0f);
  weights.push_back(0.0f);
  weights.push_back(3.0f);
  // Group 3: scale by 4.0
  weights.push_back(4.0f);
  weights.push_back(0.0f);
  weights.push_back(0.0f);
  weights.push_back(4.0f);

  auto it = weights.begin();
  conv.set_weights_(it);

  conv.SetMaxBufferSize(64);

  Eigen::MatrixXf input(in_channels, num_frames);
  for (int i = 0; i < in_channels; i++)
  {
    input(i, 0) = static_cast<float>(i + 1);
  }

  Eigen::MatrixXf output = conv.process(input, num_frames);

  assert(output.rows() == out_channels);
  assert(output.cols() == num_frames);
  // Group 0: channels 0-1 scaled by 1.0
  assert(std::abs(output(0, 0) - 1.0f) < 0.01f);
  assert(std::abs(output(1, 0) - 2.0f) < 0.01f);
  // Group 1: channels 2-3 scaled by 2.0
  assert(std::abs(output(2, 0) - 6.0f) < 0.01f); // 3.0 * 2.0
  assert(std::abs(output(3, 0) - 8.0f) < 0.01f); // 4.0 * 2.0
  // Group 2: channels 4-5 scaled by 3.0
  assert(std::abs(output(4, 0) - 15.0f) < 0.01f); // 5.0 * 3.0
  assert(std::abs(output(5, 0) - 18.0f) < 0.01f); // 6.0 * 3.0
  // Group 3: channels 6-7 scaled by 4.0
  assert(std::abs(output(6, 0) - 28.0f) < 0.01f); // 7.0 * 4.0
  assert(std::abs(output(7, 0) - 32.0f) < 0.01f); // 8.0 * 4.0
}

// Test that groups don't mix channels (channel isolation)
void test_process_grouped_channel_isolation()
{
  const int in_channels = 6;
  const int out_channels = 6;
  const bool do_bias = false;
  const int groups = 3;
  nam::Conv1x1 conv(in_channels, out_channels, do_bias, groups);
  const int num_frames = 1;

  // 3 groups, each processes 2 channels
  // Group 0: channels 0-1, set to zero (zero matrix)
  // Group 1: channels 2-3, identity
  // Group 2: channels 4-5, identity
  std::vector<float> weights;
  // Group 0: zero matrix
  weights.push_back(0.0f);
  weights.push_back(0.0f);
  weights.push_back(0.0f);
  weights.push_back(0.0f);
  // Group 1: identity
  weights.push_back(1.0f);
  weights.push_back(0.0f);
  weights.push_back(0.0f);
  weights.push_back(1.0f);
  // Group 2: identity
  weights.push_back(1.0f);
  weights.push_back(0.0f);
  weights.push_back(0.0f);
  weights.push_back(1.0f);

  auto it = weights.begin();
  conv.set_weights_(it);

  conv.SetMaxBufferSize(64);

  Eigen::MatrixXf input(in_channels, num_frames);
  input(0, 0) = 10.0f; // Should be zeroed by group 0
  input(1, 0) = 20.0f; // Should be zeroed by group 0
  input(2, 0) = 30.0f; // Should pass through group 1
  input(3, 0) = 40.0f; // Should pass through group 1
  input(4, 0) = 50.0f; // Should pass through group 2
  input(5, 0) = 60.0f; // Should pass through group 2

  Eigen::MatrixXf output = conv.process(input, num_frames);

  assert(output.rows() == out_channels);
  assert(output.cols() == num_frames);
  // Group 0: should be zero
  assert(std::abs(output(0, 0)) < 0.01f);
  assert(std::abs(output(1, 0)) < 0.01f);
  // Group 1: should pass through
  assert(std::abs(output(2, 0) - 30.0f) < 0.01f);
  assert(std::abs(output(3, 0) - 40.0f) < 0.01f);
  // Group 2: should pass through
  assert(std::abs(output(4, 0) - 50.0f) < 0.01f);
  assert(std::abs(output(5, 0) - 60.0f) < 0.01f);
}

// Test process_ with groups
void test_process_underscore_grouped()
{
  const int in_channels = 4;
  const int out_channels = 4;
  const bool do_bias = false;
  const int groups = 2;
  nam::Conv1x1 conv(in_channels, out_channels, do_bias, groups);
  const int num_frames = 1;

  // Group 0: identity, Group 1: scale by 2.0
  std::vector<float> weights;
  // Group 0: identity
  weights.push_back(1.0f);
  weights.push_back(0.0f);
  weights.push_back(0.0f);
  weights.push_back(1.0f);
  // Group 1: scale by 2.0
  weights.push_back(2.0f);
  weights.push_back(0.0f);
  weights.push_back(0.0f);
  weights.push_back(2.0f);

  auto it = weights.begin();
  conv.set_weights_(it);

  conv.SetMaxBufferSize(64);

  Eigen::MatrixXf input(in_channels, num_frames);
  input(0, 0) = 1.0f;
  input(1, 0) = 2.0f;
  input(2, 0) = 3.0f;
  input(3, 0) = 4.0f;

  conv.process_(input, num_frames);
  auto output = conv.GetOutput().leftCols(num_frames);

  assert(output.rows() == out_channels);
  assert(output.cols() == num_frames);
  assert(std::abs(output(0, 0) - 1.0f) < 0.01f);
  assert(std::abs(output(1, 0) - 2.0f) < 0.01f);
  assert(std::abs(output(2, 0) - 6.0f) < 0.01f); // 2.0 * 3.0
  assert(std::abs(output(3, 0) - 8.0f) < 0.01f); // 2.0 * 4.0
}

// Test SetMaxBufferSize
void test_set_max_buffer_size()
{
  const int in_channels = 2;
  const int out_channels = 3;
  const bool do_bias = false;
  nam::Conv1x1 conv(in_channels, out_channels, do_bias);
  const int maxBufferSize = 128;

  conv.SetMaxBufferSize(maxBufferSize);
  auto output = conv.GetOutput();
  assert(output.rows() == out_channels);
  assert(output.cols() == maxBufferSize);
}

// Test multiple calls to process
void test_process_multiple_calls()
{
  const int in_channels = 2;
  const int out_channels = 2;
  const bool do_bias = false;
  nam::Conv1x1 conv(in_channels, out_channels, do_bias);
  // Identity matrix
  std::vector<float> weights{1.0f, 0.0f, 0.0f, 1.0f};
  auto it = weights.begin();
  conv.set_weights_(it);
  conv.SetMaxBufferSize(64);

  Eigen::MatrixXf input1(in_channels, 1);
  input1(0, 0) = 1.0f;
  input1(1, 0) = 2.0f;

  Eigen::MatrixXf output1 = conv.process(input1, 1);
  assert(std::abs(output1(0, 0) - 1.0f) < 0.01f);
  assert(std::abs(output1(1, 0) - 2.0f) < 0.01f);

  Eigen::MatrixXf input2(in_channels, 1);
  input2(0, 0) = 3.0f;
  input2(1, 0) = 4.0f;

  Eigen::MatrixXf output2 = conv.process(input2, 1);
  assert(std::abs(output2(0, 0) - 3.0f) < 0.01f);
  assert(std::abs(output2(1, 0) - 4.0f) < 0.01f);
}

// Test that grouped convolution process_() is real-time safe (no allocations)
void test_process_grouped_realtime_safe()
{
  const int in_channels = 8;
  const int out_channels = 8;
  const bool do_bias = true;
  const int groups = 4;
  const int num_frames = 64;

  nam::Conv1x1 conv(in_channels, out_channels, do_bias, groups);

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
  // Add bias
  for (int i = 0; i < out_channels; i++)
  {
    weights.push_back(0.0f);
  }

  auto it = weights.begin();
  conv.set_weights_(it);
  conv.SetMaxBufferSize(num_frames);

  // Create input buffer
  Eigen::MatrixXf input(in_channels, num_frames);
  for (int i = 0; i < in_channels; i++)
    for (int j = 0; j < num_frames; j++)
      input(i, j) = static_cast<float>(i + j);

  // Run allocation test
  allocation_tracking::run_allocation_test_no_allocations(
    nullptr,
    [&]() {
      conv.process_(input, num_frames);
    },
    nullptr, "test_process_grouped_realtime_safe");
}

// Test that non-grouped convolution process_() is also real-time safe
void test_process_realtime_safe()
{
  const int in_channels = 16;
  const int out_channels = 16;
  const bool do_bias = true;
  const int num_frames = 64;

  nam::Conv1x1 conv(in_channels, out_channels, do_bias);

  // Initialize weights (identity)
  std::vector<float> weights;
  for (int i = 0; i < out_channels; i++)
  {
    for (int j = 0; j < in_channels; j++)
    {
      weights.push_back(i == j ? 1.0f : 0.0f);
    }
  }
  // Add bias
  for (int i = 0; i < out_channels; i++)
  {
    weights.push_back(0.0f);
  }

  auto it = weights.begin();
  conv.set_weights_(it);
  conv.SetMaxBufferSize(num_frames);

  // Create input buffer
  Eigen::MatrixXf input(in_channels, num_frames);
  for (int i = 0; i < in_channels; i++)
    for (int j = 0; j < num_frames; j++)
      input(i, j) = static_cast<float>(i + j);

  // Run allocation test
  allocation_tracking::run_allocation_test_no_allocations(
    nullptr,
    [&]() {
      conv.process_(input, num_frames);
    },
    nullptr, "test_process_realtime_safe");
}
} // namespace test_conv_1x1
