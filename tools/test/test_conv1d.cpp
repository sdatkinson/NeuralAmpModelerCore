// Tests for Conv1D

#include <Eigen/Dense>
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

#include "NAM/conv1d.h"

namespace test_conv1d
{
// Test basic construction
void test_construct()
{
  nam::Conv1D conv;
  assert(conv.get_dilation() == 1);
  assert(conv.get_in_channels() == 0);
  assert(conv.get_out_channels() == 0);
  assert(conv.get_kernel_size() == 0);
}

// Test construction with provided shape
void test_construct_with_shape()
{
  nam::Conv1D conv(2, 3, 5, true, 7);
  assert(conv.get_dilation() == 7);
  assert(conv.get_in_channels() == 2);
  assert(conv.get_out_channels() == 3);
  assert(conv.get_kernel_size() == 5);
  assert(conv.has_bias());
}

// Test set_size_ and getters
void test_set_size()
{
  nam::Conv1D conv;
  const int in_channels = 2;
  const int out_channels = 4;
  const int kernel_size = 3;
  const bool do_bias = true;
  const int dilation = 2;

  conv.set_size_(in_channels, out_channels, kernel_size, do_bias, dilation);

  assert(conv.get_in_channels() == in_channels);
  assert(conv.get_out_channels() == out_channels);
  assert(conv.get_kernel_size() == kernel_size);
  assert(conv.get_dilation() == dilation);
  assert(conv.has_bias() == do_bias);
}

// Test Reset() initializes buffers
void test_reset()
{
  nam::Conv1D conv;
  const int in_channels = 2;
  const int out_channels = 4;
  const int kernel_size = 3;
  const int maxBufferSize = 64;
  const double sampleRate = 48000.0;

  conv.set_size_(in_channels, out_channels, kernel_size, false, 1);
  conv.SetMaxBufferSize(maxBufferSize);

  // After Reset, GetOutput should work
  // (Even thoguh GetOutput() doesn't make sense to call before Process())
  auto output = conv.GetOutput().leftCols(maxBufferSize);
  assert(output.rows() == out_channels);
  assert(output.cols() == maxBufferSize);
}

// Test basic Process() with simple convolution
void test_process_basic()
{
  nam::Conv1D conv;
  const int in_channels = 1;
  const int out_channels = 1;
  const int kernel_size = 2;
  const bool do_bias = false;
  const int dilation = 1;
  const int num_frames = 4;

  conv.set_size_(in_channels, out_channels, kernel_size, do_bias, dilation);

  // Set weights: kernel[0] = [[1.0]], kernel[1] = [[2.0]]
  // With offset calculation: k=0 has offset=-1 (looks at t-1), k=1 has offset=0 (looks at t)
  // So: output = weight[0] * input[t-1] + weight[1] * input[t] = 1.0 * input[t-1] + 2.0 * input[t]
  std::vector<float> weights{1.0f, 2.0f};
  auto it = weights.begin();
  conv.set_weights_(it);

  conv.SetMaxBufferSize(64);

  // Create input: [1.0, 2.0, 3.0, 4.0]
  Eigen::MatrixXf input(in_channels, num_frames);
  input(0, 0) = 1.0f;
  input(0, 1) = 2.0f;
  input(0, 2) = 3.0f;
  input(0, 3) = 4.0f;

  // Process
  conv.Process(input, num_frames);

  // Get output
  auto output = conv.GetOutput().leftCols(num_frames);

  // Expected outputs (with zero padding for first frame):
  // output[0] = 1.0 * 0.0 (zero-padding) + 2.0 * 1.0 = 2.0
  // output[1] = 1.0 * 1.0 + 2.0 * 2.0 = 5.0
  // output[2] = 1.0 * 2.0 + 2.0 * 3.0 = 8.0
  // output[3] = 1.0 * 3.0 + 2.0 * 4.0 = 11.0
  assert(output.rows() == out_channels);
  assert(output.cols() == num_frames);
  assert(abs(output(0, 0) - 2.0f) < 0.01f);
  assert(abs(output(0, 1) - 5.0f) < 0.01f);
  assert(abs(output(0, 2) - 8.0f) < 0.01f);
  assert(abs(output(0, 3) - 11.0f) < 0.01f);
}

// Test Process() with bias
void test_process_with_bias()
{
  nam::Conv1D conv;
  const int in_channels = 1;
  const int out_channels = 1;
  const int kernel_size = 2;
  const bool do_bias = true;
  const int dilation = 1;
  const int num_frames = 2;

  conv.set_size_(in_channels, out_channels, kernel_size, do_bias, dilation);

  // Set weights: kernel[0] = [[1.0]], kernel[1] = [[0.0]], bias = [5.0]
  // With offset: k=0 has offset=-1 (looks at t-1), k=1 has offset=0 (looks at t)
  // So: output = weight[0] * input[t-1] + weight[1] * input[t] + bias = 1.0 * input[t-1] + 0.0 * input[t] + 5.0
  std::vector<float> weights{1.0f, 0.0f, 5.0f};
  auto it = weights.begin();
  conv.set_weights_(it);

  conv.SetMaxBufferSize(64);

  Eigen::MatrixXf input(in_channels, num_frames);
  input(0, 0) = 2.0f;
  input(0, 1) = 3.0f;

  conv.Process(input, num_frames);
  auto output = conv.GetOutput().leftCols(num_frames);

  // Should have bias added
  assert(output.rows() == out_channels);
  assert(output.cols() == num_frames);
  // With zero-padding for first frame:
  // First frame: weight[0]*zero + weight[1]*input[0] + bias = 1.0*0.0 + 0.0*2.0 + 5.0 = 5.0
  // Second frame: weight[0]*input[0] + weight[1]*input[1] + bias = 1.0*2.0 + 0.0*3.0 + 5.0 = 7.0
  assert(std::abs(output(0, 0) - 5.0f) < 0.01f); // First frame: zero-padding + bias
  assert(std::abs(output(0, 1) - 7.0f) < 0.01f); // Second frame: input[0] + bias
}

// Test Process() with multiple channels
void test_process_multichannel()
{
  nam::Conv1D conv;
  const int in_channels = 2;
  const int out_channels = 3;
  const int kernel_size = 1;
  const bool do_bias = false;
  const int dilation = 1;
  const int num_frames = 2;

  conv.set_size_(in_channels, out_channels, kernel_size, do_bias, dilation);

  // Set simple identity-like weights for kernel[0]
  // weight[0] should be (3, 2) matrix
  // Let's use: [[1, 0], [0, 1], [1, 1]] which means:
  // out[0] = in[0]
  // out[1] = in[1]
  // out[2] = in[0] + in[1]
  std::vector<float> weights;
  // kernel[0] weights (3x2 matrix, row-major flattened)
  weights.push_back(1.0f); // out[0], in[0]
  weights.push_back(0.0f); // out[0], in[1]
  weights.push_back(0.0f); // out[1], in[0]
  weights.push_back(1.0f); // out[1], in[1]
  weights.push_back(1.0f); // out[2], in[0]
  weights.push_back(1.0f); // out[2], in[1]

  auto it = weights.begin();
  conv.set_weights_(it);

  conv.SetMaxBufferSize(64);

  Eigen::MatrixXf input(in_channels, num_frames);
  input(0, 0) = 1.0f;
  input(1, 0) = 2.0f;
  input(0, 1) = 3.0f;
  input(1, 1) = 4.0f;

  conv.Process(input, num_frames);
  auto output = conv.GetOutput().leftCols(num_frames);

  assert(output.rows() == out_channels);
  assert(output.cols() == num_frames);
  // out[0] = in[0] = 1.0
  // out[1] = in[1] = 2.0
  // out[2] = in[0] + in[1] = 3.0
  assert(std::abs(output(0, 0) - 1.0f) < 0.01f);
  assert(std::abs(output(1, 0) - 2.0f) < 0.01f);
  assert(std::abs(output(2, 0) - 3.0f) < 0.01f);
}

// Test Process() with dilation
void test_process_dilation()
{
  nam::Conv1D conv;
  const int in_channels = 1;
  const int out_channels = 1;
  const int kernel_size = 2;
  const bool do_bias = false;
  const int dilation = 2;
  const int num_frames = 4;

  conv.set_size_(in_channels, out_channels, kernel_size, do_bias, dilation);

  // Set weights: kernel[0] = [[1.0]], kernel[1] = [[2.0]]
  // With dilation=2: k=0 has offset=-2 (looks at t-2), k=1 has offset=0 (looks at t)
  // So: output = weight[0] * input[t-2] + weight[1] * input[t] = 1.0 * input[t-2] + 2.0 * input[t]
  std::vector<float> weights{1.0f, 2.0f};
  auto it = weights.begin();
  conv.set_weights_(it);

  conv.SetMaxBufferSize(64);

  Eigen::MatrixXf input(in_channels, num_frames);
  input(0, 0) = 1.0f;
  input(0, 1) = 2.0f;
  input(0, 2) = 3.0f;
  input(0, 3) = 4.0f;

  conv.Process(input, num_frames);
  auto output = conv.GetOutput().leftCols(num_frames);

  assert(output.rows() == out_channels);
  assert(output.cols() == num_frames);
  // Output should be computed correctly with dilation (with zero-padding)
  // out[0] = 1.0 * 0.0 (zero-padding) + 2.0 * 1.0 = 2.0
  // out[1] = 1.0 * 0.0 (zero-padding) + 2.0 * 2.0 = 4.0
  // out[2] = 1.0 * 1.0 + 2.0 * 3.0 = 1.0 + 6.0 = 7.0
  // out[3] = 1.0 * 2.0 + 2.0 * 4.0 = 2.0 + 8.0 = 10.0
  assert(abs(output(0, 0) - 2.0f) < 0.01f);
  assert(abs(output(0, 1) - 4.0f) < 0.01f);
  assert(abs(output(0, 2) - 7.0f) < 0.01f);
  assert(abs(output(0, 3) - 10.0f) < 0.01f);
}

// Test multiple Process() calls (ring buffer functionality)
void test_process_multiple_calls()
{
  nam::Conv1D conv;
  const int in_channels = 1;
  const int out_channels = 1;
  const int kernel_size = 2;
  const bool do_bias = false;
  const int dilation = 1;
  const int num_frames = 2;

  conv.set_size_(in_channels, out_channels, kernel_size, do_bias, dilation);

  // Set weights: kernel[0] = [[1.0]], kernel[1] = [[1.0]]
  // With offset: k=0 has offset=-1 (looks at t-1), k=1 has offset=0 (looks at t)
  // So: output = weight[0] * input[t-1] + weight[1] * input[t] = input[t-1] + input[t]
  std::vector<float> weights{1.0f, 1.0f};
  auto it = weights.begin();
  conv.set_weights_(it);

  conv.SetMaxBufferSize(num_frames);

  // 3 calls should trigger rewind.
  Eigen::MatrixXf input(in_channels, num_frames);
  input(0, 0) = 1.0f;
  input(0, 1) = 2.0f;
  for (int i = 0; i < 3; i++)
  {
    conv.Process(input, num_frames);
  }
  auto output = conv.GetOutput().leftCols(num_frames);
  assert(output.rows() == out_channels);
  assert(output.cols() == num_frames);
  // After 3 calls, the last call processes input [1, 2]
  // It should use history from the previous call (which also had [1, 2])
  // output[0] = weight[0] * (history from previous call's last frame) + weight[1] * input[0]
  //           = 1.0 * 2.0 + 1.0 * 1.0 = 3.0
  // output[1] = weight[0] * input[0] + weight[1] * input[1]
  //           = 1.0 * 1.0 + 1.0 * 2.0 = 3.0
  // This tests that ring buffer maintains history across multiple calls
  assert(abs(output(0, 0) - 3.0f) < 0.01f);
  assert(abs(output(0, 1) - 3.0f) < 0.01f);
}

// Test GetOutput() with different num_frames
void test_get_output_different_sizes()
{
  nam::Conv1D conv;
  const int in_channels = 1;
  const int out_channels = 1;
  const int kernel_size = 1;
  const bool do_bias = false;
  const int maxBufferSize = 64;

  conv.set_size_(in_channels, out_channels, kernel_size, do_bias, 1);

  // Identity weight
  std::vector<float> weights{1.0f};
  auto it = weights.begin();
  conv.set_weights_(it);

  conv.SetMaxBufferSize(maxBufferSize);

  Eigen::MatrixXf input(in_channels, 4);
  input(0, 0) = 1.0f;
  input(0, 1) = 2.0f;
  input(0, 2) = 3.0f;
  input(0, 3) = 4.0f;

  conv.Process(input, 4);

  // Get different sized outputs
  auto output_all = conv.GetOutput().leftCols(4);
  assert(output_all.cols() == 4);

  auto output_partial = conv.GetOutput().leftCols(2);
  assert(output_partial.cols() == 2);
  assert(output_partial.rows() == out_channels);
}

// Test set_size_and_weights_
void test_set_size_and_weights()
{
  nam::Conv1D conv;
  const int in_channels = 1;
  const int out_channels = 1;
  const int kernel_size = 2;
  const bool do_bias = false;
  const int dilation = 1;
  const int groups = 1;

  std::vector<float> weights{1.0f, 2.0f};
  auto it = weights.begin();
  conv.set_size_and_weights_(in_channels, out_channels, kernel_size, dilation, do_bias, groups, it);

  assert(conv.get_in_channels() == in_channels);
  assert(conv.get_out_channels() == out_channels);
  assert(conv.get_kernel_size() == kernel_size);
  assert(conv.get_dilation() == dilation);
  assert(it == weights.end()); // All weights should be consumed
}

// Test get_num_weights()
void test_get_num_weights()
{
  nam::Conv1D conv;
  const int in_channels = 2;
  const int out_channels = 3;
  const int kernel_size = 2;
  const bool do_bias = true;
  const int dilation = 1;

  conv.set_size_(in_channels, out_channels, kernel_size, do_bias, dilation);

  // Expected: kernel_size * (out_channels * in_channels) + (bias ? out_channels : 0)
  // = 2 * (3 * 2) + 3 = 2 * 6 + 3 = 15
  long expected = kernel_size * (out_channels * in_channels) + out_channels;
  long actual = conv.get_num_weights();

  assert(actual == expected);

  // Test without bias
  nam::Conv1D conv_no_bias;
  conv_no_bias.set_size_(in_channels, out_channels, kernel_size, false, dilation);
  expected = kernel_size * (out_channels * in_channels);
  actual = conv_no_bias.get_num_weights();
  assert(actual == expected);
}

// Test that Reset() can be called multiple times
void test_reset_multiple()
{
  nam::Conv1D conv;
  const int in_channels = 1;
  const int out_channels = 1;
  const int kernel_size = 1;

  conv.set_size_(in_channels, out_channels, kernel_size, false, 1);

  std::vector<float> weights{1.0f};
  auto it = weights.begin();
  conv.set_weights_(it);

  // Reset with different buffer sizes
  conv.SetMaxBufferSize(64);
  {
    auto output1 = conv.GetOutput().leftCols(64);
    assert(output1.cols() == 64);
  } // output1 goes out of scope here, releasing the block reference

  conv.SetMaxBufferSize(128);
  auto output2 = conv.GetOutput().leftCols(128);
  assert(output2.cols() == 128);
}

// Test basic grouped convolution with 2 groups
void test_process_grouped_basic()
{
  nam::Conv1D conv;
  const int in_channels = 4;
  const int out_channels = 4;
  const int kernel_size = 1;
  const bool do_bias = false;
  const int dilation = 1;
  const int groups = 2;
  const int num_frames = 2;

  conv.set_size_(in_channels, out_channels, kernel_size, do_bias, dilation, groups);

  // For grouped convolution with 2 groups:
  // Group 0: processes in_channels[0:1] -> out_channels[0:1]
  // Group 1: processes in_channels[2:3] -> out_channels[2:3]
  // Each group has out_per_group=2, in_per_group=2
  // Weight layout: for each kernel position k, weights are [group0, group1]
  // For kernel_size=1, we have one weight matrix per group: (2, 2) each
  // Weight ordering: for g=0, then g=1, for each (i,j) in (out_per_group, in_per_group), for each k
  std::vector<float> weights;
  // Group 0, kernel[0]: identity-like weights
  weights.push_back(1.0f); // out[0], in[0]
  weights.push_back(0.0f); // out[0], in[1]
  weights.push_back(0.0f); // out[1], in[0]
  weights.push_back(1.0f); // out[1], in[1]
  // Group 1, kernel[0]: double weights
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

  conv.Process(input, num_frames);
  auto output = conv.GetOutput().leftCols(num_frames);

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
  nam::Conv1D conv;
  const int in_channels = 4;
  const int out_channels = 4;
  const int kernel_size = 1;
  const bool do_bias = true;
  const int dilation = 1;
  const int groups = 2;
  const int num_frames = 1;

  conv.set_size_(in_channels, out_channels, kernel_size, do_bias, dilation, groups);

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

  conv.Process(input, num_frames);
  auto output = conv.GetOutput().leftCols(num_frames);

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
  nam::Conv1D conv;
  const int in_channels = 8;
  const int out_channels = 8;
  const int kernel_size = 1;
  const bool do_bias = false;
  const int dilation = 1;
  const int groups = 4;
  const int num_frames = 1;

  conv.set_size_(in_channels, out_channels, kernel_size, do_bias, dilation, groups);

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

  conv.Process(input, num_frames);
  auto output = conv.GetOutput().leftCols(num_frames);

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

// Test grouped convolution with kernel_size > 1
void test_process_grouped_kernel_size()
{
  nam::Conv1D conv;
  const int in_channels = 4;
  const int out_channels = 4;
  const int kernel_size = 2;
  const bool do_bias = false;
  const int dilation = 1;
  const int groups = 2;
  const int num_frames = 3;

  conv.set_size_(in_channels, out_channels, kernel_size, do_bias, dilation, groups);

  // Each group: 2 in_channels, 2 out_channels, kernel_size=2
  // Weight layout: for each group g, for each (i,j), for each k
  std::vector<float> weights;
  // Group 0, kernel[0] (t-1): identity
  weights.push_back(1.0f); // out[0], in[0]
  weights.push_back(0.0f); // out[0], in[1]
  weights.push_back(0.0f); // out[1], in[0]
  weights.push_back(1.0f); // out[1], in[1]
  // Group 0, kernel[1] (t): double
  weights.push_back(2.0f);
  weights.push_back(0.0f);
  weights.push_back(0.0f);
  weights.push_back(2.0f);
  // Group 1, kernel[0] (t-1): triple
  weights.push_back(3.0f); // out[2], in[2]
  weights.push_back(0.0f); // out[2], in[3]
  weights.push_back(0.0f); // out[3], in[2]
  weights.push_back(3.0f); // out[3], in[3]
  // Group 1, kernel[1] (t): quadruple
  weights.push_back(4.0f);
  weights.push_back(0.0f);
  weights.push_back(0.0f);
  weights.push_back(4.0f);

  auto it = weights.begin();
  conv.set_weights_(it);

  conv.SetMaxBufferSize(64);

  Eigen::MatrixXf input(in_channels, num_frames);
  input(0, 0) = 1.0f; // Group 0
  input(1, 0) = 2.0f;
  input(2, 0) = 3.0f; // Group 1
  input(3, 0) = 4.0f;
  input(0, 1) = 5.0f;
  input(1, 1) = 6.0f;
  input(2, 1) = 7.0f;
  input(3, 1) = 8.0f;
  input(0, 2) = 9.0f;
  input(1, 2) = 10.0f;
  input(2, 2) = 11.0f;
  input(3, 2) = 12.0f;

  conv.Process(input, num_frames);
  auto output = conv.GetOutput().leftCols(num_frames);

  assert(output.rows() == out_channels);
  assert(output.cols() == num_frames);
  // Frame 0: zero-padding for t-1, so only kernel[1] contributes
  // Group 0: out[0] = 2.0 * 1.0 = 2.0, out[1] = 2.0 * 2.0 = 4.0
  assert(std::abs(output(0, 0) - 2.0f) < 0.01f);
  assert(std::abs(output(1, 0) - 4.0f) < 0.01f);
  // Group 1: out[2] = 4.0 * 3.0 = 12.0, out[3] = 4.0 * 4.0 = 16.0
  assert(std::abs(output(2, 0) - 12.0f) < 0.01f);
  assert(std::abs(output(3, 0) - 16.0f) < 0.01f);
  // Frame 1: kernel[0] * input[0] + kernel[1] * input[1]
  // Group 0: out[0] = 1.0 * 1.0 + 2.0 * 5.0 = 11.0
  assert(std::abs(output(0, 1) - 11.0f) < 0.01f);
  assert(std::abs(output(1, 1) - 14.0f) < 0.01f); // 1.0 * 2.0 + 2.0 * 6.0
  // Group 1: out[2] = 3.0 * 3.0 + 4.0 * 7.0 = 37.0
  assert(std::abs(output(2, 1) - 37.0f) < 0.01f);
  assert(std::abs(output(3, 1) - 44.0f) < 0.01f); // 3.0 * 4.0 + 4.0 * 8.0
}

// Test grouped convolution with dilation
void test_process_grouped_dilation()
{
  nam::Conv1D conv;
  const int in_channels = 4;
  const int out_channels = 4;
  const int kernel_size = 2;
  const bool do_bias = false;
  const int dilation = 2;
  const int groups = 2;
  const int num_frames = 4;

  conv.set_size_(in_channels, out_channels, kernel_size, do_bias, dilation, groups);

  // Each group: 2 in_channels, 2 out_channels, kernel_size=2, dilation=2
  std::vector<float> weights;
  // Group 0, kernel[0] (t-2): scale by 1.0
  weights.push_back(1.0f);
  weights.push_back(0.0f);
  weights.push_back(0.0f);
  weights.push_back(1.0f);
  // Group 0, kernel[1] (t): scale by 2.0
  weights.push_back(2.0f);
  weights.push_back(0.0f);
  weights.push_back(0.0f);
  weights.push_back(2.0f);
  // Group 1, kernel[0] (t-2): scale by 3.0
  weights.push_back(3.0f);
  weights.push_back(0.0f);
  weights.push_back(0.0f);
  weights.push_back(3.0f);
  // Group 1, kernel[1] (t): scale by 4.0
  weights.push_back(4.0f);
  weights.push_back(0.0f);
  weights.push_back(0.0f);
  weights.push_back(4.0f);

  auto it = weights.begin();
  conv.set_weights_(it);

  conv.SetMaxBufferSize(64);

  Eigen::MatrixXf input(in_channels, num_frames);
  for (int t = 0; t < num_frames; t++)
  {
    input(0, t) = static_cast<float>(t + 1); // Group 0
    input(1, t) = static_cast<float>(t + 1) * 2;
    input(2, t) = static_cast<float>(t + 1) * 3; // Group 1
    input(3, t) = static_cast<float>(t + 1) * 4;
  }

  conv.Process(input, num_frames);
  auto output = conv.GetOutput().leftCols(num_frames);

  assert(output.rows() == out_channels);
  assert(output.cols() == num_frames);
  // Frame 0: zero-padding for t-2, so only kernel[1] contributes
  // Group 0: out[0] = 2.0 * 1.0 = 2.0
  assert(std::abs(output(0, 0) - 2.0f) < 0.01f);
  // Frame 1: zero-padding for t-2, so only kernel[1] contributes
  assert(std::abs(output(0, 1) - 4.0f) < 0.01f); // 2.0 * 2.0
  // Frame 2: kernel[0] * input[0] + kernel[1] * input[2]
  // Group 0: out[0] = 1.0 * 1.0 + 2.0 * 3.0 = 7.0
  assert(std::abs(output(0, 2) - 7.0f) < 0.01f);
  // Frame 3: kernel[0] * input[1] + kernel[1] * input[3]
  // Group 0: out[0] = 1.0 * 2.0 + 2.0 * 4.0 = 10.0
  assert(std::abs(output(0, 3) - 10.0f) < 0.01f);
}

// Test that groups properly isolate channels (no cross-group interaction)
void test_process_grouped_channel_isolation()
{
  nam::Conv1D conv;
  const int in_channels = 6;
  const int out_channels = 6;
  const int kernel_size = 1;
  const bool do_bias = false;
  const int dilation = 1;
  const int groups = 3;
  const int num_frames = 1;

  conv.set_size_(in_channels, out_channels, kernel_size, do_bias, dilation, groups);

  // Each group processes 2 channels
  // Group 0: in[0:1] -> out[0:1], use weight 1.0
  // Group 1: in[2:3] -> out[2:3], use weight 2.0
  // Group 2: in[4:5] -> out[4:5], use weight 3.0
  std::vector<float> weights;
  // Group 0: identity with scale 1.0
  weights.push_back(1.0f);
  weights.push_back(0.0f);
  weights.push_back(0.0f);
  weights.push_back(1.0f);
  // Group 1: identity with scale 2.0
  weights.push_back(2.0f);
  weights.push_back(0.0f);
  weights.push_back(0.0f);
  weights.push_back(2.0f);
  // Group 2: identity with scale 3.0
  weights.push_back(3.0f);
  weights.push_back(0.0f);
  weights.push_back(0.0f);
  weights.push_back(3.0f);

  auto it = weights.begin();
  conv.set_weights_(it);

  conv.SetMaxBufferSize(64);

  Eigen::MatrixXf input(in_channels, num_frames);
  // Set input so that if groups were not isolated, we'd see cross-contamination
  input(0, 0) = 10.0f; // Group 0
  input(1, 0) = 20.0f;
  input(2, 0) = 30.0f; // Group 1
  input(3, 0) = 40.0f;
  input(4, 0) = 50.0f; // Group 2
  input(5, 0) = 60.0f;

  conv.Process(input, num_frames);
  auto output = conv.GetOutput().leftCols(num_frames);

  assert(output.rows() == out_channels);
  assert(output.cols() == num_frames);
  // Verify isolation: each group only affects its own output channels
  assert(std::abs(output(0, 0) - 10.0f) < 0.01f); // Group 0: 1.0 * 10.0
  assert(std::abs(output(1, 0) - 20.0f) < 0.01f); // Group 0: 1.0 * 20.0
  assert(std::abs(output(2, 0) - 60.0f) < 0.01f); // Group 1: 2.0 * 30.0
  assert(std::abs(output(3, 0) - 80.0f) < 0.01f); // Group 1: 2.0 * 40.0
  assert(std::abs(output(4, 0) - 150.0f) < 0.01f); // Group 2: 3.0 * 50.0
  assert(std::abs(output(5, 0) - 180.0f) < 0.01f); // Group 2: 3.0 * 60.0
  // Verify no cross-contamination: output[0] should not depend on input[2:5]
  // If there was contamination, output[0] would be different
}

// Test weight count calculation for grouped convolutions
void test_get_num_weights_grouped()
{
  nam::Conv1D conv;
  const int in_channels = 8;
  const int out_channels = 6;
  const int kernel_size = 3;
  const bool do_bias = true;
  const int dilation = 1;
  const int groups = 2;

  conv.set_size_(in_channels, out_channels, kernel_size, do_bias, dilation, groups);

  // For grouped convolution:
  // num_weights = kernel_size * (out_channels * in_channels) / num_groups + bias
  // = 3 * (6 * 8) / 2 + 6 = 3 * 48 / 2 + 6 = 3 * 24 + 6 = 72 + 6 = 78
  long expected = kernel_size * (out_channels * in_channels) / groups + out_channels;
  long actual = conv.get_num_weights();

  assert(actual == expected);

  // Test without bias
  nam::Conv1D conv_no_bias;
  conv_no_bias.set_size_(in_channels, out_channels, kernel_size, false, dilation, groups);
  expected = kernel_size * (out_channels * in_channels) / groups;
  actual = conv_no_bias.get_num_weights();
  assert(actual == expected);

  // Test with 4 groups
  nam::Conv1D conv_4groups;
  const int groups_4 = 4;
  conv_4groups.set_size_(8, 8, 2, false, 1, groups_4);
  expected = 2 * (8 * 8) / groups_4; // = 2 * 64 / 4 = 32
  actual = conv_4groups.get_num_weights();
  assert(actual == expected);
}
}; // namespace test_conv1d
