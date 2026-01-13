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
  auto output = conv.GetOutput(maxBufferSize);
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
  // This means: output = 1.0 * input[t] + 2.0 * input[t-1]
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
  auto output = conv.GetOutput(num_frames);

  // Expected outputs (assuming zero padding for first frame):
  // output[0] = 1.0 * 1.0 + 2.0 * 0.0 = 1.0 (with zero padding)
  // output[1] = 1.0 * 2.0 + 2.0 * 1.0 = 4.0
  // output[2] = 1.0 * 3.0 + 2.0 * 2.0 = 7.0
  // output[3] = 1.0 * 4.0 + 2.0 * 3.0 = 10.0
  // But actually, ring buffer stores history, so:
  // After first call, buffer has [1, 2, 3, 4] at write_pos
  // output[0] = 1.0 * 1.0 + 2.0 * 0.0 = 1.0 (lookback 1, reads from write_pos-1 which is 0)
  // Actually, let me think about this more carefully...

  // The convolution reads from the ring buffer with lookback
  // For kernel_size=2, dilation=1, we need lookback of 1 for the first tap
  // So for position i, we read from write_pos - (kernel_size-1-i)*dilation
  // Actually the computation happens with the history in the ring buffer

  // Let's verify the output dimensions and that it's non-zero
  assert(output.rows() == out_channels);
  assert(output.cols() == num_frames);
  assert(abs(output(0, 0) - 1.0f) < 0.01f); // Should have some output
  assert(abs(output(0, 1) - 4.0f) < 0.01f); // Should have some output
  assert(abs(output(0, 2) - 7.0f) < 0.01f); // Should have some output
  assert(abs(output(0, 3) - 10.0f) < 0.01f); // Should have some output
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
  // output = 1.0 * input[t] + 0.0 * input[t-1] + 5.0 = input[t] + 5.0
  std::vector<float> weights{1.0f, 0.0f, 5.0f};
  auto it = weights.begin();
  conv.set_weights_(it);

  conv.SetMaxBufferSize(64);

  Eigen::MatrixXf input(in_channels, num_frames);
  input(0, 0) = 2.0f;
  input(0, 1) = 3.0f;

  conv.Process(input, num_frames);
  auto output = conv.GetOutput(num_frames);

  // Should have bias added
  assert(output.rows() == out_channels);
  assert(output.cols() == num_frames);
  // Output should include both weights and bias
  // For first frame: kernel[0]*input[0] + kernel[1]*zero_padding + bias = 1.0*2.0 + 0.0*0.0 + 5.0 = 7.0
  // For second frame: kernel[0]*input[1] + kernel[1]*input[0] + bias = 1.0*3.0 + 0.0*2.0 + 5.0 = 8.0
  // With zero-padding for first frame:
  // First frame: weight[0]*zero + weight[1]*input[0] + bias = 1.0*0.0 + 0.0*2.0 + 5.0 = 5.0
  // Second frame: weight[0]*input[0] + weight[1]*input[1] + bias = 1.0*2.0 + 0.0*3.0 + 5.0 = 7.0
  // Note: With kernel_size=2, the first output uses zero-padding for the lookback
  assert(std::abs(output(0, 0) - 5.0f) < 0.01f); // First frame: zero*padding + bias
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
  auto output = conv.GetOutput(num_frames);

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
  // With dilation=2: output = 1.0 * input[t] + 2.0 * input[t-2]
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
  auto output = conv.GetOutput(num_frames);

  assert(output.rows() == out_channels);
  assert(output.cols() == num_frames);
  // Output should be computed correctly with dilation
  // out[0] = 1.0 * input[0] + 2.0 * 0.0 (zero-padding) = 1.0
  // out[1] = 1.0 * input[1] + 2.0 * 0.0 (zero-padding) = 2.0
  // out[2] = 1.0 * input[2] + 2.0 * input[0] = 3.0 + 2.0 = 5.0
  // out[3] = 1.0 * input[3] + 2.0 * input[1] = 4.0 + 4.0 = 8.0
  assert(abs(output(0, 0) - 1.0f) < 0.01f);
  assert(abs(output(0, 1) - 2.0f) < 0.01f);
  assert(abs(output(0, 2) - 5.0f) < 0.01f);
  assert(abs(output(0, 3) - 8.0f) < 0.01f);
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
  // output = input[t] + input[t-1]
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
  auto output = conv.GetOutput(num_frames);
  assert(output2.rows() == out_channels);
  assert(output2.cols() == num_frames);
  // output2[0] should use input2[0] + history from input1[1] (last frame of first call)
  // This tests that ring buffer maintains history
  // out[0] = 1.0 * 1.0 + 2.0 * 2.0 = 1.0 + 4.0 = 5.0
  // out[1] = 1.0 * 2.0 + 2.0 * 1.0 = 2.0 + 2.0 = 4.0
  assert(abs(output(0, 0) - 5.0f) < 0.01f);
  assert(abs(output(0, 1) - 4.0f) < 0.01f);
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
  auto output_all = conv.GetOutput(4);
  assert(output_all.cols() == 4);

  auto output_partial = conv.GetOutput(2);
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

  std::vector<float> weights{1.0f, 2.0f};
  auto it = weights.begin();
  conv.set_size_and_weights_(in_channels, out_channels, kernel_size, dilation, do_bias, it);

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
  auto output1 = conv.GetOutput(64);
  assert(output1.cols() == 64);

  conv.SetMaxBufferSize(128);
  auto output2 = conv.GetOutput(128);
  assert(output2.cols() == 128);
}
}; // namespace test_conv1d
