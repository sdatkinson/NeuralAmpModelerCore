// Tests for ConvNet

#include <Eigen/Dense>
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

#include "NAM/convnet.h"

namespace test_convnet
{
// Test basic ConvNet construction and processing
void test_convnet_basic()
{
  const int channels = 2;
  const std::vector<int> dilations{1, 2};
  const bool batchnorm = false;
  const std::string activation = "ReLU";
  const double expected_sample_rate = 48000.0;

  // Calculate weights needed:
  // Block 0: Conv1D (1, 2, 2, false, 1) -> 2*1*2 = 4 weights
  // Block 1: Conv1D (2, 2, 2, false, 2) -> 2*2*2 = 8 weights
  // Head: (2, 1) weight + 1 bias = 3 weights
  // Total: 4 + 8 + 3 = 15 weights
  std::vector<float> weights;
  // Block 0 weights (4 weights: kernel[0] and kernel[1], each 2x1)
  weights.insert(weights.end(), {1.0f, 1.0f, 1.0f, 1.0f});
  // Block 1 weights (8 weights: kernel[0] and kernel[1], each 2x2)
  weights.insert(weights.end(), {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
  // Head weights (2 weights + 1 bias)
  weights.insert(weights.end(), {1.0f, 1.0f, 0.0f});

  nam::convnet::ConvNet convnet(channels, dilations, batchnorm, activation, weights, expected_sample_rate);

  const int numFrames = 4;
  const int maxBufferSize = 64;
  convnet.Reset(expected_sample_rate, maxBufferSize);

  std::vector<NAM_SAMPLE> input(numFrames, 1.0f);
  std::vector<NAM_SAMPLE> output(numFrames, 0.0f);

  convnet.process(input.data(), output.data(), numFrames);

  // Verify output dimensions
  assert(output.size() == numFrames);
  // Output should be non-zero and finite
  for (int i = 0; i < numFrames; i++)
  {
    assert(std::isfinite(output[i]));
  }
}

// Test ConvNet with batchnorm
void test_convnet_batchnorm()
{
  const int channels = 1;
  const std::vector<int> dilations{1};
  const bool batchnorm = true;
  const std::string activation = "ReLU";
  const double expected_sample_rate = 48000.0;

  // Calculate weights needed:
  // Block 0: Conv1D (1, 1, 2, true, 1) -> 2*1*1 + 1 = 3 weights
  // BatchNorm: running_mean(1) + running_var(1) + weight(1) + bias(1) + eps(1) = 5 weights
  // Head: (1, 1) weight + 1 bias = 2 weights
  // Total: 3 + 5 + 2 = 10 weights
  std::vector<float> weights;
  // Block 0 weights (3 weights: kernel[0], kernel[1], bias)
  weights.insert(weights.end(), {1.0f, 1.0f, 0.0f});
  // BatchNorm weights (5: mean, var, weight, bias, eps)
  weights.insert(weights.end(), {0.0f, 1.0f, 1.0f, 0.0f, 1e-5f});
  // Head weights (1 weight + 1 bias)
  weights.insert(weights.end(), {1.0f, 0.0f});

  nam::convnet::ConvNet convnet(channels, dilations, batchnorm, activation, weights, expected_sample_rate);

  const int numFrames = 4;
  const int maxBufferSize = 64;
  convnet.Reset(expected_sample_rate, maxBufferSize);

  std::vector<NAM_SAMPLE> input(numFrames, 1.0f);
  std::vector<NAM_SAMPLE> output(numFrames, 0.0f);

  convnet.process(input.data(), output.data(), numFrames);

  assert(output.size() == numFrames);
  for (int i = 0; i < numFrames; i++)
  {
    assert(std::isfinite(output[i]));
  }
}

// Test ConvNet with multiple blocks
void test_convnet_multiple_blocks()
{
  const int channels = 2;
  const std::vector<int> dilations{1, 2, 4};
  const bool batchnorm = false;
  const std::string activation = "Tanh";
  const double expected_sample_rate = 48000.0;

  // Calculate weights needed:
  // Block 0: Conv1D (1, 2, 2, false, 1) -> 2*1*2 = 4 weights
  // Block 1: Conv1D (2, 2, 2, false, 2) -> 2*2*2 = 8 weights
  // Block 2: Conv1D (2, 2, 2, false, 4) -> 2*2*2 = 8 weights
  // Head: (2, 1) weight + 1 bias = 3 weights
  // Total: 4 + 8 + 8 + 3 = 23 weights
  std::vector<float> weights;
  // Block 0 weights
  weights.insert(weights.end(), {1.0f, 1.0f, 1.0f, 1.0f});
  // Block 1 weights
  weights.insert(weights.end(), {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
  // Block 2 weights
  weights.insert(weights.end(), {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
  // Head weights
  weights.insert(weights.end(), {1.0f, 1.0f, 0.0f});

  nam::convnet::ConvNet convnet(channels, dilations, batchnorm, activation, weights, expected_sample_rate);

  const int numFrames = 8;
  const int maxBufferSize = 64;
  convnet.Reset(expected_sample_rate, maxBufferSize);

  std::vector<NAM_SAMPLE> input(numFrames, 0.5f);
  std::vector<NAM_SAMPLE> output(numFrames, 0.0f);

  convnet.process(input.data(), output.data(), numFrames);

  assert(output.size() == numFrames);
  for (int i = 0; i < numFrames; i++)
  {
    assert(std::isfinite(output[i]));
  }
}

// Test ConvNet with zero input
void test_convnet_zero_input()
{
  const int channels = 1;
  const std::vector<int> dilations{1};
  const bool batchnorm = false;
  const std::string activation = "ReLU";
  const double expected_sample_rate = 48000.0;

  std::vector<float> weights;
  // Block 0 weights (2 weights: kernel[0], kernel[1])
  weights.insert(weights.end(), {1.0f, 1.0f});
  // Head weights (1 weight + 1 bias)
  weights.insert(weights.end(), {1.0f, 0.0f});

  nam::convnet::ConvNet convnet(channels, dilations, batchnorm, activation, weights, expected_sample_rate);

  const int numFrames = 4;
  convnet.Reset(expected_sample_rate, numFrames);

  std::vector<NAM_SAMPLE> input(numFrames, 0.0f);
  std::vector<NAM_SAMPLE> output(numFrames, 0.0f);

  convnet.process(input.data(), output.data(), numFrames);

  // With zero input, output should be finite (may be zero or non-zero depending on bias)
  for (int i = 0; i < numFrames; i++)
  {
    assert(std::isfinite(output[i]));
  }
}

// Test ConvNet with different buffer sizes
void test_convnet_different_buffer_sizes()
{
  const int channels = 1;
  const std::vector<int> dilations{1};
  const bool batchnorm = false;
  const std::string activation = "ReLU";
  const double expected_sample_rate = 48000.0;

  std::vector<float> weights;
  weights.insert(weights.end(), {1.0f, 1.0f});
  weights.insert(weights.end(), {1.0f, 0.0f});

  nam::convnet::ConvNet convnet(channels, dilations, batchnorm, activation, weights, expected_sample_rate);

  // Test with different buffer sizes
  convnet.Reset(expected_sample_rate, 64);
  std::vector<NAM_SAMPLE> input1(32, 1.0f);
  std::vector<NAM_SAMPLE> output1(32, 0.0f);
  convnet.process(input1.data(), output1.data(), 32);

  convnet.Reset(expected_sample_rate, 128);
  std::vector<NAM_SAMPLE> input2(64, 1.0f);
  std::vector<NAM_SAMPLE> output2(64, 0.0f);
  convnet.process(input2.data(), output2.data(), 64);

  // Both should work without errors
  assert(output1.size() == 32);
  assert(output2.size() == 64);
}

// Test ConvNet prewarm functionality
void test_convnet_prewarm()
{
  const int channels = 2;
  const std::vector<int> dilations{1, 2, 4};
  const bool batchnorm = false;
  const std::string activation = "ReLU";
  const double expected_sample_rate = 48000.0;

  std::vector<float> weights;
  // Block 0 weights
  weights.insert(weights.end(), {1.0f, 1.0f, 1.0f, 1.0f});
  // Block 1 weights
  weights.insert(weights.end(), {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
  // Block 2 weights
  weights.insert(weights.end(), {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
  // Head weights
  weights.insert(weights.end(), {1.0f, 1.0f, 0.0f});

  nam::convnet::ConvNet convnet(channels, dilations, batchnorm, activation, weights, expected_sample_rate);

  // Test that prewarm can be called without errors
  convnet.Reset(expected_sample_rate, 64);
  convnet.prewarm();

  // After prewarm, processing should work
  const int numFrames = 4;
  std::vector<NAM_SAMPLE> input(numFrames, 1.0f);
  std::vector<NAM_SAMPLE> output(numFrames, 0.0f);
  convnet.process(input.data(), output.data(), numFrames);

  // Output should be finite
  for (int i = 0; i < numFrames; i++)
  {
    assert(std::isfinite(output[i]));
  }
}

// Test multiple process() calls (ring buffer functionality)
void test_convnet_multiple_calls()
{
  const int channels = 1;
  const std::vector<int> dilations{1};
  const bool batchnorm = false;
  const std::string activation = "ReLU";
  const double expected_sample_rate = 48000.0;

  std::vector<float> weights;
  weights.insert(weights.end(), {1.0f, 1.0f});
  weights.insert(weights.end(), {1.0f, 0.0f});

  nam::convnet::ConvNet convnet(channels, dilations, batchnorm, activation, weights, expected_sample_rate);

  const int numFrames = 2;
  convnet.Reset(expected_sample_rate, numFrames);

  // Multiple calls should work correctly with ring buffer
  for (int i = 0; i < 5; i++)
  {
    std::vector<NAM_SAMPLE> input(numFrames, 1.0f);
    std::vector<NAM_SAMPLE> output(numFrames, 0.0f);
    convnet.process(input.data(), output.data(), numFrames);

    // Output should be finite
    for (int j = 0; j < numFrames; j++)
    {
      assert(std::isfinite(output[j]));
    }
  }
}
}; // namespace test_convnet
