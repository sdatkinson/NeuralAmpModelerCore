// Tests for WaveNet head1x1 functionality

#include <Eigen/Dense>
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

#include "NAM/wavenet.h"

namespace test_wavenet
{
namespace test_head1x1
{

void test_head1x1_inactive()
{
  // Test that when head1x1 is inactive, the layer behaves as before
  const int conditionSize = 1;
  const int channels = 2;
  const int bottleneck = channels;
  const int kernelSize = 1;
  const int dilation = 1;
  const std::string activation = "ReLU";
  const bool gated = false;
  const int groups_input = 1;
  const int groups_1x1 = 1;
  const bool head1x1_active = false;
  
  nam::wavenet::Head1x1Params head1x1_params(head1x1_active, channels, 1);
  auto layer = nam::wavenet::_Layer(
    conditionSize, channels, bottleneck, kernelSize, dilation, activation, gated, groups_input, groups_1x1, head1x1_params);

  // Set weights (same as non-gated layer test)
  // With bottleneck=channels=2:
  // Conv: (channels, bottleneck, kernelSize) + bias = (2, 2, 1) + 2 = 4 + 2 = 6 weights
  // Input mixin: (conditionSize, bottleneck) = (1, 2) = 2 weights
  // 1x1: (bottleneck, channels) + bias = (2, 2) + 2 = 4 + 2 = 6 weights
  std::vector<float> weights{// Conv: weights=1.0, bias=0.0
                             1.0f, 0.0f, 0.0f, 1.0f, // weights (identity)
                             0.0f, 0.0f, // bias
                             // Input mixin: weights=1.0
                             1.0f, 1.0f,
                             // 1x1: weights=1.0, bias=0.0
                             1.0f, 0.0f, 0.0f, 1.0f, // weights (identity)
                             0.0f, 0.0f};

  auto it = weights.begin();
  layer.set_weights_(it);
  assert(it == weights.end());

  const int numFrames = 2;
  layer.SetMaxBufferSize(numFrames);

  Eigen::MatrixXf input(channels, numFrames);
  Eigen::MatrixXf condition(conditionSize, numFrames);
  input.fill(1.0f);
  condition.fill(1.0f);

  layer.Process(input, condition, numFrames);

  auto layer_output = layer.GetOutputNextLayer().leftCols(numFrames);
  auto head_output = layer.GetOutputHead().leftCols(numFrames);

  // With identity-like weights: input=1, condition=1
  // conv output = 1*1 + 0 = 1
  // input_mixin output = 1*1 = 1
  // z = 1 + 1 = 2
  // ReLU(2) = 2
  // 1x1 output = 1*2 + 0 = 2
  // layer_output = input + 1x1_output = 1 + 2 = 3
  // head_output = activated z = 2 (no head1x1 applied)
  const float expectedLayerOutput = 3.0f;
  const float expectedHeadOutput = 2.0f;
  for (int i = 0; i < numFrames; i++)
  {
    assert(std::abs(layer_output(0, i) - expectedLayerOutput) < 0.01f);
    assert(std::abs(head_output(0, i) - expectedHeadOutput) < 0.01f);
  }
}

void test_head1x1_active()
{
  // Test that when head1x1 is active, it processes the head output
  const int conditionSize = 1;
  const int channels = 2;
  const int bottleneck = channels;
  const int kernelSize = 1;
  const int dilation = 1;
  const std::string activation = "ReLU";
  const bool gated = false;
  const int groups_input = 1;
  const int groups_1x1 = 1;
  const bool head1x1_active = true;
  const int head1x1_groups = 1;
  
  // Create head1x1 with different out_channels to verify it's being used
  nam::wavenet::Head1x1Params head1x1_params(head1x1_active, channels, head1x1_groups);
  auto layer = nam::wavenet::_Layer(
    conditionSize, channels, bottleneck, kernelSize, dilation, activation, gated, groups_input, groups_1x1, head1x1_params);

  // Set weights: conv, input_mixin, 1x1, head1x1
  // With bottleneck=channels=2:
  // Conv: (channels, bottleneck, kernelSize) + bias = (2, 2, 1) + 2 = 4 + 2 = 6 weights
  // Input mixin: (conditionSize, bottleneck) = (1, 2) = 2 weights
  // 1x1: (bottleneck, channels) + bias = (2, 2) + 2 = 4 + 2 = 6 weights
  // head1x1: (bottleneck, head1x1_out_channels) + bias = (2, 2) + 2 = 4 + 2 = 6 weights
  std::vector<float> weights{// Conv: weights=1.0, bias=0.0
                             1.0f, 0.0f, 0.0f, 1.0f, // weights (identity)
                             0.0f, 0.0f, // bias
                             // Input mixin: weights=1.0
                             1.0f, 1.0f,
                             // 1x1: weights=1.0, bias=0.0
                             1.0f, 0.0f, 0.0f, 1.0f, // weights (identity)
                             0.0f, 0.0f,
                             // head1x1: weights=0.5, bias=0.1
                             0.5f, 0.0f, 0.0f, 0.5f, // weights
                             0.1f, 0.1f};

  auto it = weights.begin();
  layer.set_weights_(it);
  assert(it == weights.end());

  const int numFrames = 2;
  layer.SetMaxBufferSize(numFrames);

  Eigen::MatrixXf input(channels, numFrames);
  Eigen::MatrixXf condition(conditionSize, numFrames);
  input.fill(1.0f);
  condition.fill(1.0f);

  layer.Process(input, condition, numFrames);

  auto layer_output = layer.GetOutputNextLayer().leftCols(numFrames);
  auto head_output = layer.GetOutputHead().leftCols(numFrames);

  // With head1x1 active:
  // conv output = 1*1 + 0 = 1
  // input_mixin output = 1*1 = 1
  // z = 1 + 1 = 2
  // ReLU(2) = 2
  // 1x1 output = 1*2 + 0 = 2
  // layer_output = input + 1x1_output = 1 + 2 = 3
  // head1x1 output = 0.5*2 + 0.1 = 1.1
  // head_output = head1x1_output = 1.1
  const float expectedLayerOutput = 3.0f;
  const float expectedHeadOutput = 1.1f;
  for (int i = 0; i < numFrames; i++)
  {
    assert(std::abs(layer_output(0, i) - expectedLayerOutput) < 0.02f);
    assert(std::abs(head_output(0, i) - expectedHeadOutput) < 0.02f);
  }
}

void test_head1x1_gated()
{
  // Test head1x1 with gated activation - simplified to test dimensions only
  const int conditionSize = 1;
  const int channels = 2;
  const int bottleneck = channels;
  const int kernelSize = 1;
  const int dilation = 1;
  const std::string activation = "ReLU";
  const bool gated = true;
  const int groups_input = 1;
  const int groups_1x1 = 1;
  const bool head1x1_active = true;
  const int head1x1_groups = 1;
  
  nam::wavenet::Head1x1Params head1x1_params(head1x1_active, channels, head1x1_groups);
  auto layer = nam::wavenet::_Layer(
    conditionSize, channels, bottleneck, kernelSize, dilation, activation, gated, groups_input, groups_1x1, head1x1_params);

  // For gated: conv outputs 2*bottleneck, input_mixin outputs 2*bottleneck, 1x1 outputs channels
  // head1x1 outputs channels
  // With gated=true, bottleneck=channels=2:
  // Conv: (channels, 2*bottleneck, kernelSize) + bias = (2, 4, 1) + 4 = 8 + 4 = 12 weights
  // Input mixin: (conditionSize, 2*bottleneck) = (1, 4) = 4 weights
  // 1x1: (bottleneck, channels) + bias = (2, 2) + 2 = 4 + 2 = 6 weights
  // head1x1: (bottleneck, head1x1_out_channels) + bias = (2, 2) + 2 = 4 + 2 = 6 weights
  std::vector<float> weights{
    // Conv: (channels, 2*bottleneck, kernelSize=1) weights + (2*bottleneck,) bias
    // Weight layout: for each kernel position, for each output channel, for each input channel
    // For kernel position 0:
    // Output channel 0: connects to input channels 0 and 1
    1.0f, 0.0f, // output channel 0
    // Output channel 1: connects to input channels 0 and 1
    0.0f, 1.0f, // output channel 1
    // Output channel 2: connects to input channels 0 and 1
    1.0f, 0.0f, // output channel 2
    // Output channel 3: connects to input channels 0 and 1
    0.0f, 1.0f, // output channel 3
    // Bias: 2*bottleneck values
    0.0f, 0.0f, 0.0f, 0.0f,
    // Input mixin: (conditionSize, 2*bottleneck) weights (all 1.0 for simplicity)
    1.0f, 1.0f, 1.0f, 1.0f,
    // 1x1: (bottleneck, channels) weights + (channels,) bias (identity)
    1.0f, 0.0f, 0.0f, 1.0f, // weights (identity)
    0.0f, 0.0f, // bias
    // head1x1: (bottleneck, head1x1_out_channels) weights + (head1x1_out_channels,) bias
    0.5f, 0.0f, 0.0f, 0.5f, // weights
    0.1f, 0.1f};

  auto it = weights.begin();
  layer.set_weights_(it);
  assert(it == weights.end());

  const int numFrames = 2;
  layer.SetMaxBufferSize(numFrames);

  Eigen::MatrixXf input(channels, numFrames);
  Eigen::MatrixXf condition(conditionSize, numFrames);
  input.fill(1.0f);
  condition.fill(1.0f);

  layer.Process(input, condition, numFrames);

  auto layer_output = layer.GetOutputNextLayer().leftCols(numFrames);
  auto head_output = layer.GetOutputHead().leftCols(numFrames);

  // Test that dimensions are correct and outputs are reasonable
  // Layer output should have 'channels' rows
  assert(layer_output.rows() == channels);
  assert(layer_output.cols() == numFrames);
  
  // Head output should have head1x1_out_channels rows (same as channels in this case)
  assert(head_output.rows() == head1x1_params.out_channels);
  assert(head_output.cols() == numFrames);

  // Verify the outputs are reasonable (not NaN, not infinite)
  for (int i = 0; i < numFrames; i++)
  {
    for (int c = 0; c < channels; c++)
    {
      assert(!std::isnan(layer_output(c, i)));
      assert(!std::isinf(layer_output(c, i)));
    }
    for (int c = 0; c < head1x1_params.out_channels; c++)
    {
      assert(!std::isnan(head_output(c, i)));
      assert(!std::isinf(head_output(c, i)));
    }
  }
}

void test_head1x1_groups()
{
  // Test head1x1 with groups
  const int conditionSize = 1;
  const int channels = 4;
  const int bottleneck = channels;
  const int kernelSize = 1;
  const int dilation = 1;
  const std::string activation = "ReLU";
  const bool gated = false;
  const int groups_input = 1;
  const int groups_1x1 = 1;
  const bool head1x1_active = true;
  const int head1x1_groups = 2; // Grouped head1x1
  
  nam::wavenet::Head1x1Params head1x1_params(head1x1_active, channels, head1x1_groups);
  auto layer = nam::wavenet::_Layer(
    conditionSize, channels, bottleneck, kernelSize, dilation, activation, gated, groups_input, groups_1x1, head1x1_params);

  // With grouped head1x1, we need to provide weights for each group
  // For groups=2, channels=4, bottleneck=4: each group has 2 in_channels and 2 out_channels
  // With bottleneck=channels=4:
  // Conv: (channels, bottleneck, kernelSize) + bias = (4, 4, 1) + 4 = 16 + 4 = 20 weights
  // Input mixin: (conditionSize, bottleneck) = (1, 4) = 4 weights
  // 1x1: (bottleneck, channels) + bias = (4, 4) + 4 = 16 + 4 = 20 weights
  // head1x1: grouped with groups=2, (bottleneck, head1x1_out_channels) + bias = (4, 4) + 4 = 16 + 4 = 20 weights
  // For grouped conv1x1: weights are organized per group
  // Each group: (out_channels_per_group, in_channels_per_group) + bias_per_group = (2, 2) + 2 = 6 weights per group
  std::vector<float> weights{
    // Conv: (channels, bottleneck, kernelSize=1) + bias (identity weights)
    1.0f, 0.0f, 0.0f, 0.0f, // output channel 0
    0.0f, 1.0f, 0.0f, 0.0f, // output channel 1
    0.0f, 0.0f, 1.0f, 0.0f, // output channel 2
    0.0f, 0.0f, 0.0f, 1.0f, // output channel 3
    // Conv bias: bottleneck values
    0.0f, 0.0f, 0.0f, 0.0f,
    // Input mixin: (conditionSize, bottleneck) weights
    1.0f, 1.0f, 1.0f, 1.0f,
    // 1x1: (bottleneck, channels) + bias (identity weights)
    1.0f, 0.0f, 0.0f, 0.0f, // output channel 0
    0.0f, 1.0f, 0.0f, 0.0f, // output channel 1
    0.0f, 0.0f, 1.0f, 0.0f, // output channel 2
    0.0f, 0.0f, 0.0f, 1.0f, // output channel 3
    // 1x1 bias: channels values
    0.0f, 0.0f, 0.0f, 0.0f,
    // head1x1: for each group, (out_channels_per_group, in_channels_per_group) + bias_per_group
    // Group 1: (2,2) weights + 2 bias
    1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
    // Group 2: (2,2) weights + 2 bias  
    1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f};

  auto it = weights.begin();
  layer.set_weights_(it);
  assert(it == weights.end());

  const int numFrames = 2;
  layer.SetMaxBufferSize(numFrames);

  Eigen::MatrixXf input(channels, numFrames);
  Eigen::MatrixXf condition(conditionSize, numFrames);
  input.fill(1.0f);
  condition.fill(1.0f);

  layer.Process(input, condition, numFrames);

  auto layer_output = layer.GetOutputNextLayer().leftCols(numFrames);
  auto head_output = layer.GetOutputHead().leftCols(numFrames);

  // With grouped head1x1 (identity weights), output should be similar to input
  // but processed through the groups
  for (int i = 0; i < numFrames; i++)
  {
    // Check that outputs are reasonable (not NaN, not infinite)
    assert(!std::isnan(layer_output(0, i)));
    assert(!std::isinf(layer_output(0, i)));
    assert(!std::isnan(head_output(0, i)));
    assert(!std::isinf(head_output(0, i)));
  }
}

void test_head1x1_different_out_channels()
{
  // Test head1x1 with different out_channels than input channels
  const int conditionSize = 1;
  const int channels = 4;
  const int bottleneck = channels;
  const int kernelSize = 1;
  const int dilation = 1;
  const std::string activation = "ReLU";
  const bool gated = false;
  const int groups_input = 1;
  const int groups_1x1 = 1;
  const bool head1x1_active = true;
  const int head1x1_out_channels = 2; // Different from bottleneck
  const int head1x1_groups = 1;
  
  nam::wavenet::Head1x1Params head1x1_params(head1x1_active, head1x1_out_channels, head1x1_groups);
  auto layer = nam::wavenet::_Layer(
    conditionSize, channels, bottleneck, kernelSize, dilation, activation, gated, groups_input, groups_1x1, head1x1_params);

  // head1x1 should map from bottleneck to head1x1_out_channels
  // With channels=4, bottleneck=4, head1x1_out_channels=2:
  // Conv: (channels, bottleneck, kernelSize) + bias = (4, 4, 1) + 4 = 16 + 4 = 20 weights
  // Input mixin: (conditionSize, bottleneck) = (1, 4) = 4 weights
  // 1x1: (bottleneck, channels) + bias = (4, 4) + 4 = 16 + 4 = 20 weights
  // head1x1: (bottleneck, head1x1_out_channels) + bias = (4, 2) + 2 = 8 + 2 = 10 weights
  std::vector<float> weights{
    // Conv: (channels, bottleneck, kernelSize=1) + bias (identity weights)
    1.0f, 0.0f, 0.0f, 0.0f, // output channel 0
    0.0f, 1.0f, 0.0f, 0.0f, // output channel 1
    0.0f, 0.0f, 1.0f, 0.0f, // output channel 2
    0.0f, 0.0f, 0.0f, 1.0f, // output channel 3
    // Conv bias: bottleneck values
    0.0f, 0.0f, 0.0f, 0.0f,
    // Input mixin: (conditionSize, bottleneck) weights
    1.0f, 1.0f, 1.0f, 1.0f,
    // 1x1: (bottleneck, channels) + bias (identity weights)
    1.0f, 0.0f, 0.0f, 0.0f, // output channel 0
    0.0f, 1.0f, 0.0f, 0.0f, // output channel 1
    0.0f, 0.0f, 1.0f, 0.0f, // output channel 2
    0.0f, 0.0f, 0.0f, 1.0f, // output channel 3
    // 1x1 bias: channels values
    0.0f, 0.0f, 0.0f, 0.0f,
    // head1x1: (bottleneck, head1x1_out_channels) + bias
    0.5f, 0.5f, 0.5f, 0.5f, // weights for output channel 0 (average all input channels)
    0.5f, 0.5f, 0.5f, 0.5f, // weights for output channel 1 (average all input channels)
    0.1f, 0.1f}; // bias for output channels 0 and 1

  auto it = weights.begin();
  layer.set_weights_(it);
  assert(it == weights.end());

  const int numFrames = 2;
  layer.SetMaxBufferSize(numFrames);

  Eigen::MatrixXf input(channels, numFrames);
  Eigen::MatrixXf condition(conditionSize, numFrames);
  input.fill(1.0f);
  condition.fill(1.0f);

  layer.Process(input, condition, numFrames);

  auto layer_output = layer.GetOutputNextLayer().leftCols(numFrames);
  auto head_output = layer.GetOutputHead().leftCols(numFrames);

  // head_output should have head1x1_out_channels rows, not channels
  assert(head_output.rows() == head1x1_out_channels);
  assert(head_output.cols() == numFrames);

  // Verify the outputs are reasonable
  for (int i = 0; i < numFrames; i++)
  {
    assert(!std::isnan(layer_output(0, i)));
    assert(!std::isinf(layer_output(0, i)));
    for (int c = 0; c < head1x1_out_channels; c++)
    {
      assert(!std::isnan(head_output(c, i)));
      assert(!std::isinf(head_output(c, i)));
    }
  }
}

}; // namespace test_head1x1
} // namespace test_wavenet
