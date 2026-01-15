// Tests for WaveNet Layer

#include <Eigen/Dense>
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

#include "NAM/wavenet.h"

namespace test_wavenet
{
namespace test_layer
{
void test_gated()
{
  // Assert correct nuemrics of the gating activation.
  // Issue 101
  const int conditionSize = 1;
  const int channels = 1;
  const int kernelSize = 1;
  const int dilation = 1;
  const std::string activation = "ReLU";
  const bool gated = true;
  const int groups_input = 1;
  const int groups_condition = 1;
  const int groups_1x1 = 1;
  auto layer = nam::wavenet::_Layer(conditionSize, channels, kernelSize, dilation, activation, gated, groups_input,
                                     groups_condition, groups_1x1);

  // Conv, input mixin, 1x1
  std::vector<float> weights{
    // Conv (weight, bias)  NOTE: 2 channels out bc gated, so shapes are (2,1,1), (2,)
    1.0f, 1.0f, 0.0f, 0.0f,
    // Input mixin (weight only: (2,1,1))
    1.0f, -1.0f,
    // 1x1 (weight (1,1,1), bias (1,))
    // NOTE: Weights are (1,1) on conv, (1,-1), so the inputs sum on the upper channel and cancel on the lower.
    // This should give us a nice zero if the input & condition are the same, so that'll sigmoid to 0.5 for the
    // gate.
    1.0f, 0.0f};
  auto it = weights.begin();
  layer.set_weights_(it);
  assert(it == weights.end());

  const long numFrames = 4;
  layer.SetMaxBufferSize(numFrames);

  Eigen::MatrixXf input, condition, headInput, output;
  input.resize(channels, numFrames);
  condition.resize(conditionSize, numFrames);
  headInput.resize(channels, numFrames);
  output.resize(channels, numFrames);

  const float signalValue = 0.25f;
  input.fill(signalValue);
  condition.fill(signalValue);
  // So input & condition will sum to 0.5 on the top channel (-> ReLU), cancel to 0 on bottom (-> sigmoid)

  headInput.setZero();
  output.setZero();

  layer.Process(input, condition, (int)numFrames);
  // Get outputs
  auto layer_output = layer.GetOutputNextLayer().leftCols((int)numFrames);
  auto head_output = layer.GetOutputHead().leftCols((int)numFrames);
  // Copy to test buffers for verification
  output.leftCols((int)numFrames) = layer_output;
  headInput.leftCols((int)numFrames) = head_output;

  // 0.25 + 0.25 -> 0.5 for conv & input mixin top channel
  // (0 on bottom channel)
  // Top ReLU -> preseves 0.5
  // Bottom sigmoid 0->0.5
  // Product is 0.25
  // 1x1 is unity
  // Skip-connect -> 0.25 (input) + 0.25 (output) -> 0.5 output
  // head output gets 0+0.25 = 0.25
  const float expectedOutput = 0.5;
  const float expectedHeadInput = 0.25;
  for (int i = 0; i < numFrames; i++)
  {
    const float actualOutput = output(0, i);
    const float actualHeadInput = headInput(0, i);
    // std::cout << actualOutput << std::endl;
    assert(actualOutput == expectedOutput);
    assert(actualHeadInput == expectedHeadInput);
  }
}

// Test layer getters
void test_layer_getters()
{
  const int conditionSize = 2;
  const int channels = 4;
  const int kernelSize = 3;
  const int dilation = 2;
  const std::string activation = "Tanh";
  const bool gated = false;
  const int groups_input = 1;

  const int groups_condition = 1;
  const int groups_1x1 = 1;
  auto layer = nam::wavenet::_Layer(conditionSize, channels, kernelSize, dilation, activation, gated, groups_input,
                                     groups_condition, groups_1x1);

  assert(layer.get_channels() == channels);
  assert(layer.get_kernel_size() == kernelSize);
  assert(layer.get_dilation() == dilation);
}

// Test non-gated layer processing
void test_non_gated_layer()
{
  const int conditionSize = 1;
  const int channels = 1;
  const int kernelSize = 1;
  const int dilation = 1;
  const std::string activation = "ReLU";
  const bool gated = false;
  const int groups_input = 1;

  const int groups_condition = 1;
  const int groups_1x1 = 1;
  auto layer = nam::wavenet::_Layer(conditionSize, channels, kernelSize, dilation, activation, gated, groups_input,
                                     groups_condition, groups_1x1);

  // For non-gated: conv outputs 1 channel, input_mixin outputs 1 channel, 1x1 outputs 1 channel
  // Conv: (1,1,1) weight + (1,) bias
  // Input mixin: (1,1) weight (no bias)
  // 1x1: (1,1) weight + (1,) bias
  std::vector<float> weights{// Conv: weight=1.0, bias=0.0
                             1.0f, 0.0f,
                             // Input mixin: weight=1.0
                             1.0f,
                             // 1x1: weight=1.0, bias=0.0
                             1.0f, 0.0f};

  auto it = weights.begin();
  layer.set_weights_(it);
  assert(it == weights.end());

  const int numFrames = 4;
  layer.SetMaxBufferSize(numFrames);

  Eigen::MatrixXf input(channels, numFrames);
  Eigen::MatrixXf condition(conditionSize, numFrames);
  input.fill(1.0f);
  condition.fill(1.0f);

  layer.Process(input, condition, numFrames);

  auto layer_output = layer.GetOutputNextLayer().leftCols(numFrames);
  auto head_output = layer.GetOutputHead().leftCols(numFrames);

  assert(layer_output.rows() == channels);
  assert(layer_output.cols() == numFrames);
  assert(head_output.rows() == channels);
  assert(head_output.cols() == numFrames);

  // With identity-like weights: input=1, condition=1
  // conv output = 1*1 + 0 = 1
  // input_mixin output = 1*1 = 1
  // z = 1 + 1 = 2
  // ReLU(2) = 2
  // 1x1 output = 1*2 + 0 = 2
  // layer_output = input + 1x1_output = 1 + 2 = 3
  // head_output = activated z = 2
  const float expectedLayerOutput = 3.0f;
  const float expectedHeadOutput = 2.0f;
  for (int i = 0; i < numFrames; i++)
  {
    assert(std::abs(layer_output(0, i) - expectedLayerOutput) < 0.01f);
    assert(std::abs(head_output(0, i) - expectedHeadOutput) < 0.01f);
  }
}

// Test layer with different activations
void test_layer_activations()
{
  const int conditionSize = 1;
  const int channels = 1;
  const int kernelSize = 1;
  const int dilation = 1;
  const bool gated = false;

  // Test Tanh activation
  {
    const int groups_input = 1;
    const int groups_condition = 1;
    const int groups_1x1 = 1;
    auto layer = nam::wavenet::_Layer(conditionSize, channels, kernelSize, dilation, "Tanh", gated, groups_input,
                                       groups_condition, groups_1x1);
    std::vector<float> weights{1.0f, 0.0f, 1.0f, 1.0f, 0.0f};
    auto it = weights.begin();
    layer.set_weights_(it);

    const int numFrames = 2;
    layer.SetMaxBufferSize(numFrames);

    Eigen::MatrixXf input(channels, numFrames);
    Eigen::MatrixXf condition(conditionSize, numFrames);
    input.fill(0.5f);
    condition.fill(0.5f);

    layer.Process(input, condition, numFrames);
    auto head_output = layer.GetOutputHead().leftCols(numFrames);

    // Should have applied Tanh activation, so output should be between -1 and 1.
    assert(head_output(0, 0) <= 1.0f);
    assert(head_output(0, 0) >= -1.0f);
  }
}

// Test layer with multiple channels
void test_layer_multichannel()
{
  const int conditionSize = 2;
  const int channels = 3;
  const int kernelSize = 1;
  const int dilation = 1;
  const std::string activation = "ReLU";
  const bool gated = false;
  const int groups_input = 1;

  const int groups_condition = 1;
  const int groups_1x1 = 1;
  auto layer = nam::wavenet::_Layer(conditionSize, channels, kernelSize, dilation, activation, gated, groups_input,
                                     groups_condition, groups_1x1);

  assert(layer.get_channels() == channels);

  const int numFrames = 2;
  layer.SetMaxBufferSize(numFrames);

  // Set identity-like weights (simplified)
  // Conv: (3,3,1) weights + (3,) bias
  // Input mixin: (3,2) weights
  // 1x1: (3,3) weights + (3,) bias
  std::vector<float> weights;
  // Conv weights: 3x3 identity matrix flattened
  for (int i = 0; i < 3; i++)
  {
    for (int j = 0; j < 3; j++)
    {
      weights.push_back((i == j) ? 1.0f : 0.0f);
    }
  }
  // Conv bias: zeros
  weights.insert(weights.end(), {0.0f, 0.0f, 0.0f});
  // Input mixin: (3,2) zeros
  weights.insert(weights.end(), {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
  // 1x1: (3,3) identity
  for (int i = 0; i < 3; i++)
  {
    for (int j = 0; j < 3; j++)
    {
      weights.push_back((i == j) ? 1.0f : 0.0f);
    }
  }
  // 1x1 bias: zeros
  weights.insert(weights.end(), {0.0f, 0.0f, 0.0f});

  auto it = weights.begin();
  layer.set_weights_(it);
  assert(it == weights.end());

  Eigen::MatrixXf input(channels, numFrames);
  Eigen::MatrixXf condition(conditionSize, numFrames);
  input.fill(1.0f);
  condition.fill(1.0f);

  layer.Process(input, condition, numFrames);

  auto layer_output = layer.GetOutputNextLayer().leftCols(numFrames);
  auto head_output = layer.GetOutputHead().leftCols(numFrames);

  assert(layer_output.rows() == channels);
  assert(layer_output.cols() == numFrames);
  assert(head_output.rows() == channels);
  assert(head_output.cols() == numFrames);
}
}; // namespace test_layer

} // namespace test_wavenet