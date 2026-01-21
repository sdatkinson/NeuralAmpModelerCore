// Tests for WaveNet LayerArray

#include <Eigen/Dense>
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

#include "NAM/wavenet.h"

namespace test_wavenet
{
namespace test_layer_array
{
// Test layer array construction and basic processing
void test_layer_array_basic()
{
  const int input_size = 1;
  const int condition_size = 1;
  const int head_size = 1;
  const int channels = 1;
  const int bottleneck = channels;
  const int kernel_size = 1;
  std::vector<int> dilations{1, 2};
  const auto activation = nam::activations::ActivationConfig::simple(nam::activations::ActivationType::ReLU);
  const nam::wavenet::GatingMode gating_mode = nam::wavenet::GatingMode::NONE;
  const bool head_bias = false;
  const int groups = 1;
  const int groups_1x1 = 1;

  nam::wavenet::Head1x1Params head1x1_params(false, channels, 1);
  auto layer_array =
    nam::wavenet::_LayerArray(input_size, condition_size, head_size, channels, bottleneck, kernel_size, dilations,
                              activation, gating_mode, head_bias, groups, groups_1x1, head1x1_params, "");

  const int numFrames = 4;
  layer_array.SetMaxBufferSize(numFrames);

  // Calculate expected number of weights
  // Rechannel: (1,1) weight (no bias)
  // Layer 0: conv (1,1,1) + bias, input_mixin (1,1), 1x1 (1,1) + bias
  // Layer 1: conv (1,1,1) + bias, input_mixin (1,1), 1x1 (1,1) + bias
  // Head rechannel: (1,1) weight (no bias)
  std::vector<float> weights;
  // Rechannel
  weights.push_back(1.0f);
  // Layer 0: conv (weight=1, bias=0), input_mixin (weight=1), 1x1 (weight=1, bias=0)
  weights.insert(weights.end(), {1.0f, 0.0f, 1.0f, 1.0f, 0.0f});
  // Layer 1: conv (weight=1, bias=0), input_mixin (weight=1), 1x1 (weight=1, bias=0)
  weights.insert(weights.end(), {1.0f, 0.0f, 1.0f, 1.0f, 0.0f});
  // Head rechannel
  weights.push_back(1.0f);

  auto it = weights.begin();
  layer_array.set_weights_(it);
  assert(it == weights.end());

  Eigen::MatrixXf layer_inputs(input_size, numFrames);
  Eigen::MatrixXf condition(condition_size, numFrames);
  layer_inputs.fill(1.0f);
  condition.fill(1.0f);

  layer_array.Process(layer_inputs, condition, numFrames);

  auto layer_outputs = layer_array.GetLayerOutputs().leftCols(numFrames);
  auto head_outputs = layer_array.GetHeadOutputs().leftCols(numFrames);

  assert(layer_outputs.rows() == channels);
  assert(layer_outputs.cols() == numFrames);
  assert(head_outputs.rows() == head_size);
  assert(head_outputs.cols() == numFrames);
}

// Test layer array receptive field calculation
void test_layer_array_receptive_field()
{
  const int input_size = 1;
  const int condition_size = 1;
  const int head_size = 1;
  const int channels = 1;
  const int bottleneck = channels;
  const int kernel_size = 3;
  std::vector<int> dilations{1, 2, 4};
  const auto activation = nam::activations::ActivationConfig::simple(nam::activations::ActivationType::ReLU);
  const nam::wavenet::GatingMode gating_mode = nam::wavenet::GatingMode::NONE;
  const bool head_bias = false;
  const int groups = 1;
  const int groups_1x1 = 1;
  nam::wavenet::Head1x1Params head1x1_params(false, channels, 1);

  auto layer_array =
    nam::wavenet::_LayerArray(input_size, condition_size, head_size, channels, bottleneck, kernel_size, dilations,
                              activation, gating_mode, head_bias, groups, groups_1x1, head1x1_params, "");

  long rf = layer_array.get_receptive_field();
  // Expected: sum of dilation * (kernel_size - 1) for each layer
  // Layer 0: 1 * (3-1) = 2
  // Layer 1: 2 * (3-1) = 4
  // Layer 2: 4 * (3-1) = 8
  // Total: 2 + 4 + 8 = 14
  long expected_rf = 1 * (kernel_size - 1) + 2 * (kernel_size - 1) + 4 * (kernel_size - 1);
  assert(rf == expected_rf);
}

// Test layer array with head input from previous array
void test_layer_array_with_head_input()
{
  const int input_size = 1;
  const int condition_size = 1;
  const int head_size = 1;
  const int channels = 1;
  const int bottleneck = channels;
  const int kernel_size = 1;
  std::vector<int> dilations{1};
  const auto activation = nam::activations::ActivationConfig::simple(nam::activations::ActivationType::ReLU);
  const nam::wavenet::GatingMode gating_mode = nam::wavenet::GatingMode::NONE;
  const bool head_bias = false;
  const int groups = 1;
  const int groups_1x1 = 1;
  nam::wavenet::Head1x1Params head1x1_params(false, channels, 1);

  auto layer_array =
    nam::wavenet::_LayerArray(input_size, condition_size, head_size, channels, bottleneck, kernel_size, dilations,
                              activation, gating_mode, head_bias, groups, groups_1x1, head1x1_params, "");

  const int numFrames = 2;
  layer_array.SetMaxBufferSize(numFrames);

  std::vector<float> weights{1.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f};
  auto it = weights.begin();
  layer_array.set_weights_(it);

  Eigen::MatrixXf layer_inputs(input_size, numFrames);
  Eigen::MatrixXf condition(condition_size, numFrames);
  Eigen::MatrixXf head_inputs(head_size, numFrames);
  layer_inputs.fill(1.0f);
  condition.fill(1.0f);
  head_inputs.fill(0.5f);

  layer_array.Process(layer_inputs, condition, head_inputs, numFrames);

  auto head_outputs = layer_array.GetHeadOutputs().leftCols(numFrames);
  assert(head_outputs.rows() == head_size);
  assert(head_outputs.cols() == numFrames);
}
}; // namespace test_layer_array

} // namespace test_wavenet
