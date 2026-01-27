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
// Helper function to create default (inactive) FiLM parameters
static nam::wavenet::_FiLMParams make_default_film_params()
{
  return nam::wavenet::_FiLMParams(false, false);
}

// Helper function to create a LayerArray with default FiLM parameters
static nam::wavenet::_LayerArray make_layer_array(
  const int input_size, const int condition_size, const int head_size, const int channels, const int bottleneck,
  const int kernel_size, const std::vector<int>& dilations, const nam::activations::ActivationConfig& activation_config,
  const nam::wavenet::GatingMode gating_mode, const bool head_bias, const int groups_input,
  const int groups_input_mixin, const int groups_1x1, const nam::wavenet::Head1x1Params& head1x1_params,
  const nam::activations::ActivationConfig& secondary_activation_config)
{
  auto film_params = make_default_film_params();
  // Duplicate activation_config, gating_mode, and secondary_activation_config for each layer (based on dilations size)
  std::vector<nam::activations::ActivationConfig> activation_configs(dilations.size(), activation_config);
  std::vector<nam::wavenet::GatingMode> gating_modes(dilations.size(), gating_mode);
  std::vector<nam::activations::ActivationConfig> secondary_activation_configs(
    dilations.size(), secondary_activation_config);
  std::vector<int> dilations_copy = dilations; // Make a copy since we need to move it
  nam::wavenet::LayerArrayParams params(
    input_size, condition_size, head_size, channels, bottleneck, kernel_size, std::move(dilations_copy),
    std::move(activation_configs), std::move(gating_modes), head_bias, groups_input, groups_input_mixin, groups_1x1,
    head1x1_params, std::move(secondary_activation_configs), film_params, film_params, film_params, film_params,
    film_params, film_params, film_params, film_params);
  return nam::wavenet::_LayerArray(params);
}
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
  const int groups_input_mixin = 1;
  const int groups_1x1 = 1;

  nam::wavenet::Head1x1Params head1x1_params(false, channels, 1);
  auto layer_array = make_layer_array(input_size, condition_size, head_size, channels, bottleneck, kernel_size,
                                      dilations, activation, gating_mode, head_bias, groups, groups_input_mixin,
                                      groups_1x1, head1x1_params, nam::activations::ActivationConfig{});

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
  const int groups_input_mixin = 1;
  const int groups_1x1 = 1;
  nam::wavenet::Head1x1Params head1x1_params(false, channels, 1);

  auto layer_array = make_layer_array(input_size, condition_size, head_size, channels, bottleneck, kernel_size,
                                      dilations, activation, gating_mode, head_bias, groups, groups_input_mixin,
                                      groups_1x1, head1x1_params, nam::activations::ActivationConfig{});

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
  const int groups_input_mixin = 1;
  const int groups_1x1 = 1;
  nam::wavenet::Head1x1Params head1x1_params(false, channels, 1);

  auto layer_array = make_layer_array(input_size, condition_size, head_size, channels, bottleneck, kernel_size,
                                      dilations, activation, gating_mode, head_bias, groups, groups_input_mixin,
                                      groups_1x1, head1x1_params, nam::activations::ActivationConfig{});

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

// Test layer array with different activation configs, gating modes, and secondary activations for each layer
void test_layer_array_different_activations()
{
  const int input_size = 1;
  const int condition_size = 1;
  const int head_size = 1;
  const int channels = 1;
  const int bottleneck = channels;
  const int kernel_size = 1;
  std::vector<int> dilations{1, 2, 3};
  const bool head_bias = false;
  const int groups = 1;
  const int groups_input_mixin = 1;
  const int groups_1x1 = 1;
  nam::wavenet::Head1x1Params head1x1_params(false, channels, 1);

  // Create different activation configs for each layer
  std::vector<nam::activations::ActivationConfig> activation_configs;
  activation_configs.push_back(nam::activations::ActivationConfig::simple(nam::activations::ActivationType::ReLU));
  activation_configs.push_back(nam::activations::ActivationConfig::simple(nam::activations::ActivationType::Tanh));
  activation_configs.push_back(nam::activations::ActivationConfig::simple(nam::activations::ActivationType::Sigmoid));

  // Create different gating modes for each layer: NONE, GATED, BLENDED
  std::vector<nam::wavenet::GatingMode> gating_modes;
  gating_modes.push_back(nam::wavenet::GatingMode::NONE);
  gating_modes.push_back(nam::wavenet::GatingMode::GATED);
  gating_modes.push_back(nam::wavenet::GatingMode::BLENDED);

  // Create different secondary activation configs for gated/blended layers
  std::vector<nam::activations::ActivationConfig> secondary_activation_configs;
  secondary_activation_configs.push_back(nam::activations::ActivationConfig{}); // NONE mode - empty config
  secondary_activation_configs.push_back(
    nam::activations::ActivationConfig::simple(nam::activations::ActivationType::Sigmoid)); // GATED mode - Sigmoid
  secondary_activation_configs.push_back(
    nam::activations::ActivationConfig::simple(nam::activations::ActivationType::Tanh)); // BLENDED mode - Tanh

  // Verify we have the right number of configs
  assert(activation_configs.size() == dilations.size());
  assert(gating_modes.size() == dilations.size());
  assert(secondary_activation_configs.size() == dilations.size());

  auto film_params = make_default_film_params();
  nam::wavenet::LayerArrayParams params(input_size, condition_size, head_size, channels, bottleneck, kernel_size,
                                        std::move(dilations), std::move(activation_configs), std::move(gating_modes),
                                        head_bias, groups, groups_input_mixin, groups_1x1, head1x1_params,
                                        std::move(secondary_activation_configs), film_params, film_params, film_params,
                                        film_params, film_params, film_params, film_params, film_params);
  nam::wavenet::_LayerArray layer_array(params);

  const int numFrames = 4;
  layer_array.SetMaxBufferSize(numFrames);

  // Set weights: all weights = 1.0, biases = 0.0
  // Rechannel: (1,1) weight (no bias)
  // Layer 0 (NONE): conv (1->1, weight=1, bias=0), input_mixin (1->1, weight=1), 1x1 (1->1, weight=1, bias=0)
  // Layer 1 (GATED): conv (1->2, weight=1, bias=0), input_mixin (1->2, weight=1), 1x1 (1->1, weight=1, bias=0)
  //   Note: GATED doubles bottleneck channels, so conv outputs 2 channels, but 1x1 takes 1 channel input
  // Layer 2 (BLENDED): conv (1->2, weight=1, bias=0), input_mixin (1->2, weight=1), 1x1 (1->1, weight=1, bias=0)
  // Head rechannel: (1,1) weight (no bias)
  std::vector<float> weights;
  // Rechannel
  weights.push_back(1.0f);
  // Layer 0 (NONE): conv(1->1) + bias, input_mixin(1->1), 1x1(1->1) + bias
  weights.insert(weights.end(), {1.0f, 0.0f, 1.0f, 1.0f, 0.0f});
  // Layer 1 (GATED): conv(1->2) + bias, input_mixin(1->2), 1x1(1->1) + bias
  // conv: 1 input * 2 output = 2 weights, 2 biases
  // input_mixin: 1 input * 2 output = 2 weights
  // 1x1: 1 input * 1 output = 1 weight, 1 bias
  weights.insert(weights.end(), {1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 0.0f});
  // Layer 2 (BLENDED): same as GATED
  weights.insert(weights.end(), {1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 0.0f});
  // Head rechannel
  weights.push_back(1.0f);

  auto it = weights.begin();
  layer_array.set_weights_(it);
  assert(it == weights.end());

  // Test with positive input values to verify all activations work
  Eigen::MatrixXf layer_inputs(input_size, numFrames);
  Eigen::MatrixXf condition(condition_size, numFrames);
  layer_inputs.fill(2.0f); // Use larger value to make differences more pronounced
  condition.fill(1.0f);

  layer_array.Process(layer_inputs, condition, numFrames);

  auto layer_outputs = layer_array.GetLayerOutputs().leftCols(numFrames);
  auto head_outputs = layer_array.GetHeadOutputs().leftCols(numFrames);

  assert(layer_outputs.rows() == channels);
  assert(layer_outputs.cols() == numFrames);
  assert(head_outputs.rows() == head_size);
  assert(head_outputs.cols() == numFrames);

  // Verify output is reasonable (not NaN, not infinite)
  for (int i = 0; i < numFrames; i++)
  {
    assert(std::isfinite(head_outputs(0, i)));
    assert(std::isfinite(layer_outputs(0, i)));
  }

  // Verify that the different configurations produce valid outputs
  // The key is that:
  // - Layer 0 uses ReLU with NONE gating (standard activation)
  // - Layer 1 uses Tanh with GATED gating (Tanh primary, Sigmoid secondary)
  // - Layer 2 uses Sigmoid with BLENDED gating (Sigmoid primary, Tanh secondary)
  // Each should produce different behavior

  // Verify all outputs are finite and reasonable
  for (int i = 0; i < numFrames; i++)
  {
    assert(std::isfinite(head_outputs(0, i)));
    assert(std::isfinite(layer_outputs(0, i)));
  }

  // Now create a comparison LayerArray with all ReLU activations and NONE gating
  // This should produce different outputs since it doesn't have gating or saturating activations
  std::vector<int> dilations_all_relu{1, 2, 3}; // Copy of dilations for the second layer array
  std::vector<nam::activations::ActivationConfig> all_relu_configs(
    dilations_all_relu.size(), nam::activations::ActivationConfig::simple(nam::activations::ActivationType::ReLU));
  std::vector<nam::wavenet::GatingMode> all_none_gating_modes(
    dilations_all_relu.size(), nam::wavenet::GatingMode::NONE);
  std::vector<nam::activations::ActivationConfig> all_empty_secondary_configs(
    dilations_all_relu.size(), nam::activations::ActivationConfig{});
  nam::wavenet::LayerArrayParams params_all_relu(
    input_size, condition_size, head_size, channels, bottleneck, kernel_size, std::move(dilations_all_relu),
    std::move(all_relu_configs), std::move(all_none_gating_modes), head_bias, groups, groups_input_mixin, groups_1x1,
    head1x1_params, std::move(all_empty_secondary_configs), film_params, film_params, film_params, film_params,
    film_params, film_params, film_params, film_params);
  nam::wavenet::_LayerArray layer_array_all_relu(params_all_relu);
  layer_array_all_relu.SetMaxBufferSize(numFrames);

  // Create weights for all-NONE version (simpler, no gating)
  std::vector<float> weights_all_none;
  weights_all_none.push_back(1.0f); // Rechannel
  weights_all_none.insert(weights_all_none.end(), {1.0f, 0.0f, 1.0f, 1.0f, 0.0f}); // Layer 0
  weights_all_none.insert(weights_all_none.end(), {1.0f, 0.0f, 1.0f, 1.0f, 0.0f}); // Layer 1
  weights_all_none.insert(weights_all_none.end(), {1.0f, 0.0f, 1.0f, 1.0f, 0.0f}); // Layer 2
  weights_all_none.push_back(1.0f); // Head rechannel

  auto it_all_none = weights_all_none.begin();
  layer_array_all_relu.set_weights_(it_all_none);

  // Process with same positive input
  layer_array_all_relu.Process(layer_inputs, condition, numFrames);
  auto head_outputs_all_relu = layer_array_all_relu.GetHeadOutputs().leftCols(numFrames);

  // Verify outputs are different - the mixed configuration (with gating and saturating activations)
  // should produce different values than all-ReLU with no gating
  bool outputs_differ_from_all_relu = false;
  for (int i = 0; i < numFrames; i++)
  {
    if (std::abs(head_outputs(0, i) - head_outputs_all_relu(0, i)) > 0.1f)
    {
      outputs_differ_from_all_relu = true;
      break;
    }
  }
  assert(outputs_differ_from_all_relu); // Mixed config should produce different outputs than all-ReLU+NONE
}
}; // namespace test_layer_array

} // namespace test_wavenet
