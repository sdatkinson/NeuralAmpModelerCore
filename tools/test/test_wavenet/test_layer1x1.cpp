// Tests for WaveNet layer1x1 functionality

#include <Eigen/Dense>
#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "NAM/wavenet.h"

namespace test_wavenet
{
namespace test_layer1x1
{
// Helper function to create default (inactive) FiLM parameters
static nam::wavenet::_FiLMParams make_default_film_params()
{
  return nam::wavenet::_FiLMParams(false, false);
}

// Helper function to create a Layer with default FiLM parameters
static nam::wavenet::_Layer make_layer(const int condition_size, const int channels, const int bottleneck,
                                       const int kernel_size, const int dilation,
                                       const nam::activations::ActivationConfig& activation_config,
                                       const nam::wavenet::GatingMode gating_mode, const int groups_input,
                                       const int groups_input_mixin,
                                       const nam::wavenet::Layer1x1Params& layer1x1_params,
                                       const nam::wavenet::Head1x1Params& head1x1_params,
                                       const nam::activations::ActivationConfig& secondary_activation_config)
{
  auto film_params = make_default_film_params();
  nam::wavenet::LayerParams layer_params(condition_size, channels, bottleneck, kernel_size, dilation, activation_config,
                                         gating_mode, groups_input, groups_input_mixin, layer1x1_params, head1x1_params,
                                         secondary_activation_config, film_params, film_params, film_params,
                                         film_params, film_params, film_params, film_params, film_params);
  return nam::wavenet::_Layer(layer_params);
}

void test_layer1x1_active()
{
  // Test that when layer1x1 is active (default), it processes the activation output
  const int conditionSize = 1;
  const int channels = 2;
  const int bottleneck = channels;
  const int kernelSize = 1;
  const int dilation = 1;
  const auto activation = nam::activations::ActivationConfig::simple(nam::activations::ActivationType::ReLU);
  const nam::wavenet::GatingMode gating_mode = nam::wavenet::GatingMode::NONE;
  const int groups_input = 1;
  const int groups_input_mixin = 1;
  const bool layer1x1_active = true;
  const int layer1x1_groups = 1;

  nam::wavenet::Layer1x1Params layer1x1_params(layer1x1_active, layer1x1_groups);
  nam::wavenet::Head1x1Params head1x1_params(false, channels, 1);
  auto layer =
    make_layer(conditionSize, channels, bottleneck, kernelSize, dilation, activation, gating_mode, groups_input,
               groups_input_mixin, layer1x1_params, head1x1_params, nam::activations::ActivationConfig{});

  // Set weights: conv, input_mixin, layer1x1
  // With bottleneck=channels=2:
  // Conv: (channels, bottleneck, kernelSize) + bias = (2, 2, 1) + 2 = 4 + 2 = 6 weights
  // Input mixin: (conditionSize, bottleneck) = (1, 2) = 2 weights
  // layer1x1: (bottleneck, channels) + bias = (2, 2) + 2 = 4 + 2 = 6 weights
  std::vector<float> weights{
    // Conv: weights=1.0, bias=0.0 (identity)
    1.0f, 0.0f, 0.0f, 1.0f, // weights
    0.0f, 0.0f, // bias
    // Input mixin: weights=1.0
    1.0f, 1.0f,
    // layer1x1: weights=1.0, bias=0.0 (identity)
    1.0f, 0.0f, 0.0f, 1.0f, // weights
    0.0f, 0.0f // bias
  };

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

  // With identity-like weights: input=1, condition=1
  // conv output = 1*1 + 0 = 1
  // input_mixin output = 1*1 = 1
  // z = 1 + 1 = 2
  // ReLU(2) = 2
  // layer1x1 output = 1*2 + 0 = 2
  // layer_output = input + layer1x1_output = 1 + 2 = 3
  const float expectedLayerOutput = 3.0f;
  for (int i = 0; i < numFrames; i++)
  {
    assert(std::abs(layer_output(0, i) - expectedLayerOutput) < 0.01f);
    assert(std::abs(layer_output(1, i) - expectedLayerOutput) < 0.01f);
  }
}

void test_layer1x1_inactive()
{
  // Test that when layer1x1 is inactive, residual connection passes through input directly
  const int conditionSize = 1;
  const int channels = 2;
  const int bottleneck = channels; // Must equal channels when layer1x1 is inactive
  const int kernelSize = 1;
  const int dilation = 1;
  const auto activation = nam::activations::ActivationConfig::simple(nam::activations::ActivationType::ReLU);
  const nam::wavenet::GatingMode gating_mode = nam::wavenet::GatingMode::NONE;
  const int groups_input = 1;
  const int groups_input_mixin = 1;
  const bool layer1x1_active = false;
  const int layer1x1_groups = 1;

  nam::wavenet::Layer1x1Params layer1x1_params(layer1x1_active, layer1x1_groups);
  nam::wavenet::Head1x1Params head1x1_params(false, channels, 1);
  auto layer =
    make_layer(conditionSize, channels, bottleneck, kernelSize, dilation, activation, gating_mode, groups_input,
               groups_input_mixin, layer1x1_params, head1x1_params, nam::activations::ActivationConfig{});

  // Set weights: conv, input_mixin (no layer1x1 weights needed)
  // With bottleneck=channels=2:
  // Conv: (channels, bottleneck, kernelSize) + bias = (2, 2, 1) + 2 = 4 + 2 = 6 weights
  // Input mixin: (conditionSize, bottleneck) = (1, 2) = 2 weights
  std::vector<float> weights{
    // Conv: weights=1.0, bias=0.0 (identity)
    1.0f, 0.0f, 0.0f, 1.0f, // weights
    0.0f, 0.0f, // bias
    // Input mixin: weights=1.0
    1.0f, 1.0f
    // No layer1x1 weights since it's inactive
  };

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

  // With layer1x1 inactive:
  // conv output = 1*1 + 0 = 1
  // input_mixin output = 1*1 = 1
  // z = 1 + 1 = 2
  // ReLU(2) = 2
  // layer1x1 is skipped
  // layer_output = input (identity residual) = 1
  const float expectedLayerOutput = 1.0f;
  for (int i = 0; i < numFrames; i++)
  {
    assert(std::abs(layer_output(0, i) - expectedLayerOutput) < 0.01f);
    assert(std::abs(layer_output(1, i) - expectedLayerOutput) < 0.01f);
  }
}

void test_layer1x1_inactive_bottleneck_mismatch()
{
  // Test that creating a layer with layer1x1 inactive but bottleneck != channels throws an error
  const int conditionSize = 1;
  const int channels = 2;
  const int bottleneck = 4; // Different from channels - should fail
  const int kernelSize = 1;
  const int dilation = 1;
  const auto activation = nam::activations::ActivationConfig::simple(nam::activations::ActivationType::ReLU);
  const nam::wavenet::GatingMode gating_mode = nam::wavenet::GatingMode::NONE;
  const int groups_input = 1;
  const int groups_input_mixin = 1;
  const bool layer1x1_active = false;
  const int layer1x1_groups = 1;

  nam::wavenet::Layer1x1Params layer1x1_params(layer1x1_active, layer1x1_groups);
  nam::wavenet::Head1x1Params head1x1_params(false, channels, 1);
  auto film_params = make_default_film_params();

  nam::wavenet::LayerParams layer_params(conditionSize, channels, bottleneck, kernelSize, dilation, activation,
                                         gating_mode, groups_input, groups_input_mixin, layer1x1_params, head1x1_params,
                                         nam::activations::ActivationConfig{}, film_params, film_params, film_params,
                                         film_params, film_params, film_params, film_params, film_params);

  // This should throw an exception at construction time
  bool threw_exception = false;
  try
  {
    auto layer = nam::wavenet::_Layer(layer_params);
  }
  catch (const std::invalid_argument& e)
  {
    threw_exception = true;
    // Verify the error message mentions bottleneck and channels
    std::string error_msg = e.what();
    assert(error_msg.find("bottleneck") != std::string::npos);
    assert(error_msg.find("channels") != std::string::npos);
  }
  assert(threw_exception);
}

void test_layer1x1_post_film_active()
{
  // Test that layer1x1_post_film works when layer1x1 is active
  const int conditionSize = 1;
  const int channels = 2;
  const int bottleneck = channels;
  const int kernelSize = 1;
  const int dilation = 1;
  const auto activation = nam::activations::ActivationConfig::simple(nam::activations::ActivationType::ReLU);
  const nam::wavenet::GatingMode gating_mode = nam::wavenet::GatingMode::NONE;
  const int groups_input = 1;
  const int groups_input_mixin = 1;
  const bool layer1x1_active = true;
  const int layer1x1_groups = 1;

  nam::wavenet::Layer1x1Params layer1x1_params(layer1x1_active, layer1x1_groups);
  nam::wavenet::Head1x1Params head1x1_params(false, channels, 1);
  auto film_params = make_default_film_params();
  nam::wavenet::_FiLMParams layer1x1_post_film_params(true, true, 1); // Active FiLM

  nam::wavenet::LayerParams layer_params(conditionSize, channels, bottleneck, kernelSize, dilation, activation,
                                         gating_mode, groups_input, groups_input_mixin, layer1x1_params, head1x1_params,
                                         nam::activations::ActivationConfig{}, film_params, film_params, film_params,
                                         film_params, film_params, film_params, layer1x1_post_film_params, film_params);

  auto layer = nam::wavenet::_Layer(layer_params);

  // Set weights: conv, input_mixin, layer1x1, layer1x1_post_film
  // With bottleneck=channels=2:
  // Conv: (channels, bottleneck, kernelSize) + bias = (2, 2, 1) + 2 = 4 + 2 = 6 weights
  // Input mixin: (conditionSize, bottleneck) = (1, 2) = 2 weights
  // layer1x1: (bottleneck, channels) + bias = (2, 2) + 2 = 4 + 2 = 6 weights
  // layer1x1_post_film: (conditionSize, 2*channels) + bias = (1, 4) + 4 = 4 + 4 = 8 weights (with shift)
  std::vector<float> weights{
    // Conv: weights=1.0, bias=0.0 (identity)
    1.0f, 0.0f, 0.0f, 1.0f, // weights
    0.0f, 0.0f, // bias
    // Input mixin: weights=1.0
    1.0f, 1.0f,
    // layer1x1: weights=1.0, bias=0.0 (identity)
    1.0f, 0.0f, 0.0f, 1.0f, // weights
    0.0f, 0.0f, // bias
    // layer1x1_post_film: (conditionSize, 2*channels) + bias (with shift)
    1.0f, 1.0f, // scale weights
    0.0f, 0.0f, // shift weights
    0.0f, 0.0f, 0.0f, 0.0f // bias
  };

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

  // Verify outputs are reasonable (not NaN, not infinite)
  for (int i = 0; i < numFrames; i++)
  {
    assert(!std::isnan(layer_output(0, i)));
    assert(!std::isinf(layer_output(0, i)));
    assert(!std::isnan(layer_output(1, i)));
    assert(!std::isinf(layer_output(1, i)));
  }
}

void test_layer1x1_post_film_inactive_with_layer1x1_inactive()
{
  // Test that layer1x1_post_film cannot be active when layer1x1 is inactive
  const int conditionSize = 1;
  const int channels = 2;
  const int bottleneck = channels;
  const int kernelSize = 1;
  const int dilation = 1;
  const auto activation = nam::activations::ActivationConfig::simple(nam::activations::ActivationType::ReLU);
  const nam::wavenet::GatingMode gating_mode = nam::wavenet::GatingMode::NONE;
  const int groups_input = 1;
  const int groups_input_mixin = 1;
  const bool layer1x1_active = false;
  const int layer1x1_groups = 1;

  nam::wavenet::Layer1x1Params layer1x1_params(layer1x1_active, layer1x1_groups);
  nam::wavenet::Head1x1Params head1x1_params(false, channels, 1);
  auto film_params = make_default_film_params();
  nam::wavenet::_FiLMParams layer1x1_post_film_params(true, true, 1); // Active FiLM - should fail

  nam::wavenet::LayerParams layer_params(conditionSize, channels, bottleneck, kernelSize, dilation, activation,
                                         gating_mode, groups_input, groups_input_mixin, layer1x1_params, head1x1_params,
                                         nam::activations::ActivationConfig{}, film_params, film_params, film_params,
                                         film_params, film_params, film_params, layer1x1_post_film_params, film_params);

  // This should throw an exception
  bool threw_exception = false;
  try
  {
    auto layer = nam::wavenet::_Layer(layer_params);
  }
  catch (const std::invalid_argument& e)
  {
    threw_exception = true;
    // Verify the error message mentions layer1x1_post_film
    std::string error_msg = e.what();
    assert(error_msg.find("layer1x1_post_film") != std::string::npos);
  }
  assert(threw_exception);
}

void test_layer1x1_gated()
{
  // Test layer1x1 with gated activation
  const int conditionSize = 1;
  const int channels = 2;
  const int bottleneck = channels;
  const int kernelSize = 1;
  const int dilation = 1;
  const auto activation = nam::activations::ActivationConfig::simple(nam::activations::ActivationType::ReLU);
  const nam::wavenet::GatingMode gating_mode = nam::wavenet::GatingMode::GATED;
  const int groups_input = 1;
  const int groups_input_mixin = 1;
  const bool layer1x1_active = true;
  const int layer1x1_groups = 1;

  nam::wavenet::Layer1x1Params layer1x1_params(layer1x1_active, layer1x1_groups);
  nam::wavenet::Head1x1Params head1x1_params(false, channels, 1);
  auto sigmoid_config = nam::activations::ActivationConfig::simple(nam::activations::ActivationType::Sigmoid);
  auto layer = make_layer(conditionSize, channels, bottleneck, kernelSize, dilation, activation, gating_mode,
                          groups_input, groups_input_mixin, layer1x1_params, head1x1_params, sigmoid_config);

  // With gated: conv outputs 2*bottleneck, input_mixin outputs 2*bottleneck, layer1x1 outputs channels
  // With gated=true, bottleneck=channels=2:
  // Conv: (channels, 2*bottleneck, kernelSize) + bias = (2, 4, 1) + 4 = 8 + 4 = 12 weights
  // Input mixin: (conditionSize, 2*bottleneck) = (1, 4) = 4 weights
  // layer1x1: (bottleneck, channels) + bias = (2, 2) + 2 = 4 + 2 = 6 weights
  std::vector<float> weights;
  // Conv weights: (2, 4, 1) + bias(4)
  weights.push_back(1.0f);
  weights.push_back(0.0f);
  weights.push_back(0.0f);
  weights.push_back(1.0f);
  weights.push_back(1.0f);
  weights.push_back(0.0f);
  weights.push_back(0.0f);
  weights.push_back(1.0f);
  weights.push_back(0.0f);
  weights.push_back(0.0f);
  weights.push_back(0.0f);
  weights.push_back(0.0f);
  // Input mixin: (1, 4)
  weights.push_back(1.0f);
  weights.push_back(1.0f);
  weights.push_back(1.0f);
  weights.push_back(1.0f);
  // layer1x1: (2, 2) + bias(2)
  weights.push_back(1.0f);
  weights.push_back(0.0f);
  weights.push_back(0.0f);
  weights.push_back(1.0f);
  weights.push_back(0.0f);
  weights.push_back(0.0f);

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

  // Verify outputs are reasonable
  for (int i = 0; i < numFrames; i++)
  {
    assert(!std::isnan(layer_output(0, i)));
    assert(!std::isinf(layer_output(0, i)));
    assert(!std::isnan(layer_output(1, i)));
    assert(!std::isinf(layer_output(1, i)));
  }
}

void test_layer1x1_groups()
{
  // Test layer1x1 with groups
  const int conditionSize = 1;
  const int channels = 4;
  const int bottleneck = channels;
  const int kernelSize = 1;
  const int dilation = 1;
  const auto activation = nam::activations::ActivationConfig::simple(nam::activations::ActivationType::ReLU);
  const nam::wavenet::GatingMode gating_mode = nam::wavenet::GatingMode::NONE;
  const int groups_input = 1;
  const int groups_input_mixin = 1;
  const bool layer1x1_active = true;
  const int layer1x1_groups = 2; // Grouped layer1x1

  nam::wavenet::Layer1x1Params layer1x1_params(layer1x1_active, layer1x1_groups);
  nam::wavenet::Head1x1Params head1x1_params(false, channels, 1);
  auto layer =
    make_layer(conditionSize, channels, bottleneck, kernelSize, dilation, activation, gating_mode, groups_input,
               groups_input_mixin, layer1x1_params, head1x1_params, nam::activations::ActivationConfig{});

  // With grouped layer1x1, we need to provide weights for each group
  // For groups=2, channels=4, bottleneck=4: each group has 2 in_channels and 2 out_channels
  // With bottleneck=channels=4:
  // Conv: (channels, bottleneck, kernelSize) + bias = (4, 4, 1) + 4 = 16 + 4 = 20 weights
  // Input mixin: (conditionSize, bottleneck) = (1, 4) = 4 weights
  // layer1x1: grouped with groups=2, (bottleneck, channels) + bias = (4, 4) + 4 = 16 + 4 = 20 weights
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
    // layer1x1: for each group, (out_channels_per_group, in_channels_per_group) + bias_per_group
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

  // Verify outputs are reasonable
  for (int i = 0; i < numFrames; i++)
  {
    for (int c = 0; c < channels; c++)
    {
      assert(!std::isnan(layer_output(c, i)));
      assert(!std::isinf(layer_output(c, i)));
    }
  }
}

}; // namespace test_layer1x1
} // namespace test_wavenet
