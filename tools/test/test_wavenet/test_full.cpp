// Tests for full WaveNet model

#include <Eigen/Dense>
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

#include "NAM/wavenet.h"

namespace test_wavenet
{
namespace test_full
{
// Helper function to create default (inactive) FiLM parameters
static nam::wavenet::_FiLMParams make_default_film_params()
{
  return nam::wavenet::_FiLMParams(false, false);
}

// Helper function to create LayerArrayParams with default FiLM parameters
static nam::wavenet::LayerArrayParams make_layer_array_params(
  const int input_size, const int condition_size, const int head_size, const int channels, const int bottleneck,
  const int kernel_size, std::vector<int>&& dilations, const nam::activations::ActivationConfig& activation_config,
  const nam::wavenet::GatingMode gating_mode, const bool head_bias, const int groups_input,
  const int groups_input_mixin, const int groups_1x1, const nam::wavenet::Head1x1Params& head1x1_params,
  const nam::activations::ActivationConfig& secondary_activation_config)
{
  auto film_params = make_default_film_params();
  // Duplicate activation_config for each layer (based on dilations size)
  std::vector<nam::activations::ActivationConfig> activation_configs(dilations.size(), activation_config);
  return nam::wavenet::LayerArrayParams(input_size, condition_size, head_size, channels, bottleneck, kernel_size,
                                        std::move(dilations), std::move(activation_configs), gating_mode, head_bias,
                                        groups_input, groups_input_mixin, groups_1x1, head1x1_params,
                                        secondary_activation_config, film_params, film_params, film_params, film_params,
                                        film_params, film_params, film_params, film_params);
}
// Test full WaveNet model
void test_wavenet_model()
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
  const float head_scale = 1.0f;
  const bool with_head = false;
  const int groups = 1;
  const int groups_input_mixin = 1;
  const int groups_1x1 = 1;
  const bool head1x1_active = false;

  nam::wavenet::Head1x1Params head1x1_params(head1x1_active, channels, 1);
  nam::activations::ActivationConfig empty_config{};
  nam::wavenet::LayerArrayParams params = make_layer_array_params(
    input_size, condition_size, head_size, channels, bottleneck, kernel_size, std::move(dilations), activation,
    gating_mode, head_bias, groups, groups_input_mixin, groups_1x1, head1x1_params, empty_config);
  std::vector<nam::wavenet::LayerArrayParams> layer_array_params;
  layer_array_params.push_back(std::move(params));

  // Calculate weights needed
  // Layer array 0:
  //   Rechannel: (1,1) weight
  //   Layer 0: conv (1,1,1) + bias, input_mixin (1,1), 1x1 (1,1) + bias
  //   Head rechannel: (1,1) weight
  // Head scale: 1 float
  std::vector<float> weights;
  weights.push_back(1.0f); // Rechannel
  weights.insert(weights.end(), {1.0f, 0.0f, 1.0f, 1.0f, 0.0f}); // Layer 0
  weights.push_back(1.0f); // Head rechannel
  weights.push_back(head_scale); // Head scale

  std::unique_ptr<nam::wavenet::WaveNet> condition_dsp = nullptr;
  auto wavenet = std::make_unique<nam::wavenet::WaveNet>(
    input_size, layer_array_params, head_scale, with_head, weights, std::move(condition_dsp), 48000.0);

  const int numFrames = 4;
  const int maxBufferSize = 64;
  wavenet->Reset(48000.0, maxBufferSize);

  std::vector<NAM_SAMPLE> input(numFrames, 1.0f);
  std::vector<NAM_SAMPLE> output(numFrames, 0.0f);
  NAM_SAMPLE* inputPtrs[] = {input.data()};
  NAM_SAMPLE* outputPtrs[] = {output.data()};

  wavenet->process(inputPtrs, outputPtrs, numFrames);

  // Verify output dimensions
  assert(output.size() == numFrames);
  // Output should be non-zero
  for (int i = 0; i < numFrames; i++)
  {
    assert(std::isfinite(output[i]));
  }
}

// Test WaveNet with multiple layer arrays
void test_wavenet_multiple_arrays()
{
  const int input_size = 1;
  const int condition_size = 1;
  const int head_size = 1;
  const int channels = 1;
  const int kernel_size = 1;
  std::vector<int> dilations{1};
  const auto activation = nam::activations::ActivationConfig::simple(nam::activations::ActivationType::ReLU);
  const nam::wavenet::GatingMode gating_mode = nam::wavenet::GatingMode::NONE;
  const bool head_bias = false;
  const float head_scale = 0.5f;
  const bool with_head = false;
  const int groups = 1;
  const int groups_input_mixin = 1;

  std::vector<nam::wavenet::LayerArrayParams> layer_array_params;
  // First array
  std::vector<int> dilations1{1};
  const int bottleneck = channels;
  const int groups_1x1 = 1;
  const bool head1x1_active = false;

  nam::wavenet::Head1x1Params head1x1_params(head1x1_active, channels, 1);
  layer_array_params.push_back(make_layer_array_params(input_size, condition_size, head_size, channels, bottleneck,
                                                       kernel_size, std::move(dilations1), activation, gating_mode,
                                                       head_bias, groups, groups_input_mixin, groups_1x1,
                                                       head1x1_params, nam::activations::ActivationConfig{}));
  // Second array (head_size of first must match channels of second)
  std::vector<int> dilations2{1};
  layer_array_params.push_back(make_layer_array_params(head_size, condition_size, head_size, channels, bottleneck,
                                                       kernel_size, std::move(dilations2), activation, gating_mode,
                                                       head_bias, groups, groups_input_mixin, groups_1x1,
                                                       head1x1_params, nam::activations::ActivationConfig{}));

  std::vector<float> weights;
  // Array 0: rechannel, layer, head_rechannel
  weights.insert(weights.end(), {1.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f});
  // Array 1: rechannel, layer, head_rechannel
  weights.insert(weights.end(), {1.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f});
  weights.push_back(head_scale);

  std::unique_ptr<nam::wavenet::WaveNet> condition_dsp = nullptr;
  auto wavenet = std::make_unique<nam::wavenet::WaveNet>(
    input_size, layer_array_params, head_scale, with_head, weights, std::move(condition_dsp), 48000.0);

  const int numFrames = 4;
  const int maxBufferSize = 64;
  wavenet->Reset(48000.0, maxBufferSize);

  std::vector<NAM_SAMPLE> input(numFrames, 1.0f);
  std::vector<NAM_SAMPLE> output(numFrames, 0.0f);
  NAM_SAMPLE* inputPtrs[] = {input.data()};
  NAM_SAMPLE* outputPtrs[] = {output.data()};

  wavenet->process(inputPtrs, outputPtrs, numFrames);

  assert(output.size() == numFrames);
  for (int i = 0; i < numFrames; i++)
  {
    assert(std::isfinite(output[i]));
  }
}

// Test WaveNet with zero input
void test_wavenet_zero_input()
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
  const float head_scale = 1.0f;
  const bool with_head = false;
  const int groups = 1;
  const int groups_input_mixin = 1;
  const int groups_1x1 = 1;
  const bool head1x1_active = false;
  nam::wavenet::Head1x1Params head1x1_params(head1x1_active, channels, 1);

  nam::wavenet::LayerArrayParams params =
    make_layer_array_params(input_size, condition_size, head_size, channels, bottleneck, kernel_size,
                            std::move(dilations), activation, gating_mode, head_bias, groups, groups_input_mixin,
                            groups_1x1, head1x1_params, nam::activations::ActivationConfig{});
  std::vector<nam::wavenet::LayerArrayParams> layer_array_params;
  layer_array_params.push_back(std::move(params));

  std::vector<float> weights{1.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, head_scale};

  std::unique_ptr<nam::wavenet::WaveNet> condition_dsp = nullptr;
  auto wavenet = std::make_unique<nam::wavenet::WaveNet>(
    input_size, layer_array_params, head_scale, with_head, weights, std::move(condition_dsp), 48000.0);

  const int numFrames = 4;
  wavenet->Reset(48000.0, numFrames);

  std::vector<NAM_SAMPLE> input(numFrames, 0.0f);
  std::vector<NAM_SAMPLE> output(numFrames, 0.0f);
  NAM_SAMPLE* inputPtrs[] = {input.data()};
  NAM_SAMPLE* outputPtrs[] = {output.data()};

  wavenet->process(inputPtrs, outputPtrs, numFrames);

  // With zero input, output should be finite (may be zero or non-zero depending on bias)
  for (int i = 0; i < numFrames; i++)
  {
    assert(std::isfinite(output[i]));
  }
}

// Test WaveNet with different buffer sizes
void test_wavenet_different_buffer_sizes()
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
  const float head_scale = 1.0f;
  const bool with_head = false;
  const int groups = 1;
  const int groups_input_mixin = 1;
  const int groups_1x1 = 1;
  const bool head1x1_active = false;
  nam::wavenet::Head1x1Params head1x1_params(head1x1_active, channels, 1);

  nam::wavenet::LayerArrayParams params =
    make_layer_array_params(input_size, condition_size, head_size, channels, bottleneck, kernel_size,
                            std::move(dilations), activation, gating_mode, head_bias, groups, groups_input_mixin,
                            groups_1x1, head1x1_params, nam::activations::ActivationConfig{});
  std::vector<nam::wavenet::LayerArrayParams> layer_array_params;
  layer_array_params.push_back(std::move(params));

  std::vector<float> weights{1.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, head_scale};

  std::unique_ptr<nam::wavenet::WaveNet> condition_dsp = nullptr;
  auto wavenet = std::make_unique<nam::wavenet::WaveNet>(
    input_size, layer_array_params, head_scale, with_head, weights, std::move(condition_dsp), 48000.0);

  // Test with different buffer sizes
  wavenet->Reset(48000.0, 64);
  std::vector<NAM_SAMPLE> input1(32, 1.0f);
  std::vector<NAM_SAMPLE> output1(32, 0.0f);
  NAM_SAMPLE* inputPtrs1[] = {input1.data()};
  NAM_SAMPLE* outputPtrs1[] = {output1.data()};
  wavenet->process(inputPtrs1, outputPtrs1, 32);

  wavenet->Reset(48000.0, 128);
  std::vector<NAM_SAMPLE> input2(64, 1.0f);
  std::vector<NAM_SAMPLE> output2(64, 0.0f);
  NAM_SAMPLE* inputPtrs2[] = {input2.data()};
  NAM_SAMPLE* outputPtrs2[] = {output2.data()};
  wavenet->process(inputPtrs2, outputPtrs2, 64);

  // Both should work without errors
  assert(output1.size() == 32);
  assert(output2.size() == 64);
}

// Test WaveNet prewarm functionality
void test_wavenet_prewarm()
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
  const float head_scale = 1.0f;
  const bool with_head = false;
  const int groups = 1;
  const int groups_input_mixin = 1;
  const int groups_1x1 = 1;
  const bool head1x1_active = false;

  nam::wavenet::Head1x1Params head1x1_params(head1x1_active, channels, 1);

  nam::wavenet::LayerArrayParams params =
    make_layer_array_params(input_size, condition_size, head_size, channels, bottleneck, kernel_size,
                            std::move(dilations), activation, gating_mode, head_bias, groups, groups_input_mixin,
                            groups_1x1, head1x1_params, nam::activations::ActivationConfig{});
  std::vector<nam::wavenet::LayerArrayParams> layer_array_params;
  layer_array_params.push_back(std::move(params));

  std::vector<float> weights;
  // Rechannel: (1,1) weight, no bias
  weights.push_back(1.0f);
  // 3 layers: each needs:
  //   Conv: kernel_size=3, in_channels=1, out_channels=1, bias=true -> 3*1*1 + 1 = 4 weights
  //   Input mixin: condition_size=1, out_channels=1, no bias -> 1 weight
  //   1x1: in_channels=1, out_channels=1, bias=true -> 1*1 + 1 = 2 weights
  //   Total per layer: 7 weights
  for (int i = 0; i < 3; i++)
  {
    // Conv weights: 3 weights (kernel_size * in_channels * out_channels) + 1 bias
    weights.insert(weights.end(), {1.0f, 1.0f, 1.0f, 0.0f});
    // Input mixin: 1 weight
    weights.push_back(1.0f);
    // 1x1: 1 weight + 1 bias
    weights.insert(weights.end(), {1.0f, 0.0f});
  }
  // Head rechannel: (1,1) weight, no bias
  weights.push_back(1.0f);
  weights.push_back(head_scale);

  std::unique_ptr<nam::wavenet::WaveNet> condition_dsp = nullptr;
  auto wavenet = std::make_unique<nam::wavenet::WaveNet>(
    input_size, layer_array_params, head_scale, with_head, weights, std::move(condition_dsp), 48000.0);

  // Test that prewarm can be called without errors
  wavenet->Reset(48000.0, 64);
  wavenet->prewarm();

  // After prewarm, processing should work
  const int numFrames = 4;
  std::vector<NAM_SAMPLE> input(numFrames, 1.0f);
  std::vector<NAM_SAMPLE> output(numFrames, 0.0f);
  NAM_SAMPLE* inputPtrs[] = {input.data()};
  NAM_SAMPLE* outputPtrs[] = {output.data()};
  wavenet->process(inputPtrs, outputPtrs, numFrames);

  // Output should be finite
  for (int i = 0; i < numFrames; i++)
  {
    assert(std::isfinite(output[i]));
  }
}
}; // namespace test_full

} // namespace test_wavenet
