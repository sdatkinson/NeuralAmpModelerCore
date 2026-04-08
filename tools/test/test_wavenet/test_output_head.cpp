// Tests for WaveNet post-stack head (Python ``Head`` module)

#include <Eigen/Dense>
#include <cassert>
#include <cmath>
#include <iostream>
#include <optional>
#include <vector>

#include "NAM/wavenet.h"

namespace test_wavenet
{
namespace test_output_head
{

static nam::wavenet::_FiLMParams make_inactive_film()
{
  return nam::wavenet::_FiLMParams(false, false);
}

static nam::wavenet::LayerArrayParams make_layer_array_params(
  const int input_size, const int condition_size, const int head_size, const int channels, const int bottleneck,
  std::vector<int>&& kernel_sizes, std::vector<int>&& dilations,
  const nam::activations::ActivationConfig& activation_config, const nam::wavenet::GatingMode gating_mode,
  const bool head_bias, const int groups_input, const int groups_input_mixin,
  const nam::wavenet::Layer1x1Params& layer1x1_params, const nam::wavenet::Head1x1Params& head1x1_params,
  const nam::activations::ActivationConfig& secondary_activation_config)
{
  auto film = make_inactive_film();
  std::vector<nam::activations::ActivationConfig> activation_configs(dilations.size(), activation_config);
  std::vector<nam::wavenet::GatingMode> gating_modes(dilations.size(), gating_mode);
  std::vector<nam::activations::ActivationConfig> secondary_activation_configs(dilations.size(),
                                                                               secondary_activation_config);
  return nam::wavenet::LayerArrayParams(
    input_size, condition_size, head_size, channels, bottleneck, std::move(kernel_sizes), std::move(dilations),
    std::move(activation_configs), std::move(gating_modes), head_bias, groups_input, groups_input_mixin,
    layer1x1_params, head1x1_params, std::move(secondary_activation_configs), film, film, film, film, film, film, film,
    film);
}

void test_post_stack_head_receptive_field()
{
  nam::wavenet::WaveNetHeadParams p;
  p.in_channels = 2;
  p.channels = 3;
  p.out_channels = 1;
  p.kernel_sizes = {3, 5};
  p.activation_config = nam::activations::ActivationConfig::simple(nam::activations::ActivationType::Tanh);
  nam::wavenet::PostStackHead head(p);
  // Python: 1 + (3-1) + (5-1) = 7
  assert(head.receptive_field() == 7);
}

void test_wavenet_with_post_stack_head_processes()
{
  const int input_size = 1;
  const int condition_size = 1;
  const int head_size = 1;
  const int channels = 1;
  const int bottleneck = channels;
  const int kernel_size = 1;
  std::vector<int> dilations{1};
  std::vector<int> kernel_sizes(dilations.size(), kernel_size);
  const auto activation = nam::activations::ActivationConfig::simple(nam::activations::ActivationType::ReLU);
  const nam::wavenet::GatingMode gating_mode = nam::wavenet::GatingMode::NONE;
  const bool head_bias = false;
  const float head_scale = 0.5f;
  const bool with_head = true;
  const int groups = 1;
  const int groups_input_mixin = 1;
  nam::wavenet::Layer1x1Params layer1x1_params(true, 1);
  nam::wavenet::Head1x1Params head1x1_params(false, channels, 1);
  nam::activations::ActivationConfig empty_config{};
  nam::wavenet::LayerArrayParams layer_params = make_layer_array_params(
    input_size, condition_size, head_size, channels, bottleneck, std::move(kernel_sizes), std::move(dilations),
    activation, gating_mode, head_bias, groups, groups_input_mixin, layer1x1_params, head1x1_params, empty_config);
  std::vector<nam::wavenet::LayerArrayParams> layer_array_params;
  layer_array_params.push_back(std::move(layer_params));

  nam::wavenet::WaveNetHeadParams hp;
  hp.in_channels = 1;
  hp.channels = 1;
  hp.out_channels = 1;
  hp.kernel_sizes = {1};
  hp.activation_config = nam::activations::ActivationConfig::simple(nam::activations::ActivationType::Tanh);

  std::vector<float> weights;
  weights.push_back(1.0f); // Rechannel
  weights.insert(weights.end(), {1.0f, 0.0f, 1.0f, 1.0f, 0.0f}); // Layer 0
  weights.push_back(1.0f); // Head rechannel
  weights.push_back(1.0f); // Post-stack conv weight (1x1)
  weights.push_back(0.0f); // Post-stack conv bias
  weights.push_back(head_scale);

  std::unique_ptr<nam::wavenet::WaveNet> condition_dsp = nullptr;
  auto wavenet = std::make_unique<nam::wavenet::WaveNet>(input_size, layer_array_params, head_scale, with_head,
                                                         std::optional<nam::wavenet::WaveNetHeadParams>(std::move(hp)),
                                                         std::move(weights), std::move(condition_dsp), 48000.0);

  const int numFrames = 8;
  const int maxBufferSize = 64;
  wavenet->Reset(48000.0, maxBufferSize);
  wavenet->prewarm();

  std::vector<NAM_SAMPLE> input(numFrames, 0.1f);
  std::vector<NAM_SAMPLE> output(numFrames, 0.0f);
  NAM_SAMPLE* inputPtrs[] = {input.data()};
  NAM_SAMPLE* outputPtrs[] = {output.data()};

  wavenet->process(inputPtrs, outputPtrs, numFrames);

  for (int i = 0; i < numFrames; i++)
    assert(std::isfinite(output[i]));
}

void test_wavenet_with_two_layer_post_stack_head_applies_activation_per_layer_input()
{
  // Regression for multi-layer post-stack head execution:
  // each layer must apply its activation to that layer's input, not always the
  // original head input buffer.
  const int input_size = 1;
  const int condition_size = 1;
  const int head_size = 1;
  const int channels = 1;
  const int bottleneck = channels;
  const int kernel_size = 1;
  std::vector<int> dilations{1};
  std::vector<int> kernel_sizes(dilations.size(), kernel_size);
  const auto activation = nam::activations::ActivationConfig::simple(nam::activations::ActivationType::ReLU);
  const nam::wavenet::GatingMode gating_mode = nam::wavenet::GatingMode::NONE;
  const bool head_bias = false;
  const float head_scale = 1.0f;
  const bool with_head = true;
  const int groups = 1;
  const int groups_input_mixin = 1;
  nam::wavenet::Layer1x1Params layer1x1_params(true, 1);
  nam::wavenet::Head1x1Params head1x1_params(false, channels, 1);
  nam::activations::ActivationConfig empty_config{};
  nam::wavenet::LayerArrayParams layer_params = make_layer_array_params(
    input_size, condition_size, head_size, channels, bottleneck, std::move(kernel_sizes), std::move(dilations),
    activation, gating_mode, head_bias, groups, groups_input_mixin, layer1x1_params, head1x1_params, empty_config);
  std::vector<nam::wavenet::LayerArrayParams> layer_array_params;
  layer_array_params.push_back(std::move(layer_params));

  nam::wavenet::WaveNetHeadParams hp;
  hp.in_channels = 1;
  hp.channels = 1;
  hp.out_channels = 1;
  hp.kernel_sizes = {1, 1};
  hp.activation_config = nam::activations::ActivationConfig::simple(nam::activations::ActivationType::ReLU);

  std::vector<float> weights;
  // Main WaveNet (single 1x1 layer array), identity mapping to head input:
  weights.push_back(1.0f); // Rechannel weight
  weights.insert(weights.end(), {1.0f, 0.0f, 1.0f, 1.0f, 0.0f}); // Layer 0 weights
  weights.push_back(1.0f); // Head rechannel weight
  // Post-stack head (2x [ReLU -> Conv1d(k=1)]):
  // First conv: y = -1*x + 0
  // Second conv: y = 2*x + 0
  // For negative input, correct chain gives 0 (ReLU before second conv on first conv output).
  weights.push_back(-1.0f); // Head layer 0 conv weight
  weights.push_back(0.0f); // Head layer 0 conv bias
  weights.push_back(2.0f); // Head layer 1 conv weight
  weights.push_back(0.0f); // Head layer 1 conv bias
  weights.push_back(head_scale);

  std::unique_ptr<nam::wavenet::WaveNet> condition_dsp = nullptr;
  auto wavenet = std::make_unique<nam::wavenet::WaveNet>(input_size, layer_array_params, head_scale, with_head,
                                                         std::optional<nam::wavenet::WaveNetHeadParams>(std::move(hp)),
                                                         std::move(weights), std::move(condition_dsp), 48000.0);

  const int numFrames = 8;
  const int maxBufferSize = 64;
  wavenet->Reset(48000.0, maxBufferSize);
  wavenet->prewarm();

  std::vector<NAM_SAMPLE> input(numFrames, -0.25f);
  std::vector<NAM_SAMPLE> output(numFrames, 0.0f);
  NAM_SAMPLE* inputPtrs[] = {input.data()};
  NAM_SAMPLE* outputPtrs[] = {output.data()};
  wavenet->process(inputPtrs, outputPtrs, numFrames);

  for (int i = 0; i < numFrames; i++)
    assert(std::fabs(output[i]) < 1.0e-6f);
}

} // namespace test_output_head
} // namespace test_wavenet
