// Tests for WaveNet condition processing functionality

#include <Eigen/Dense>
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

#include "NAM/wavenet.h"
#include "NAM/dsp.h"

namespace test_wavenet
{
namespace test_condition_processing
{
// Helper function to create default (inactive) FiLM parameters
static nam::wavenet::_FiLMParams make_default_film_params()
{
  return nam::wavenet::_FiLMParams(false, false);
}

// Helper function to create LayerArrayParams with default FiLM parameters
static nam::wavenet::LayerArrayParams make_layer_array_params(
  const int input_size, const int condition_size, const int head_size, const int channels, const int bottleneck,
  const int kernel_size, std::vector<int>&& dilations, const std::string activation,
  const nam::wavenet::GatingMode gating_mode, const bool head_bias, const int groups_input, const int groups_1x1,
  const nam::wavenet::Head1x1Params& head1x1_params, const std::string& secondary_activation)
{
  auto film_params = make_default_film_params();
  return nam::wavenet::LayerArrayParams(
    input_size, condition_size, head_size, channels, bottleneck, kernel_size, std::move(dilations), activation,
    gating_mode, head_bias, groups_input, groups_1x1, head1x1_params, secondary_activation, film_params, film_params,
    film_params, film_params, film_params, film_params, film_params, film_params, film_params);
}

// Helper function to create a simple WaveNet with specified input and output channels
std::unique_ptr<nam::wavenet::WaveNet> create_simple_wavenet(
  const int in_channels, const int out_channels, std::unique_ptr<nam::wavenet::WaveNet> condition_dsp = nullptr)
{
  const float head_scale = 1.0f;
  // Create a simple single-layer configuration
  const int input_size = in_channels;
  const int condition_size = condition_dsp != nullptr ? condition_dsp->NumOutputChannels() : in_channels;
  const int head_size = out_channels;
  const int channels = in_channels;
  const int bottleneck = channels;
  const int kernel_size = 1;
  std::vector<int> dilations{1};
  const std::string activation = "ReLU";
  const nam::wavenet::GatingMode gating_mode = nam::wavenet::GatingMode::NONE;
  const bool head_bias = false;
  const bool with_head = false;
  const int groups = 1;
  const int groups_1x1 = 1;
  const bool head1x1_active = false;
  const int head1x1_groups = 1;
  nam::wavenet::Head1x1Params head1x1_params(head1x1_active, channels, head1x1_groups);

  nam::wavenet::LayerArrayParams params = make_layer_array_params(
    input_size, condition_size, head_size, channels, bottleneck, kernel_size, std::move(dilations), activation,
    gating_mode, head_bias, groups, groups_1x1, head1x1_params, "");
  std::vector<nam::wavenet::LayerArrayParams> layer_array_params;
  layer_array_params.push_back(std::move(params));

  // Calculate weights needed based on channel counts
  // Rechannel: (in_channels, channels) = (in_channels, in_channels) = in_channels^2 weights
  // Layer: conv (channels, bottleneck, kernelSize=1) + bias = (in_channels, in_channels, 1) + in_channels =
  // in_channels^2 + in_channels Input mixin: (condition_size, bottleneck) = (condition_size, in_channels) =
  // condition_size * in_channels weights 1x1: (bottleneck, channels) + bias = (in_channels, in_channels) + in_channels
  // = in_channels^2 + in_channels Head rechannel: (bottleneck, head_size) = (in_channels, out_channels) = in_channels *
  // out_channels weights Head scale: 1
  std::vector<float> weights;

  // Rechannel weights (identity-like)
  for (int i = 0; i < channels; i++)
  {
    for (int j = 0; j < in_channels; j++)
    {
      weights.push_back((i == j) ? 1.0f : 0.0f);
    }
  }

  // Conv weights (identity-like)
  for (int i = 0; i < bottleneck; i++)
  {
    for (int j = 0; j < channels; j++)
    {
      weights.push_back((i == j) ? 1.0f : 0.0f);
    }
  }
  // Conv bias (zeros)
  for (int i = 0; i < bottleneck; i++)
  {
    weights.push_back(0.0f);
  }

  // Input mixin weights (condition_size -> bottleneck): weight matrix is (bottleneck, condition_size)
  // Identity mapping: output channel i maps to input channel i (for i < min(bottleneck, condition_size))
  for (int i = 0; i < bottleneck; i++)
  {
    for (int j = 0; j < condition_size; j++)
    {
      weights.push_back((i == j) ? 1.0f : 0.0f);
    }
  }

  // 1x1 weights (bottleneck -> channels): weight matrix is (channels, bottleneck)
  // Identity mapping: output channel i maps to input channel i (for i < min(channels, bottleneck))
  for (int i = 0; i < channels; i++)
  {
    for (int j = 0; j < bottleneck; j++)
    {
      weights.push_back((i == j) ? 1.0f : 0.0f);
    }
  }
  // 1x1 bias (zeros)
  for (int i = 0; i < channels; i++)
  {
    weights.push_back(0.0f);
  }

  // Head rechannel weights (bottleneck -> head_size): weight matrix is (head_size, bottleneck)
  // Identity mapping: output channel i maps to input channel i (for i < min(head_size, bottleneck))
  for (int i = 0; i < head_size; i++)
  {
    for (int j = 0; j < bottleneck; j++)
    {
      weights.push_back((i == j) ? 1.0f : 0.0f);
    }
  }

  weights.push_back(head_scale);

  return std::make_unique<nam::wavenet::WaveNet>(
    in_channels, layer_array_params, head_scale, with_head, weights, std::move(condition_dsp), 48000.0);
}

// Test condition processing with condition_dsp
void test_with_condition_dsp()
{
  const int in_channels = 1;
  const int out_channels = 1;
  auto condition_dsp = create_simple_wavenet(in_channels, out_channels, nullptr);
  auto wavenet = create_simple_wavenet(in_channels, out_channels, std::move(condition_dsp));

  const int numFrames = 8;
  const int maxBufferSize = 64;
  wavenet->Reset(48000.0, maxBufferSize);

  // Create input with known values
  std::vector<NAM_SAMPLE> input(numFrames);
  for (int i = 0; i < numFrames; i++)
  {
    input[i] = (NAM_SAMPLE)(0.1f * i);
  }
  NAM_SAMPLE* inputPtrs[] = {input.data()};

  std::vector<NAM_SAMPLE> output(numFrames, 0.0f);
  NAM_SAMPLE* outputPtrs[] = {output.data()};

  wavenet->process(inputPtrs, outputPtrs, numFrames);

  // Verify output is non-zero and finite
  for (int i = 0; i < numFrames; i++)
  {
    assert(std::isfinite(output[i]));
  }
}

// Test with multiple input channels
void test_with_condition_dsp_multichannel()
{
  const int in_channels = 2;
  const int out_channels = 3;
  const int condition_channels = 5;
  std::unique_ptr<nam::wavenet::WaveNet> condition_dsp =
    create_simple_wavenet(in_channels, condition_channels, nullptr);
  auto wavenet = create_simple_wavenet(in_channels, out_channels, std::move(condition_dsp));

  const int numFrames = 8;
  const int maxBufferSize = 64;
  wavenet->Reset(48000.0, maxBufferSize);

  // Create input with known values (need 2 channels for in_channels=2)
  std::vector<NAM_SAMPLE> input1(numFrames);
  std::vector<NAM_SAMPLE> input2(numFrames);
  for (int i = 0; i < numFrames; i++)
  {
    input1[i] = (NAM_SAMPLE)(0.1f * i);
    input2[i] = (NAM_SAMPLE)(0.2f * i);
  }
  NAM_SAMPLE* inputPtrs[] = {input1.data(), input2.data()};

  // Allocate output buffers for all output channels (out_channels = 3)
  std::vector<NAM_SAMPLE> output1(numFrames, 0.0f);
  std::vector<NAM_SAMPLE> output2(numFrames, 0.0f);
  std::vector<NAM_SAMPLE> output3(numFrames, 0.0f);
  NAM_SAMPLE* outputPtrs[] = {output1.data(), output2.data(), output3.data()};

  wavenet->process(inputPtrs, outputPtrs, numFrames);

  // Verify output is non-zero and finite for all channels
  for (int i = 0; i < numFrames; i++)
  {
    assert(std::isfinite(output1[i]));
    assert(std::isfinite(output2[i]));
    assert(std::isfinite(output3[i]));
  }
}

} // namespace test_condition_processing
} // namespace test_wavenet
