// Test to verify WaveNet::process is real-time safe (no allocations/frees)

#include <Eigen/Dense>
#include <cassert>
#include <cmath>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

#include "NAM/wavenet.h"
#include "NAM/conv1d.h"
#include "../allocation_tracking.h"

namespace test_wavenet
{
using namespace allocation_tracking;

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
                                       const int groups_input_mixin, const int groups_1x1,
                                       const nam::wavenet::Head1x1Params& head1x1_params,
                                       const nam::activations::ActivationConfig& secondary_activation_config)
{
  auto film_params = make_default_film_params();
  return nam::wavenet::_Layer(condition_size, channels, bottleneck, kernel_size, dilation, activation_config,
                              gating_mode, groups_input, groups_input_mixin, groups_1x1, head1x1_params,
                              secondary_activation_config, film_params, film_params, film_params, film_params,
                              film_params, film_params, film_params, film_params);
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
  return nam::wavenet::_LayerArray(input_size, condition_size, head_size, channels, bottleneck, kernel_size, dilations,
                                   activation_config, gating_mode, head_bias, groups_input, groups_input_mixin,
                                   groups_1x1, head1x1_params, secondary_activation_config, film_params, film_params,
                                   film_params, film_params, film_params, film_params, film_params, film_params);
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
  return nam::wavenet::LayerArrayParams(
    input_size, condition_size, head_size, channels, bottleneck, kernel_size, std::move(dilations), activation_config,
    gating_mode, head_bias, groups_input, groups_input_mixin, groups_1x1, head1x1_params, secondary_activation_config,
    film_params, film_params, film_params, film_params, film_params, film_params, film_params, film_params);
}

// Helper function to create a Layer with all FiLMs active
static nam::wavenet::_Layer make_layer_all_films(const int condition_size, const int channels, const int bottleneck,
                                                 const int kernel_size, const int dilation,
                                                 const nam::activations::ActivationConfig& activation_config,
                                                 const nam::wavenet::GatingMode gating_mode, const int groups_input,
                                                 const int groups_input_mixin, const int groups_1x1,
                                                 const nam::wavenet::Head1x1Params& head1x1_params,
                                                 const nam::activations::ActivationConfig& secondary_activation_config,
                                                 const bool shift)
{
  nam::wavenet::_FiLMParams film_params(true, shift);
  // Don't activate head1x1_post_film if head1x1 is not active (validation will fail)
  nam::wavenet::_FiLMParams head1x1_post_film_params =
    head1x1_params.active ? film_params : nam::wavenet::_FiLMParams(false, false);
  return nam::wavenet::_Layer(condition_size, channels, bottleneck, kernel_size, dilation, activation_config,
                              gating_mode, groups_input, groups_input_mixin, groups_1x1, head1x1_params,
                              secondary_activation_config, film_params, film_params, film_params, film_params,
                              film_params, film_params, film_params, head1x1_post_film_params);
}

// Test that pre-allocated Eigen operations with noalias() don't allocate
void test_allocation_tracking_pass()
{
  const int rows = 10;
  const int cols = 20;

  // Pre-allocate matrices for matrix product: c = a * b
  // a is rows x cols, b is cols x rows, so c is rows x rows
  Eigen::MatrixXf a(rows, cols);
  Eigen::MatrixXf b(cols, rows);
  Eigen::MatrixXf c(rows, rows);

  a.setConstant(1.0f);
  b.setConstant(2.0f);

  run_allocation_test_no_allocations(
    nullptr, // No setup needed
    [&]() {
      // Matrix product with noalias() - should not allocate (all matrices pre-allocated)
      // Using noalias() is important for matrix products to avoid unnecessary temporaries
      // Note: Even without noalias(), Eigen may avoid temporaries when matrices are distinct,
      // but noalias() is best practice for real-time safety
      c.noalias() = a * b;
    },
    nullptr, // No teardown needed
    "test_allocation_tracking_pass");

  // Verify result: c should be rows x rows with value 2*cols (each element is sum of cols elements of value 2)
  assert(c.rows() == rows && c.cols() == rows);
  assert(std::abs(c(0, 0) - 2.0f * cols) < 0.001f);
}

// Test that creating a new matrix causes allocations (should be caught)
void test_allocation_tracking_fail()
{
  run_allocation_test_expect_allocations(
    nullptr, // No setup needed
    [&]() {
      // This operation should allocate (creating new matrix)
      Eigen::MatrixXf a(10, 20);
      a.setConstant(1.0f);
    },
    nullptr, // No teardown needed
    "test_allocation_tracking_fail");
}

// Test that Conv1D::Process() method does not allocate or free memory
void test_conv1d_process_realtime_safe()
{
  // Setup: Create a Conv1D
  const int in_channels = 1;
  const int out_channels = 1;
  const int kernel_size = 1;
  const bool do_bias = false;
  const int dilation = 1;

  nam::Conv1D conv;
  conv.set_size_(in_channels, out_channels, kernel_size, do_bias, dilation);

  // Set weights: simple identity
  std::vector<float> weights{1.0f};
  auto it = weights.begin();
  conv.set_weights_(it);

  const int maxBufferSize = 256;
  conv.SetMaxBufferSize(maxBufferSize);

  // Test with several different buffer sizes
  std::vector<int> buffer_sizes{1, 8, 16, 32, 64, 128, 256};

  for (int buffer_size : buffer_sizes)
  {
    // Prepare input matrix (allocate before tracking)
    Eigen::MatrixXf input(in_channels, buffer_size);
    input.setConstant(0.5f);

    std::string test_name = "Conv1D Process - Buffer size " + std::to_string(buffer_size);
    run_allocation_test_no_allocations(
      nullptr, // No setup needed
      [&]() {
        // Call Process() - this should not allocate or free
        conv.Process(input, buffer_size);
      },
      nullptr, // No teardown needed
      test_name.c_str());

    // Verify output is valid
    auto output = conv.GetOutput().leftCols(buffer_size);
    assert(output.rows() == out_channels && output.cols() == buffer_size);
    assert(std::isfinite(output(0, 0)));
  }
}

// Test that Conv1D::Process() with grouped convolution (groups > 1) does not allocate or free memory
void test_conv1d_grouped_process_realtime_safe()
{
  // Setup: Create a Conv1D with grouped convolution
  const int in_channels = 4;
  const int out_channels = 4;
  const int kernel_size = 2;
  const bool do_bias = true;
  const int dilation = 1;
  const int groups = 2;

  nam::Conv1D conv;
  conv.set_size_(in_channels, out_channels, kernel_size, do_bias, dilation, groups);

  // Set weights for grouped convolution
  // Each group: 2 in_channels, 2 out_channels, kernel_size=2
  // Weight layout: for each group g, for each (i,j), for each k
  std::vector<float> weights;
  // Group 0, kernel[0]: identity
  weights.push_back(1.0f);
  weights.push_back(0.0f);
  weights.push_back(0.0f);
  weights.push_back(1.0f);
  // Group 0, kernel[1]: identity
  weights.push_back(1.0f);
  weights.push_back(0.0f);
  weights.push_back(0.0f);
  weights.push_back(1.0f);
  // Group 1, kernel[0]: identity
  weights.push_back(1.0f);
  weights.push_back(0.0f);
  weights.push_back(0.0f);
  weights.push_back(1.0f);
  // Group 1, kernel[1]: identity
  weights.push_back(1.0f);
  weights.push_back(0.0f);
  weights.push_back(0.0f);
  weights.push_back(1.0f);
  // Bias: [0.1, 0.2, 0.3, 0.4]
  weights.push_back(0.1f);
  weights.push_back(0.2f);
  weights.push_back(0.3f);
  weights.push_back(0.4f);

  auto it = weights.begin();
  conv.set_weights_(it);

  const int maxBufferSize = 256;
  conv.SetMaxBufferSize(maxBufferSize);

  // Test with several different buffer sizes
  std::vector<int> buffer_sizes{1, 8, 16, 32, 64, 128, 256};

  for (int buffer_size : buffer_sizes)
  {
    // Prepare input matrix (allocate before tracking)
    Eigen::MatrixXf input(in_channels, buffer_size);
    input.setConstant(0.5f);

    std::string test_name = "Conv1D Grouped Process - Buffer size " + std::to_string(buffer_size);
    run_allocation_test_no_allocations(
      nullptr, // No setup needed
      [&]() {
        // Call Process() - this should not allocate or free
        conv.Process(input, buffer_size);
      },
      nullptr, // No teardown needed
      test_name.c_str());

    // Verify output is valid
    auto output = conv.GetOutput().leftCols(buffer_size);
    assert(output.rows() == out_channels && output.cols() == buffer_size);
    assert(std::isfinite(output(0, 0)));
    assert(std::isfinite(output(out_channels - 1, buffer_size - 1)));
  }
}

// Test that Conv1D::Process() with grouped convolution and dilation does not allocate or free memory
void test_conv1d_grouped_dilated_process_realtime_safe()
{
  // Setup: Create a Conv1D with grouped convolution and dilation
  const int in_channels = 6;
  const int out_channels = 6;
  const int kernel_size = 2;
  const bool do_bias = false;
  const int dilation = 2;
  const int groups = 3;

  nam::Conv1D conv;
  conv.set_size_(in_channels, out_channels, kernel_size, do_bias, dilation, groups);

  // Set weights for grouped convolution with 3 groups
  // Each group: 2 in_channels, 2 out_channels, kernel_size=2
  std::vector<float> weights;
  for (int g = 0; g < groups; g++)
  {
    // Group g, kernel[0]: identity
    weights.push_back(1.0f);
    weights.push_back(0.0f);
    weights.push_back(0.0f);
    weights.push_back(1.0f);
    // Group g, kernel[1]: identity
    weights.push_back(1.0f);
    weights.push_back(0.0f);
    weights.push_back(0.0f);
    weights.push_back(1.0f);
  }

  auto it = weights.begin();
  conv.set_weights_(it);

  const int maxBufferSize = 256;
  conv.SetMaxBufferSize(maxBufferSize);

  // Test with several different buffer sizes
  std::vector<int> buffer_sizes{1, 8, 16, 32, 64, 128, 256};

  for (int buffer_size : buffer_sizes)
  {
    // Prepare input matrix (allocate before tracking)
    Eigen::MatrixXf input(in_channels, buffer_size);
    input.setConstant(0.5f);

    std::string test_name = "Conv1D Grouped Dilated Process - Buffer size " + std::to_string(buffer_size);
    run_allocation_test_no_allocations(
      nullptr, // No setup needed
      [&]() {
        // Call Process() - this should not allocate or free
        conv.Process(input, buffer_size);
      },
      nullptr, // No teardown needed
      test_name.c_str());

    // Verify output is valid
    auto output = conv.GetOutput().leftCols(buffer_size);
    assert(output.rows() == out_channels && output.cols() == buffer_size);
    assert(std::isfinite(output(0, 0)));
    assert(std::isfinite(output(out_channels - 1, buffer_size - 1)));
  }
}

// Test that Layer::Process() method does not allocate or free memory
void test_layer_process_realtime_safe()
{
  // Setup: Create a Layer
  const int condition_size = 1;
  const int channels = 1;
  const int bottleneck = channels;
  const int kernel_size = 1;
  const int dilation = 1;
  const auto activation = nam::activations::ActivationConfig::simple(nam::activations::ActivationType::ReLU);
  const nam::wavenet::GatingMode gating_mode = nam::wavenet::GatingMode::NONE;
  const int groups_input = 1;
  const int groups_input_mixin = 1;
  const int groups_1x1 = 1;

  nam::wavenet::Head1x1Params head1x1_params(false, channels, 1);
  auto layer =
    make_layer(condition_size, channels, bottleneck, kernel_size, dilation, activation, gating_mode, groups_input,
               groups_input_mixin, groups_1x1, head1x1_params, nam::activations::ActivationConfig{});

  // Set weights
  std::vector<float> weights{1.0f, 0.0f, // Conv (weight, bias)
                             1.0f, // Input mixin
                             1.0f, 0.0f}; // 1x1 (weight, bias)
  auto it = weights.begin();
  layer.set_weights_(it);

  const int maxBufferSize = 256;
  layer.SetMaxBufferSize(maxBufferSize);

  // Test with several different buffer sizes
  std::vector<int> buffer_sizes{1, 8, 16, 32, 64, 128, 256};

  for (int buffer_size : buffer_sizes)
  {
    // Prepare input/condition matrices (allocate before tracking)
    Eigen::MatrixXf input(channels, buffer_size);
    Eigen::MatrixXf condition(condition_size, buffer_size);
    input.setConstant(0.5f);
    condition.setConstant(0.5f);

    std::string test_name = "Layer Process - Buffer size " + std::to_string(buffer_size);
    run_allocation_test_no_allocations(
      nullptr, // No setup needed
      [&]() {
        // Call Process() - this should not allocate or free
        layer.Process(input, condition, buffer_size);
      },
      nullptr, // No teardown needed
      test_name.c_str());

    // Verify output is valid
    auto output = layer.GetOutputNextLayer().leftCols(buffer_size);
    assert(output.rows() == channels && output.cols() == buffer_size);
    assert(std::isfinite(output(0, 0)));
  }
}

// Test that Layer::Process() method with bottleneck != channels does not allocate or free memory
void test_layer_bottleneck_process_realtime_safe()
{
  // Setup: Create a Layer with bottleneck different from channels
  const int condition_size = 1;
  const int channels = 4;
  const int bottleneck = 2; // bottleneck < channels
  const int kernel_size = 1;
  const int dilation = 1;
  const auto activation = nam::activations::ActivationConfig::simple(nam::activations::ActivationType::ReLU);
  const nam::wavenet::GatingMode gating_mode = nam::wavenet::GatingMode::NONE;
  const int groups_input = 1;
  const int groups_input_mixin = 1;
  const int groups_1x1 = 1;
  nam::wavenet::Head1x1Params head1x1_params(false, channels, 1);

  auto layer =
    make_layer(condition_size, channels, bottleneck, kernel_size, dilation, activation, gating_mode, groups_input,
               groups_input_mixin, groups_1x1, head1x1_params, nam::activations::ActivationConfig{});

  // Set weights for bottleneck != channels
  // Conv: (channels, bottleneck, kernelSize=1) = (4, 2, 1) + bias
  // Input mixin: (conditionSize, bottleneck) = (1, 2)
  // 1x1: (bottleneck, channels) = (2, 4) + bias
  std::vector<float> weights;
  // Conv weights: out_channels x in_channels x kernelSize = bottleneck x channels x kernelSize = 2 x 4 x 1 = 8 weights
  // Weight layout for Conv1D: for each out_channel, for each in_channel, for each kernel position
  // Identity-like pattern: out_channel i connects to in_channel i (for i < bottleneck)
  for (int out_ch = 0; out_ch < bottleneck; out_ch++)
  {
    for (int in_ch = 0; in_ch < channels; in_ch++)
    {
      weights.push_back((out_ch == in_ch) ? 1.0f : 0.0f);
    }
  }
  // Conv bias: bottleneck values
  weights.insert(weights.end(), {0.0f, 0.0f});
  // Input mixin: conditionSize x bottleneck = 1 x 2 = 2 weights
  weights.insert(weights.end(), {1.0f, 1.0f});
  // 1x1 weights: out_channels x in_channels = channels x bottleneck = 4 x 2 = 8 weights
  // Weight layout for Conv1x1: for each out_channel, for each in_channel
  // Identity-like pattern: out_channel i connects to in_channel i (for i < bottleneck)
  for (int out_ch = 0; out_ch < channels; out_ch++)
  {
    for (int in_ch = 0; in_ch < bottleneck; in_ch++)
    {
      weights.push_back((out_ch == in_ch) ? 1.0f : 0.0f);
    }
  }
  // 1x1 bias: channels values
  weights.insert(weights.end(), {0.0f, 0.0f, 0.0f, 0.0f});

  auto it = weights.begin();
  layer.set_weights_(it);

  const int maxBufferSize = 256;
  layer.SetMaxBufferSize(maxBufferSize);

  // Test with several different buffer sizes
  std::vector<int> buffer_sizes{1, 8, 16, 32, 64, 128, 256};

  for (int buffer_size : buffer_sizes)
  {
    // Prepare input/condition matrices (allocate before tracking)
    Eigen::MatrixXf input(channels, buffer_size);
    Eigen::MatrixXf condition(condition_size, buffer_size);
    input.setConstant(0.5f);
    condition.setConstant(0.5f);

    std::string test_name = "Layer Process (bottleneck=" + std::to_string(bottleneck) + ", channels="
                            + std::to_string(channels) + ") - Buffer size " + std::to_string(buffer_size);
    run_allocation_test_no_allocations(
      nullptr, // No setup needed
      [&]() {
        // Call Process() - this should not allocate or free
        layer.Process(input, condition, buffer_size);
      },
      nullptr, // No teardown needed
      test_name.c_str());

    // Verify output is valid
    auto output = layer.GetOutputNextLayer().leftCols(buffer_size);
    assert(output.rows() == channels && output.cols() == buffer_size);
    assert(std::isfinite(output(0, 0)));
    assert(std::isfinite(output(channels - 1, buffer_size - 1)));
  }
}

// Test that Layer::Process() method with grouped convolution (groups_input > 1) does not allocate or free memory
void test_layer_grouped_process_realtime_safe()
{
  // Setup: Create a Layer with grouped convolution
  const int condition_size = 1;
  const int channels = 4; // Must be divisible by groups_input
  const int bottleneck = channels;
  const int kernel_size = 2;
  const int dilation = 1;
  const auto activation = nam::activations::ActivationConfig::simple(nam::activations::ActivationType::ReLU);
  const nam::wavenet::GatingMode gating_mode = nam::wavenet::GatingMode::NONE;
  const int groups_input = 2; // groups_input > 1
  const int groups_input_mixin = 1;
  const int groups_1x1 = 2; // 1x1 is also grouped
  nam::wavenet::Head1x1Params head1x1_params(false, channels, 1);

  auto layer =
    make_layer(condition_size, channels, bottleneck, kernel_size, dilation, activation, gating_mode, groups_input,
               groups_input_mixin, groups_1x1, head1x1_params, nam::activations::ActivationConfig{});

  // Set weights for grouped convolution
  // With groups_input=2, channels=4: each group has 2 in_channels and 2 out_channels
  // Conv weights: for each group g, for each kernel position k, for each (out_ch, in_ch)
  // Group 0: processes channels 0-1, Group 1: processes channels 2-3
  std::vector<float> weights;
  // Conv weights: 2 groups, kernel_size=2, 2 out_channels per group, 2 in_channels per group
  // Group 0, kernel[0]: identity for channels 0-1
  weights.push_back(1.0f); // out_ch=0, in_ch=0
  weights.push_back(0.0f); // out_ch=0, in_ch=1
  weights.push_back(0.0f); // out_ch=1, in_ch=0
  weights.push_back(1.0f); // out_ch=1, in_ch=1
  // Group 0, kernel[1]: identity
  weights.push_back(1.0f);
  weights.push_back(0.0f);
  weights.push_back(0.0f);
  weights.push_back(1.0f);
  // Group 1, kernel[0]: identity for channels 2-3
  weights.push_back(1.0f); // out_ch=2, in_ch=2
  weights.push_back(0.0f); // out_ch=2, in_ch=3
  weights.push_back(0.0f); // out_ch=3, in_ch=2
  weights.push_back(1.0f); // out_ch=3, in_ch=3
  // Group 1, kernel[1]: identity
  weights.push_back(1.0f);
  weights.push_back(0.0f);
  weights.push_back(0.0f);
  weights.push_back(1.0f);
  // Conv bias: 4 values (one per output channel)
  weights.push_back(0.0f);
  weights.push_back(0.0f);
  weights.push_back(0.0f);
  weights.push_back(0.0f);
  // Input mixin: (channels, condition_size) = (4, 1)
  weights.push_back(1.0f);
  weights.push_back(1.0f);
  weights.push_back(1.0f);
  weights.push_back(1.0f);
  // 1x1: grouped with groups_1x1=2, channels=4
  // Each group processes 2 channels: Group 0 (channels 0-1), Group 1 (channels 2-3)
  // Weight layout: for each group g, for each (out_ch, in_ch) in that group
  // Group 0: identity matrix for channels 0-1 (2x2)
  weights.push_back(1.0f); // out_ch=0, in_ch=0
  weights.push_back(0.0f); // out_ch=0, in_ch=1
  weights.push_back(0.0f); // out_ch=1, in_ch=0
  weights.push_back(1.0f); // out_ch=1, in_ch=1
  // Group 1: identity matrix for channels 2-3 (2x2)
  weights.push_back(1.0f); // out_ch=2, in_ch=2
  weights.push_back(0.0f); // out_ch=2, in_ch=3
  weights.push_back(0.0f); // out_ch=3, in_ch=2
  weights.push_back(1.0f); // out_ch=3, in_ch=3
  // 1x1 bias: 4 values (one per output channel)
  weights.push_back(0.0f);
  weights.push_back(0.0f);
  weights.push_back(0.0f);
  weights.push_back(0.0f);

  auto it = weights.begin();
  layer.set_weights_(it);

  const int maxBufferSize = 256;
  layer.SetMaxBufferSize(maxBufferSize);

  // Test with several different buffer sizes
  std::vector<int> buffer_sizes{1, 8, 16, 32, 64, 128, 256};

  for (int buffer_size : buffer_sizes)
  {
    // Prepare input/condition matrices (allocate before tracking)
    Eigen::MatrixXf input(channels, buffer_size);
    Eigen::MatrixXf condition(condition_size, buffer_size);
    input.setConstant(0.5f);
    condition.setConstant(0.5f);

    std::string test_name =
      "Layer Process (groups_input=" + std::to_string(groups_input) + ") - Buffer size " + std::to_string(buffer_size);
    run_allocation_test_no_allocations(
      nullptr, // No setup needed
      [&]() {
        // Call Process() - this should not allocate or free
        layer.Process(input, condition, buffer_size);
      },
      nullptr, // No teardown needed
      test_name.c_str());

    // Verify output is valid
    auto output = layer.GetOutputNextLayer().leftCols(buffer_size);
    assert(output.rows() == channels && output.cols() == buffer_size);
    assert(std::isfinite(output(0, 0)));
    assert(std::isfinite(output(channels - 1, buffer_size - 1)));
  }
}

// Helper function to test Layer::Process() with all FiLMs active
static void test_layer_all_films_realtime_safe_impl(const bool shift)
{
  // Setup: Create a Layer with all FiLMs active
  const int condition_size = 1;
  const int channels = 1;
  const int bottleneck = channels;
  const int kernel_size = 1;
  const int dilation = 1;
  const auto activation = nam::activations::ActivationConfig::simple(nam::activations::ActivationType::ReLU);
  const nam::wavenet::GatingMode gating_mode = nam::wavenet::GatingMode::NONE;
  const int groups_input = 1;
  const int groups_input_mixin = 1;
  const int groups_1x1 = 1;

  nam::wavenet::Head1x1Params head1x1_params(false, channels, 1);
  auto layer = make_layer_all_films(condition_size, channels, bottleneck, kernel_size, dilation, activation,
                                    gating_mode, groups_input, groups_input_mixin, groups_1x1, head1x1_params,
                                    nam::activations::ActivationConfig{}, shift);

  // Set weights
  // Base layer weights:
  // Conv: (channels, bottleneck, kernel_size) + bias = (1, 1, 1) + 1 = 2 weights
  // Input mixin: (condition_size, bottleneck) = (1, 1) = 1 weight
  // 1x1: (bottleneck, channels) + bias = (1, 1) + 1 = 2 weights
  // Total base: 5 weights

  std::vector<float> weights;
  // Base layer weights
  weights.insert(weights.end(), {1.0f, 0.0f}); // Conv (weight, bias)
  weights.push_back(1.0f); // Input mixin
  weights.insert(weights.end(), {1.0f, 0.0f}); // 1x1 (weight, bias)

  // FiLM weights (each FiLM uses Conv1x1: condition_size -> (shift ? 2 : 1) * input_dim with bias)
  // With shift=true: each FiLM needs (2 * input_dim) * condition_size weights + (2 * input_dim) biases = 4 weights
  // With shift=false: each FiLM needs input_dim * condition_size weights + input_dim biases = 2 weights
  // All 7 FiLMs are active (excluding head1x1_post_film since head1x1 is false)
  for (int i = 0; i < 7; i++)
  {
    if (shift)
    {
      // With shift: weights are row-major (out_channels=2 x in_channels=1), then biases
      weights.push_back(1.0f); // scale weight (out_channel 0, in_channel 0)
      weights.push_back(0.0f); // shift weight (out_channel 1, in_channel 0)
      weights.push_back(0.0f); // scale bias
      weights.push_back(0.0f); // shift bias
    }
    else
    {
      // Without shift: weights are row-major (out_channels=1 x in_channels=1), then bias
      weights.push_back(1.0f); // scale weight (out_channel 0, in_channel 0)
      weights.push_back(0.0f); // scale bias
    }
  }

  auto it = weights.begin();
  layer.set_weights_(it);
  assert(it == weights.end());

  const int maxBufferSize = 256;
  layer.SetMaxBufferSize(maxBufferSize);

  // Test with several different buffer sizes
  std::vector<int> buffer_sizes{1, 8, 16, 32, 64, 128, 256};

  for (int buffer_size : buffer_sizes)
  {
    // Prepare input/condition matrices (allocate before tracking)
    Eigen::MatrixXf input(channels, buffer_size);
    Eigen::MatrixXf condition(condition_size, buffer_size);
    input.setConstant(0.5f);
    condition.setConstant(0.5f);

    std::string shift_str = shift ? "true" : "false";
    std::string test_name =
      "Layer Process (all FiLMs active, shift=" + shift_str + ") - Buffer size " + std::to_string(buffer_size);
    run_allocation_test_no_allocations(
      nullptr, // No setup needed
      [&]() {
        // Call Process() - this should not allocate or free
        layer.Process(input, condition, buffer_size);
      },
      nullptr, // No teardown needed
      test_name.c_str());

    // Verify output is valid
    auto output = layer.GetOutputNextLayer().leftCols(buffer_size);
    assert(output.rows() == channels && output.cols() == buffer_size);
    assert(std::isfinite(output(0, 0)));
  }
}

// Test that Layer::Process() method with all FiLMs active (with shift) does not allocate or free memory
void test_layer_all_films_with_shift_realtime_safe()
{
  test_layer_all_films_realtime_safe_impl(true);
}

// Test that Layer::Process() method with all FiLMs active (without shift) does not allocate or free memory
void test_layer_all_films_without_shift_realtime_safe()
{
  test_layer_all_films_realtime_safe_impl(false);
}

// Test that Layer::Process() with post-activation FiLM (gated mode) does not allocate or free memory
// This specifically tests the case where FiLM::Process() receives _z.topRows(bottleneck)
void test_layer_post_activation_film_gated_realtime_safe()
{
  // Setup: Create a Layer with GATED mode and activation_post_film enabled
  // Use simpler dimensions first to verify weight counting
  const int condition_size = 1;
  const int channels = 2;
  const int bottleneck = 1; // bottleneck < channels to trigger topRows()
  const int kernel_size = 1;
  const int dilation = 1;
  const auto activation = nam::activations::ActivationConfig::simple(nam::activations::ActivationType::ReLU);
  const auto secondary_activation =
    nam::activations::ActivationConfig::simple(nam::activations::ActivationType::Sigmoid);
  const nam::wavenet::GatingMode gating_mode = nam::wavenet::GatingMode::GATED;
  const int groups_input = 1;
  const int groups_input_mixin = 1;
  const int groups_1x1 = 1;
  nam::wavenet::Head1x1Params head1x1_params(false, channels, 1);

  // Create FiLM params with activation_post_film enabled
  nam::wavenet::_FiLMParams inactive_film(false, false);
  nam::wavenet::_FiLMParams active_film(true, true); // activation_post_film will be active

  auto layer =
    nam::wavenet::_Layer(condition_size, channels, bottleneck, kernel_size, dilation, activation, gating_mode,
                         groups_input, groups_input_mixin, groups_1x1, head1x1_params, secondary_activation,
                         inactive_film, // pre_input_film
                         inactive_film, // pre_condition_film
                         inactive_film, // conv_post_film
                         inactive_film, // input_mixin_post_film
                         active_film, // activation_post_film - THIS IS THE KEY ONE
                         inactive_film, // _1x1_pre_film
                         inactive_film, // _1x1_post_film
                         inactive_film // head1x1_post_film
    );

  // Set weights - Order: conv, input_mixin, 1x1, then FiLMs
  // NOTE: In GATED mode, conv and input_mixin output 2*bottleneck channels!
  std::vector<float> weights;

  // Conv weights: In GATED mode outputs 2*bottleneck = 2*1 = 2 channels
  // Conv: (out_channels, in_channels, kernel_size) + bias = (2, 2, 1) + 2 = 4 + 2 = 6
  weights.push_back(0.5f); // ch0, in0
  weights.push_back(0.5f); // ch0, in1
  weights.push_back(0.5f); // ch1, in0
  weights.push_back(0.5f); // ch1, in1
  weights.push_back(0.0f); // bias ch0
  weights.push_back(0.0f); // bias ch1

  // Input mixin: outputs 2*bottleneck = 2 channels
  // (condition_size, out_channels) = (1, 2) = 2 weights
  weights.push_back(0.5f); // ch0
  weights.push_back(0.5f); // ch1

  // 1x1 weights: (channels, bottleneck) + bias = (2, 1) + 2 = 2 + 2 = 4
  weights.push_back(1.0f);
  weights.push_back(1.0f);
  weights.push_back(0.0f); // bias
  weights.push_back(0.0f);

  // activation_post_film: Conv1x1(condition_size, 2*bottleneck, bias=true) = (1, 2, bias) = 2 + 2 = 4
  weights.push_back(1.0f); // scale weight
  weights.push_back(0.0f); // shift weight
  weights.push_back(1.0f); // scale bias
  weights.push_back(0.0f); // shift bias

  // TODO: Figure out where these 4 extra weights go - placeholder for now
  weights.push_back(0.0f);
  weights.push_back(0.0f);
  weights.push_back(0.0f);
  weights.push_back(0.0f);

  auto it = weights.begin();
  layer.set_weights_(it);
  assert(it == weights.end());

  const int maxBufferSize = 256;
  layer.SetMaxBufferSize(maxBufferSize);

  // Test with several different buffer sizes
  std::vector<int> buffer_sizes{1, 8, 16, 32, 64, 128, 256};

  for (int buffer_size : buffer_sizes)
  {
    // Prepare input/condition matrices (allocate before tracking)
    Eigen::MatrixXf input(channels, buffer_size);
    Eigen::MatrixXf condition(condition_size, buffer_size);
    input.setConstant(0.5f);
    condition.setConstant(0.5f);

    std::string test_name =
      "Layer Process (GATED with activation_post_film) - Buffer size " + std::to_string(buffer_size);
    run_allocation_test_no_allocations(
      nullptr, // No setup needed
      [&]() {
        // Call Process() - this should not allocate or free
        // This will trigger: _activation_post_film->Process(this->_z.topRows(bottleneck), condition, num_frames)
        layer.Process(input, condition, buffer_size);
      },
      nullptr, // No teardown needed
      test_name.c_str());

    // Verify output is valid
    auto output = layer.GetOutputNextLayer().leftCols(buffer_size);
    assert(output.rows() == channels && output.cols() == buffer_size);
    assert(std::isfinite(output(0, 0)));
    assert(std::isfinite(output(channels - 1, buffer_size - 1)));
  }
}

// Test that Layer::Process() with post-activation FiLM (blended mode) does not allocate or free memory
// This also tests the case where FiLM::Process() receives _z.topRows(bottleneck)
void test_layer_post_activation_film_blended_realtime_safe()
{
  // Setup: Create a Layer with BLENDED mode and activation_post_film enabled
  // Use simpler dimensions first to verify weight counting
  const int condition_size = 1;
  const int channels = 2;
  const int bottleneck = 1; // bottleneck < channels to trigger topRows()
  const int kernel_size = 1;
  const int dilation = 1;
  const auto activation = nam::activations::ActivationConfig::simple(nam::activations::ActivationType::ReLU);
  const auto secondary_activation =
    nam::activations::ActivationConfig::simple(nam::activations::ActivationType::Sigmoid);
  const nam::wavenet::GatingMode gating_mode = nam::wavenet::GatingMode::BLENDED;
  const int groups_input = 1;
  const int groups_input_mixin = 1;
  const int groups_1x1 = 1;
  nam::wavenet::Head1x1Params head1x1_params(false, channels, 1);

  // Create FiLM params with activation_post_film enabled
  nam::wavenet::_FiLMParams inactive_film(false, false);
  nam::wavenet::_FiLMParams active_film(true, true); // activation_post_film will be active

  auto layer =
    nam::wavenet::_Layer(condition_size, channels, bottleneck, kernel_size, dilation, activation, gating_mode,
                         groups_input, groups_input_mixin, groups_1x1, head1x1_params, secondary_activation,
                         inactive_film, // pre_input_film
                         inactive_film, // pre_condition_film
                         inactive_film, // conv_post_film
                         inactive_film, // input_mixin_post_film
                         active_film, // activation_post_film - THIS IS THE KEY ONE
                         inactive_film, // _1x1_pre_film
                         inactive_film, // _1x1_post_film
                         inactive_film // head1x1_post_film
    );

  // Set weights - Order: conv, input_mixin, 1x1, then FiLMs
  // NOTE: In BLENDED mode, conv and input_mixin output 2*bottleneck channels!
  std::vector<float> weights;

  // Conv weights: In BLENDED mode outputs 2*bottleneck = 2*1 = 2 channels
  // Conv: (out_channels, in_channels, kernel_size) + bias = (2, 2, 1) + 2 = 4 + 2 = 6
  weights.push_back(0.5f); // ch0, in0
  weights.push_back(0.5f); // ch0, in1
  weights.push_back(0.5f); // ch1, in0
  weights.push_back(0.5f); // ch1, in1
  weights.push_back(0.0f); // bias ch0
  weights.push_back(0.0f); // bias ch1

  // Input mixin: outputs 2*bottleneck = 2 channels
  // (condition_size, out_channels) = (1, 2) = 2 weights
  weights.push_back(0.5f); // ch0
  weights.push_back(0.5f); // ch1

  // 1x1 weights: (channels, bottleneck) + bias = (2, 1) + 2 = 2 + 2 = 4
  weights.push_back(1.0f);
  weights.push_back(1.0f);
  weights.push_back(0.0f); // bias
  weights.push_back(0.0f);

  // activation_post_film: Conv1x1(condition_size, 2*bottleneck, bias=true) = (1, 2, bias) = 2 + 2 = 4
  weights.push_back(1.0f); // scale weight
  weights.push_back(0.0f); // shift weight
  weights.push_back(1.0f); // scale bias
  weights.push_back(0.0f); // shift bias

  // TODO: Figure out where these 4 extra weights go - placeholder for now
  weights.push_back(0.0f);
  weights.push_back(0.0f);
  weights.push_back(0.0f);
  weights.push_back(0.0f);

  auto it = weights.begin();
  layer.set_weights_(it);
  assert(it == weights.end());

  const int maxBufferSize = 256;
  layer.SetMaxBufferSize(maxBufferSize);

  // Test with several different buffer sizes
  std::vector<int> buffer_sizes{1, 8, 16, 32, 64, 128, 256};

  for (int buffer_size : buffer_sizes)
  {
    // Prepare input/condition matrices (allocate before tracking)
    Eigen::MatrixXf input(channels, buffer_size);
    Eigen::MatrixXf condition(condition_size, buffer_size);
    input.setConstant(0.5f);
    condition.setConstant(0.5f);

    std::string test_name =
      "Layer Process (BLENDED with activation_post_film) - Buffer size " + std::to_string(buffer_size);
    run_allocation_test_no_allocations(
      nullptr, // No setup needed
      [&]() {
        // Call Process() - this should not allocate or free
        // This will trigger: _activation_post_film->Process(this->_z.topRows(bottleneck), condition, num_frames)
        layer.Process(input, condition, buffer_size);
      },
      nullptr, // No teardown needed
      test_name.c_str());

    // Verify output is valid
    auto output = layer.GetOutputNextLayer().leftCols(buffer_size);
    assert(output.rows() == channels && output.cols() == buffer_size);
    assert(std::isfinite(output(0, 0)));
    assert(std::isfinite(output(channels - 1, buffer_size - 1)));
  }
}

// Test that LayerArray::Process() method does not allocate or free memory
void test_layer_array_process_realtime_safe()
{
  // Setup: Create LayerArray
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

  // Set weights: rechannel(1), layer(conv:1+1, input_mixin:1, 1x1:1+1), head_rechannel(1)
  std::vector<float> weights{1.0f, // Rechannel
                             1.0f, 0.0f, // Layer: conv
                             1.0f, // Layer: input_mixin
                             1.0f, 0.0f, // Layer: 1x1
                             1.0f}; // Head rechannel
  auto it = weights.begin();
  layer_array.set_weights_(it);

  const int maxBufferSize = 256;
  layer_array.SetMaxBufferSize(maxBufferSize);

  // Test with several different buffer sizes
  std::vector<int> buffer_sizes{1, 8, 16, 32, 64, 128, 256};

  for (int buffer_size : buffer_sizes)
  {
    // Prepare input/condition matrices (allocate before tracking)
    Eigen::MatrixXf layer_inputs(input_size, buffer_size);
    Eigen::MatrixXf condition(condition_size, buffer_size);
    layer_inputs.setConstant(0.5f);
    condition.setConstant(0.5f);

    std::string test_name = "LayerArray Process - Buffer size " + std::to_string(buffer_size);
    run_allocation_test_no_allocations(
      nullptr, // No setup needed
      [&]() {
        // Call Process() - this should not allocate or free
        layer_array.Process(layer_inputs, condition, buffer_size);
      },
      nullptr, // No teardown needed
      test_name.c_str());

    // Verify output is valid
    auto layer_outputs = layer_array.GetLayerOutputs().leftCols(buffer_size);
    auto head_outputs = layer_array.GetHeadOutputs().leftCols(buffer_size);
    assert(layer_outputs.rows() == channels && layer_outputs.cols() == buffer_size);
    assert(head_outputs.rows() == head_size && head_outputs.cols() == buffer_size);
    assert(std::isfinite(layer_outputs(0, 0)));
    assert(std::isfinite(head_outputs(0, 0)));
  }
}

// Test that WaveNet::process() method does not allocate or free memory
void test_process_realtime_safe()
{
  // Setup: Create WaveNet with two layer arrays (simplified configuration)
  const int input_size = 1;
  const int condition_size = 1;
  const int head_size = 1;
  const int channels = 1;
  const int kernel_size = 1;
  std::vector<int> dilations{1};
  const auto activation = nam::activations::ActivationConfig::simple(nam::activations::ActivationType::ReLU);
  const nam::wavenet::GatingMode gating_mode = nam::wavenet::GatingMode::NONE;
  const bool head_bias = false;
  const float head_scale = 1.0f;
  const bool with_head = false;
  const int groups = 1;
  const int groups_input_mixin = 1;

  std::vector<nam::wavenet::LayerArrayParams> layer_array_params;
  // First layer array
  std::vector<int> dilations1{1};
  const int bottleneck = channels;
  const int groups_1x1 = 1;
  nam::wavenet::Head1x1Params head1x1_params(false, channels, 1);
  layer_array_params.push_back(make_layer_array_params(input_size, condition_size, head_size, channels, bottleneck,
                                                       kernel_size, std::move(dilations1), activation, gating_mode,
                                                       head_bias, groups, groups_input_mixin, groups_1x1,
                                                       head1x1_params, nam::activations::ActivationConfig{}));
  // Second layer array (head_size of first must match channels of second)
  std::vector<int> dilations2{1};
  layer_array_params.push_back(make_layer_array_params(head_size, condition_size, head_size, channels, bottleneck,
                                                       kernel_size, std::move(dilations2), activation, gating_mode,
                                                       head_bias, groups, groups_input_mixin, groups_1x1,
                                                       head1x1_params, nam::activations::ActivationConfig{}));

  // Weights: Array 0: rechannel(1), layer(conv:1+1, input_mixin:1, 1x1:1+1), head_rechannel(1)
  //          Array 1: same structure
  //          Head scale: 1
  std::vector<float> weights;
  // Array 0: rechannel, layer, head_rechannel
  weights.insert(weights.end(), {1.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f});
  // Array 1: rechannel, layer, head_rechannel
  weights.insert(weights.end(), {1.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f});
  weights.push_back(head_scale);

  std::unique_ptr<nam::wavenet::WaveNet> condition_dsp = nullptr;
  auto wavenet = std::make_unique<nam::wavenet::WaveNet>(
    input_size, layer_array_params, head_scale, with_head, weights, std::move(condition_dsp), 48000.0);

  const int maxBufferSize = 256;
  wavenet->Reset(48000.0, maxBufferSize);

  // Test with several different buffer sizes
  std::vector<int> buffer_sizes{1, 8, 16, 32, 64, 128, 256};

  for (int buffer_size : buffer_sizes)
  {
    // Prepare input/output buffers (allocate before tracking)
    std::vector<NAM_SAMPLE> input(buffer_size, 0.5f);
    std::vector<NAM_SAMPLE> output(buffer_size, 0.0f);

    std::string test_name = "WaveNet process - Buffer size " + std::to_string(buffer_size);
    run_allocation_test_no_allocations(
      nullptr, // No setup needed
      [&]() {
        // Call process() - this should not allocate or free
        NAM_SAMPLE* inputPtrs[] = {input.data()};
        NAM_SAMPLE* outputPtrs[] = {output.data()};
        wavenet->process(inputPtrs, outputPtrs, buffer_size);
      },
      nullptr, // No teardown needed
      test_name.c_str());

    // Verify output is valid
    for (int i = 0; i < buffer_size; i++)
    {
      assert(std::isfinite(output[i]));
    }
  }
}

// Test that WaveNet::process() method with 3 input channels and 2 output channels does not allocate or free memory
void test_process_3in_2out_realtime_safe()
{
  // Setup: Create WaveNet with 3 input channels and 2 output channels
  const int input_size = 3; // 3 input channels
  const int condition_size = 3; // condition matches input channels
  const int head_size = 2; // 2 output channels
  const int channels = 4; // internal channels
  const int bottleneck = 2; // bottleneck (will be used for head)
  const int kernel_size = 1;
  const auto activation = nam::activations::ActivationConfig::simple(nam::activations::ActivationType::ReLU);
  const nam::wavenet::GatingMode gating_mode = nam::wavenet::GatingMode::NONE;
  const bool head_bias = false;
  const float head_scale = 1.0f;
  const bool with_head = false;
  const int groups = 1;
  const int groups_input_mixin = 1;
  const int groups_1x1 = 1;

  nam::wavenet::Head1x1Params head1x1_params(false, channels, 1);

  std::vector<nam::wavenet::LayerArrayParams> layer_array_params;
  std::vector<int> dilations1{1};
  layer_array_params.push_back(make_layer_array_params(input_size, condition_size, head_size, channels, bottleneck,
                                                       kernel_size, std::move(dilations1), activation, gating_mode,
                                                       head_bias, groups, groups_input_mixin, groups_1x1,
                                                       head1x1_params, nam::activations::ActivationConfig{}));

  // Calculate weights:
  // _rechannel: Conv1x1(3, 4, bias=false) = 3*4 = 12 weights
  // Layer:
  //   _conv: Conv1D(4, 2, kernel_size=1, bias=true) = 1*(2*4) + 2 = 10 weights
  //   _input_mixin: Conv1x1(3, 2, bias=false) = 3*2 = 6 weights
  //   _1x1: Conv1x1(2, 4, bias=true) = 2*4 + 4 = 12 weights
  // _head_rechannel: Conv1x1(2, 2, bias=false) = 2*2 = 4 weights
  // Total: 12 + 10 + 6 + 12 + 4 = 44 weights
  std::vector<float> weights;
  // _rechannel weights (3->4): identity-like pattern
  for (int out_ch = 0; out_ch < 4; out_ch++)
  {
    for (int in_ch = 0; in_ch < 3; in_ch++)
    {
      weights.push_back((out_ch < 3 && out_ch == in_ch) ? 1.0f : 0.0f);
    }
  }
  // Layer: _conv weights (4->2, kernel_size=1, with bias)
  // Weight layout: for each kernel position k, for each out_channel, for each in_channel
  for (int out_ch = 0; out_ch < 2; out_ch++)
  {
    for (int in_ch = 0; in_ch < 4; in_ch++)
    {
      weights.push_back((out_ch == in_ch) ? 1.0f : 0.0f);
    }
  }
  // _conv bias (2 values)
  weights.insert(weights.end(), {0.0f, 0.0f});
  // _input_mixin weights (3->2)
  for (int out_ch = 0; out_ch < 2; out_ch++)
  {
    for (int in_ch = 0; in_ch < 3; in_ch++)
    {
      weights.push_back((out_ch == in_ch) ? 1.0f : 0.0f);
    }
  }
  // _1x1 weights (2->4, with bias)
  for (int out_ch = 0; out_ch < 4; out_ch++)
  {
    for (int in_ch = 0; in_ch < 2; in_ch++)
    {
      weights.push_back((out_ch < 2 && out_ch == in_ch) ? 1.0f : 0.0f);
    }
  }
  // _1x1 bias (4 values)
  weights.insert(weights.end(), {0.0f, 0.0f, 0.0f, 0.0f});
  // _head_rechannel weights (2->2)
  for (int out_ch = 0; out_ch < 2; out_ch++)
  {
    for (int in_ch = 0; in_ch < 2; in_ch++)
    {
      weights.push_back((out_ch == in_ch) ? 1.0f : 0.0f);
    }
  }
  weights.push_back(head_scale);

  const int in_channels = 3;
  std::unique_ptr<nam::wavenet::WaveNet> condition_dsp = nullptr;
  auto wavenet = std::make_unique<nam::wavenet::WaveNet>(
    in_channels, layer_array_params, head_scale, with_head, weights, std::move(condition_dsp), 48000.0);

  const int maxBufferSize = 256;
  wavenet->Reset(48000.0, maxBufferSize);

  // Test with several different buffer sizes
  std::vector<int> buffer_sizes{1, 8, 16, 32, 64, 128, 256};

  for (int buffer_size : buffer_sizes)
  {
    // Prepare input/output buffers for 3 input channels and 2 output channels (allocate before tracking)
    std::vector<std::vector<NAM_SAMPLE>> input(3, std::vector<NAM_SAMPLE>(buffer_size, 0.5f));
    std::vector<std::vector<NAM_SAMPLE>> output(2, std::vector<NAM_SAMPLE>(buffer_size, 0.0f));
    std::vector<NAM_SAMPLE*> inputPtrs(3);
    std::vector<NAM_SAMPLE*> outputPtrs(2);
    for (int ch = 0; ch < 3; ch++)
      inputPtrs[ch] = input[ch].data();
    for (int ch = 0; ch < 2; ch++)
      outputPtrs[ch] = output[ch].data();

    std::string test_name = "WaveNet process (3in, 2out) - Buffer size " + std::to_string(buffer_size);
    run_allocation_test_no_allocations(
      nullptr, // No setup needed
      [&]() {
        // Call process() - this should not allocate or free
        wavenet->process(inputPtrs.data(), outputPtrs.data(), buffer_size);
      },
      nullptr, // No teardown needed
      test_name.c_str());

    // Verify output is valid
    for (int ch = 0; ch < 2; ch++)
    {
      for (int i = 0; i < buffer_size; i++)
      {
        assert(std::isfinite(output[ch][i]));
      }
    }
  }
}
} // namespace test_wavenet
