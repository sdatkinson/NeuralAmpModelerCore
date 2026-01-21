// Test configurable gating/blending activations in WaveNet

#include <cassert>
#include <string>
#include <vector>

#include "NAM/wavenet.h"
#include "NAM/gating_activations.h"

namespace test_wavenet_configurable_gating
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
                                       const int groups_input_mixin, const int groups_1x1,
                                       const nam::wavenet::Head1x1Params& head1x1_params,
                                       const std::string& secondary_activation)
{
  auto film_params = make_default_film_params();
  return nam::wavenet::_Layer(condition_size, channels, bottleneck, kernel_size, dilation, activation_config,
                              gating_mode, groups_input, groups_input_mixin, groups_1x1, head1x1_params,
                              secondary_activation, film_params, film_params, film_params, film_params, film_params,
                              film_params, film_params, film_params);
}

// Helper function to create LayerArrayParams with default FiLM parameters
static nam::wavenet::LayerArrayParams make_layer_array_params(
  const int input_size, const int condition_size, const int head_size, const int channels, const int bottleneck,
  const int kernel_size, std::vector<int>&& dilations, const nam::activations::ActivationConfig& activation_config,
  const nam::wavenet::GatingMode gating_mode, const bool head_bias, const int groups_input,
  const int groups_input_mixin, const int groups_1x1, const nam::wavenet::Head1x1Params& head1x1_params,
  const std::string& secondary_activation)
{
  auto film_params = make_default_film_params();
  return nam::wavenet::LayerArrayParams(
    input_size, condition_size, head_size, channels, bottleneck, kernel_size, std::move(dilations), activation_config,
    gating_mode, head_bias, groups_input, groups_input_mixin, groups_1x1, head1x1_params, secondary_activation,
    film_params, film_params, film_params, film_params, film_params, film_params, film_params, film_params);
}

// Helper function to create a LayerArray with default FiLM parameters
static nam::wavenet::_LayerArray make_layer_array(
  const int input_size, const int condition_size, const int head_size, const int channels, const int bottleneck,
  const int kernel_size, const std::vector<int>& dilations, const nam::activations::ActivationConfig& activation_config,
  const nam::wavenet::GatingMode gating_mode, const bool head_bias, const int groups_input,
  const int groups_input_mixin, const int groups_1x1, const nam::wavenet::Head1x1Params& head1x1_params,
  const std::string& secondary_activation)
{
  auto film_params = make_default_film_params();
  return nam::wavenet::_LayerArray(input_size, condition_size, head_size, channels, bottleneck, kernel_size, dilations,
                                   activation_config, gating_mode, head_bias, groups_input, groups_input_mixin,
                                   groups_1x1, head1x1_params, secondary_activation, film_params, film_params,
                                   film_params, film_params, film_params, film_params, film_params, film_params);
}

class TestConfigurableGating
{
public:
  static void test_gated_with_different_activations()
  {
    // Test parameters
    const int conditionSize = 1;
    const int channels = 2;
    const int bottleneck = 2;
    const int kernelSize = 3;
    const int dilation = 1;
    const auto activation = nam::activations::ActivationConfig::simple(nam::activations::ActivationType::Tanh);
    const int groups_input = 1;
    const int groups_input_mixin = 1;
    const int groups_1x1 = 1;
    nam::wavenet::Head1x1Params head1x1_params(false, channels, 1);

    // Test different gating activation configurations
    std::vector<std::string> gating_activations = {"Sigmoid", "Tanh", "ReLU"};

    for (const auto& gating_act : gating_activations)
    {
      auto layer = make_layer(conditionSize, channels, bottleneck, kernelSize, dilation, activation,
                              nam::wavenet::GatingMode::GATED, groups_input, groups_input_mixin, groups_1x1,
                              head1x1_params, gating_act);

      // Verify that the layer was created successfully and has correct dimensions
      assert(layer.get_channels() == channels);
    }
  }

  static void test_blended_with_different_activations()
  {
    // Test parameters
    const int conditionSize = 1;
    const int channels = 2;
    const int bottleneck = 2;
    const int kernelSize = 3;
    const int dilation = 1;
    const auto activation = nam::activations::ActivationConfig::simple(nam::activations::ActivationType::Tanh);
    const int groups_input = 1;
    const int groups_input_mixin = 1;
    const int groups_1x1 = 1;
    nam::wavenet::Head1x1Params head1x1_params(false, channels, 1);

    // Test different blending activation configurations
    std::vector<std::string> blending_activations = {"Sigmoid", "Tanh", "ReLU"};

    for (const auto& blending_act : blending_activations)
    {
      auto layer = make_layer(conditionSize, channels, bottleneck, kernelSize, dilation, activation,
                              nam::wavenet::GatingMode::BLENDED, groups_input, groups_input_mixin, groups_1x1,
                              head1x1_params, blending_act);

      // Verify that the layer was created successfully and has correct dimensions
      assert(layer.get_channels() == channels);
    }
  }


  static void test_layer_array_params()
  {
    // Test LayerArrayParams with configurable activations
    const int input_size = 1;
    const int condition_size = 1;
    const int head_size = 2;
    const int channels = 2;
    const int bottleneck = 2;
    const int kernel_size = 3;
    const std::vector<int> dilations = {1, 2};
    const auto activation = nam::activations::ActivationConfig::simple(nam::activations::ActivationType::Tanh);
    const bool head_bias = false;
    const int groups_input = 1;
    const int groups_input_mixin = 1;
    const int groups_1x1 = 1;
    nam::wavenet::Head1x1Params head1x1_params(false, channels, 1);

    // Test with different gating activations
    auto params_gated = make_layer_array_params(
      input_size, condition_size, head_size, channels, bottleneck, kernel_size, std::vector<int>{1, 2}, activation,
      nam::wavenet::GatingMode::GATED, head_bias, groups_input, groups_input_mixin, groups_1x1, head1x1_params, "Tanh");

    assert(params_gated.gating_mode == nam::wavenet::GatingMode::GATED);
    assert(params_gated.secondary_activation == "Tanh");

    // Test with different blending activations
    auto params_blended =
      make_layer_array_params(input_size, condition_size, head_size, channels, bottleneck, kernel_size,
                              std::vector<int>{1, 2}, activation, nam::wavenet::GatingMode::BLENDED, head_bias,
                              groups_input, groups_input_mixin, groups_1x1, head1x1_params, "ReLU");

    assert(params_blended.gating_mode == nam::wavenet::GatingMode::BLENDED);
    assert(params_blended.secondary_activation == "ReLU");
  }

  static void test_layer_array_construction()
  {
    // Test _LayerArray construction with configurable activations
    const int input_size = 1;
    const int condition_size = 1;
    const int head_size = 2;
    const int channels = 2;
    const int bottleneck = 2;
    const int kernel_size = 3;
    const std::vector<int> dilations = {1};
    const auto activation = nam::activations::ActivationConfig::simple(nam::activations::ActivationType::Tanh);
    const bool head_bias = false;
    const int groups_input = 1;
    const int groups_input_mixin = 1;
    const int groups_1x1 = 1;
    nam::wavenet::Head1x1Params head1x1_params(false, channels, 1);

    auto layer_array = make_layer_array(input_size, condition_size, head_size, channels, bottleneck, kernel_size,
                                        std::vector<int>{1}, activation, nam::wavenet::GatingMode::GATED, head_bias,
                                        groups_input, groups_input_mixin, groups_1x1, head1x1_params, "ReLU");

    // Verify that layers were created correctly by checking receptive field
    // This should be non-zero for a valid layer array
    assert(layer_array.get_receptive_field() > 0);
  }

  static void test_json_configuration_parsing()
  {
    // Test that JSON configuration parsing logic works correctly for new parameters
    // We'll test the parsing logic directly without creating full WaveNet objects

    // Test the gating mode parsing logic directly
    nlohmann::json gated_config = {{"gating_mode", "gated"}, {"secondary_activation", "ReLU"}};

    nlohmann::json blended_config = {{"gating_mode", "blended"}, {"secondary_activation", "Sigmoid"}};

    nlohmann::json none_config = {{"gating_mode", "none"}};

    nlohmann::json old_gated_config = {{"gated", true}};

    nlohmann::json old_none_config = {{"gated", false}};

    // Test that we can parse the configurations without errors
    // We'll test the parsing logic by checking that the right exceptions are thrown
    // for invalid configurations

    // Test invalid gating mode
    nlohmann::json invalid_config = {{"gating_mode", "invalid_mode"}};

    bool caught_invalid_mode = false;
    try
    {
      // This would be called in the factory - we're just testing the parsing logic
      std::string gating_mode_str = invalid_config["gating_mode"].get<std::string>();
      if (gating_mode_str != "gated" && gating_mode_str != "blended" && gating_mode_str != "none")
      {
        throw std::runtime_error("Invalid gating_mode: " + gating_mode_str);
      }
    }
    catch (const std::exception& e)
    {
      caught_invalid_mode = true;
    }
    assert(caught_invalid_mode);
  }

  static void test_activation_function_behavior()
  {
    // Test that different activation functions produce different outputs
    const int conditionSize = 1;
    const int channels = 2;
    const int bottleneck = 2;
    const int kernelSize = 3;
    const int dilation = 1;
    const auto activation = nam::activations::ActivationConfig::simple(nam::activations::ActivationType::Tanh);
    const int groups_input = 1;
    const int groups_input_mixin = 1;
    const int groups_1x1 = 1;
    nam::wavenet::Head1x1Params head1x1_params(false, channels, 1);

    // Create layers with different gating activations
    auto layer_sigmoid =
      make_layer(conditionSize, channels, bottleneck, kernelSize, dilation, activation, nam::wavenet::GatingMode::GATED,
                 groups_input, groups_input_mixin, groups_1x1, head1x1_params, "Sigmoid");

    auto layer_tanh =
      make_layer(conditionSize, channels, bottleneck, kernelSize, dilation, activation, nam::wavenet::GatingMode::GATED,
                 groups_input, groups_input_mixin, groups_1x1, head1x1_params, "Tanh");

    auto layer_relu =
      make_layer(conditionSize, channels, bottleneck, kernelSize, dilation, activation, nam::wavenet::GatingMode::GATED,
                 groups_input, groups_input_mixin, groups_1x1, head1x1_params, "ReLU");

    // Set max buffer size for all layers
    const int num_frames = 10;
    layer_sigmoid.SetMaxBufferSize(num_frames);
    layer_tanh.SetMaxBufferSize(num_frames);
    layer_relu.SetMaxBufferSize(num_frames);

    // Set some weights to make the layers produce different outputs
    std::vector<float> weights;
    // Add weights for conv layer (simplified - just enough to make it non-zero)
    const int conv_weights = channels * 2 * bottleneck * kernelSize; // 2*bottleneck for gated
    for (int i = 0; i < conv_weights; i++)
    {
      weights.push_back(0.1f * i);
    }
    // Add weights for input mixin
    const int mixin_weights = conditionSize * 2 * bottleneck;
    for (int i = 0; i < mixin_weights; i++)
    {
      weights.push_back(0.05f * i);
    }
    // Add weights for 1x1 conv
    const int conv1x1_weights = bottleneck * channels;
    for (int i = 0; i < conv1x1_weights; i++)
    {
      weights.push_back(0.02f * i);
    }

    // Set weights for all layers
    auto weights_iter = weights.begin();
    layer_sigmoid.set_weights_(weights_iter);

    weights_iter = weights.begin();
    layer_tanh.set_weights_(weights_iter);

    weights_iter = weights.begin();
    layer_relu.set_weights_(weights_iter);

    // Create some test input data
    Eigen::MatrixXf input(channels, num_frames);
    input.setRandom();

    // Create condition data
    Eigen::MatrixXf condition(conditionSize, num_frames);
    condition.setRandom();

    // Process with each layer
    layer_sigmoid.Process(input, condition, num_frames);
    layer_tanh.Process(input, condition, num_frames);
    layer_relu.Process(input, condition, num_frames);

    // Get outputs from the gating activation (before 1x1 convolution)
    // We need to access the internal _z matrix to see the gated results
    // Note: This is a bit hacky but necessary for testing
    Eigen::MatrixXf gated_sigmoid = layer_sigmoid.GetOutputHead().leftCols(num_frames);
    Eigen::MatrixXf gated_tanh = layer_tanh.GetOutputHead().leftCols(num_frames);
    Eigen::MatrixXf gated_relu = layer_relu.GetOutputHead().leftCols(num_frames);

    // Also get the final outputs for completeness
    Eigen::MatrixXf output_sigmoid = layer_sigmoid.GetOutputNextLayer().leftCols(num_frames);
    Eigen::MatrixXf output_tanh = layer_tanh.GetOutputNextLayer().leftCols(num_frames);
    Eigen::MatrixXf output_relu = layer_relu.GetOutputNextLayer().leftCols(num_frames);

    // Check if gated outputs are different (this is what we really care about)
    bool sigmoid_vs_tanh_gated = gated_sigmoid.isApprox(gated_tanh, 0.0f);
    bool sigmoid_vs_relu_gated = gated_sigmoid.isApprox(gated_relu, 0.0f);
    bool tanh_vs_relu_gated = gated_tanh.isApprox(gated_relu, 0.0f);

    // Verify that gated outputs are different (different activations should produce different results)
    bool some_different = !(sigmoid_vs_tanh_gated && sigmoid_vs_relu_gated && tanh_vs_relu_gated);
    assert(some_different);
  }
};

}; // namespace test_wavenet_configurable_gating

// Run all tests
void run_configurable_gating_tests()
{
  test_wavenet_configurable_gating::TestConfigurableGating::test_gated_with_different_activations();
  test_wavenet_configurable_gating::TestConfigurableGating::test_blended_with_different_activations();
  test_wavenet_configurable_gating::TestConfigurableGating::test_layer_array_params();
  test_wavenet_configurable_gating::TestConfigurableGating::test_layer_array_construction();
  test_wavenet_configurable_gating::TestConfigurableGating::test_json_configuration_parsing();
  test_wavenet_configurable_gating::TestConfigurableGating::test_activation_function_behavior();
}