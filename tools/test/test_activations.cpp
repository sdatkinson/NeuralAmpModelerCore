// Tests for activation functions
//
// Things you want ot test for:
// 1. That the core elementwise funciton is snapshot-correct.
// 2. The class that wraps the core function for an array of data
// 3. .cpp: that you have the singleton defined, and that it's in the unordered map to get by string

#include <cassert>
#include <string>
#include <vector>
#include <cmath>

#include "NAM/activations.h"

namespace test_activations
{
// TODO get nonzero cases
class TestFastTanh
{
public:
  static void test_core_function()
  {
    auto TestCase = [](float input, float expectedOutput) {
      float actualOutput = nam::activations::fast_tanh(input);
      assert(actualOutput == expectedOutput);
    };
    // A few snapshot tests
    TestCase(0.0f, 0.0f);
    // TestCase(1.0f, 1.0f);
    // TestCase(-1.0f, -0.01f);
  };

  static void test_get_by_init()
  {
    auto a = nam::activations::ActivationLeakyReLU();
    _test_class(&a);
  }

  // Get the singleton and test it
  static void test_get_by_str()
  {
    const std::string name = "Fasttanh";
    auto a = nam::activations::Activation::get_activation(name);
    _test_class(a.get());
  }

private:
  // Put the class through its paces
  static void _test_class(nam::activations::Activation* a)
  {
    std::vector<float> inputs, expectedOutputs;

    inputs.push_back(0.0f);
    expectedOutputs.push_back(0.0f);

    // inputs.push_back(1.0f);
    // expectedOutputs.push_back(1.0f);

    // inputs.push_back(-1.0f);
    // expectedOutputs.push_back(-0.01f);

    a->apply(inputs.data(), (long)inputs.size());
    for (auto itActual = inputs.begin(), itExpected = expectedOutputs.begin(); itActual != inputs.end();
         ++itActual, ++itExpected)
    {
      assert(*itActual == *itExpected);
    }
  };
};

class TestLeakyReLU
{
public:
  static void test_core_function()
  {
    auto TestCase = [](float input, float expectedOutput) {
      float actualOutput = nam::activations::leaky_relu(input, 0.01);
      assert(actualOutput == expectedOutput);
    };
    // A few snapshot tests
    TestCase(0.0f, 0.0f);
    TestCase(1.0f, 1.0f);
    TestCase(-1.0f, -0.01f);
  };

  static void test_get_by_init()
  {
    auto a = nam::activations::ActivationLeakyReLU(0.01);
    _test_class(&a);
  }

  // Get the singleton and test it
  static void test_get_by_str()
  {
    const std::string name = "LeakyReLU";
    auto a = nam::activations::Activation::get_activation(name);
    _test_class(a.get());
  }

private:
  // Put the class through its paces
  static void _test_class(nam::activations::Activation* a)
  {
    std::vector<float> inputs, expectedOutputs;

    inputs.push_back(0.0f);
    expectedOutputs.push_back(0.0f);

    inputs.push_back(1.0f);
    expectedOutputs.push_back(1.0f);

    inputs.push_back(-1.0f);
    expectedOutputs.push_back(-0.01f);

    a->apply(inputs.data(), (long)inputs.size());
    for (auto itActual = inputs.begin(), itExpected = expectedOutputs.begin(); itActual != inputs.end();
         ++itActual, ++itExpected)
    {
      assert(*itActual == *itExpected);
    }
  };
};
class TestPReLU
{
public:
  static void test_core_function()
  {
    // Test the basic leaky_relu function that PReLU uses
    auto TestCase = [](float input, float slope, float expectedOutput) {
      float actualOutput = nam::activations::leaky_relu(input, slope);
      assert(actualOutput == expectedOutput);
    };

    // A few snapshot tests
    TestCase(0.0f, 0.01f, 0.0f);
    TestCase(1.0f, 0.01f, 1.0f);
    TestCase(-1.0f, 0.01f, -0.01f);
    TestCase(-1.0f, 0.05f, -0.05f); // Different slope
  }

  static void test_per_channel_behavior()
  {
    // Test that different slopes are applied to different channels
    Eigen::MatrixXf data(2, 3); // 2 channels, 3 time steps

    // Initialize with some test data
    data << -1.0f, 0.5f, 1.0f, -2.0f, -0.5f, 0.0f;

    // Create PReLU with different slopes for each channel
    std::vector<float> slopes = {0.01f, 0.05f}; // slope 0.01 for channel 0, 0.05 for channel 1
    nam::activations::ActivationPReLU prelu(slopes);

    // Apply the activation
    prelu.apply(data);

    // Verify the results
    // Channel 0 (slope = 0.01):
    assert(fabs(data(0, 0) - (-0.01f)) < 1e-6); // -1.0 * 0.01 = -0.01
    assert(fabs(data(0, 1) - 0.5f) < 1e-6); // 0.5 (positive, unchanged)
    assert(fabs(data(0, 2) - 1.0f) < 1e-6); // 1.0 (positive, unchanged)

    // Channel 1 (slope = 0.05):
    assert(fabs(data(1, 0) - (-0.10f)) < 1e-6); // -2.0 * 0.05 = -0.10
    assert(fabs(data(1, 1) - (-0.025f)) < 1e-6); // -0.5 * 0.05 = -0.025
    assert(fabs(data(1, 2) - 0.0f) < 1e-6); // 0.0 (unchanged)
  }

  static void test_wrong_number_of_channels()
  {
    // Test that we fail when we have more channels than slopes
    Eigen::MatrixXf data(3, 2); // 3 channels, 2 time steps

    // Initialize with test data
    data << -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f;

    // Create PReLU with only 2 slopes for 3 channels
    std::vector<float> slopes = {0.01f, 0.05f};
    nam::activations::ActivationPReLU prelu(slopes);

    // Apply the activation
    bool caught = false;
    try
    {
      prelu.apply(data);
    }
    catch (const std::runtime_error& e)
    {
      caught = true;
    }
    catch (...)
    {
    }

    assert(caught);
  }
};

class TestTypedActivationConfig
{
public:
  static void test_simple_config()
  {
    // Test simple() factory method
    auto config = nam::activations::ActivationConfig::simple(nam::activations::ActivationType::ReLU);
    assert(config.type == nam::activations::ActivationType::ReLU);
    assert(!config.negative_slope.has_value());
    assert(!config.negative_slopes.has_value());

    auto act = nam::activations::Activation::get_activation(config);
    assert(act != nullptr);
  }

  static void test_all_simple_types()
  {
    // Test that all simple activation types work
    std::vector<nam::activations::ActivationType> types = {
      nam::activations::ActivationType::Tanh,     nam::activations::ActivationType::Hardtanh,
      nam::activations::ActivationType::Fasttanh, nam::activations::ActivationType::ReLU,
      nam::activations::ActivationType::Sigmoid,  nam::activations::ActivationType::SiLU,
      nam::activations::ActivationType::Hardswish};

    for (auto type : types)
    {
      auto config = nam::activations::ActivationConfig::simple(type);
      auto act = nam::activations::Activation::get_activation(config);
      assert(act != nullptr);
    }
  }

  static void test_leaky_relu_config()
  {
    // Test LeakyReLU with custom negative slope
    nam::activations::ActivationConfig config;
    config.type = nam::activations::ActivationType::LeakyReLU;
    config.negative_slope = 0.2f;

    auto act = nam::activations::Activation::get_activation(config);
    assert(act != nullptr);

    // Verify the behavior
    std::vector<float> data = {-1.0f, 0.0f, 1.0f};
    act->apply(data.data(), (long)data.size());
    assert(fabs(data[0] - (-0.2f)) < 1e-6); // -1.0 * 0.2 = -0.2
    assert(fabs(data[1] - 0.0f) < 1e-6);
    assert(fabs(data[2] - 1.0f) < 1e-6);
  }

  static void test_prelu_single_slope_config()
  {
    // Test PReLU with single slope
    nam::activations::ActivationConfig config;
    config.type = nam::activations::ActivationType::PReLU;
    config.negative_slope = 0.25f;

    auto act = nam::activations::Activation::get_activation(config);
    assert(act != nullptr);
  }

  static void test_prelu_multi_slope_config()
  {
    // Test PReLU with multiple slopes (per-channel)
    nam::activations::ActivationConfig config;
    config.type = nam::activations::ActivationType::PReLU;
    config.negative_slopes = std::vector<float>{0.1f, 0.2f, 0.3f};

    auto act = nam::activations::Activation::get_activation(config);
    assert(act != nullptr);

    // Verify per-channel behavior
    Eigen::MatrixXf data(3, 2);
    data << -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f;

    act->apply(data);

    // Channel 0 (slope 0.1): -1.0 * 0.1 = -0.1
    assert(fabs(data(0, 0) - (-0.1f)) < 1e-6);
    // Channel 1 (slope 0.2): -1.0 * 0.2 = -0.2
    assert(fabs(data(1, 0) - (-0.2f)) < 1e-6);
    // Channel 2 (slope 0.3): -1.0 * 0.3 = -0.3
    assert(fabs(data(2, 0) - (-0.3f)) < 1e-6);
    // Positive values unchanged
    assert(fabs(data(0, 1) - 1.0f) < 1e-6);
  }

  static void test_leaky_hardtanh_config()
  {
    // Test LeakyHardtanh with custom parameters
    nam::activations::ActivationConfig config;
    config.type = nam::activations::ActivationType::LeakyHardtanh;
    config.min_val = -2.0f;
    config.max_val = 2.0f;
    config.min_slope = 0.1f;
    config.max_slope = 0.1f;

    auto act = nam::activations::Activation::get_activation(config);
    assert(act != nullptr);
  }

  static void test_from_json_string()
  {
    // Test from_json with string input
    nlohmann::json j = "ReLU";
    auto config = nam::activations::ActivationConfig::from_json(j);
    assert(config.type == nam::activations::ActivationType::ReLU);
  }

  static void test_from_json_object()
  {
    // Test from_json with object input
    nlohmann::json j = {{"type", "LeakyReLU"}, {"negative_slope", 0.15f}};
    auto config = nam::activations::ActivationConfig::from_json(j);
    assert(config.type == nam::activations::ActivationType::LeakyReLU);
    assert(config.negative_slope.has_value());
    assert(fabs(config.negative_slope.value() - 0.15f) < 1e-6);
  }

  static void test_from_json_prelu_multi()
  {
    // Test from_json with PReLU multi-slope
    nlohmann::json j = {{"type", "PReLU"}, {"negative_slopes", {0.1f, 0.2f, 0.3f, 0.4f}}};
    auto config = nam::activations::ActivationConfig::from_json(j);
    assert(config.type == nam::activations::ActivationType::PReLU);
    assert(config.negative_slopes.has_value());
    assert(config.negative_slopes.value().size() == 4);
  }

  static void test_unknown_activation_throws()
  {
    // Test that unknown activation type throws
    nlohmann::json j = "UnknownActivation";
    bool threw = false;
    try
    {
      nam::activations::ActivationConfig::from_json(j);
    }
    catch (const std::runtime_error& e)
    {
      threw = true;
    }
    assert(threw);
  }
};

}; // namespace test_activations
