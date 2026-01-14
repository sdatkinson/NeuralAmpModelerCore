// Tests for gating activation functions

#include <cassert>
#include <string>
#include <vector>
#include <cmath>
#include <iostream>

#include "NAM/gating_activations.h"
#include "NAM/activations.h"

namespace test_gating_activations
{

class TestGatingActivation
{
public:
  static void test_basic_functionality()
  {
    // Create test input data (2 rows, 3 columns)
    Eigen::MatrixXf input(2, 3);
    input << 1.0f, -1.0f, 0.0f, 0.5f, 0.8f, 1.0f;

    Eigen::MatrixXf output(1, 3);

    // Create gating activation with default activations (1 input channel, 1 gating channel)
    nam::gating_activations::GatingActivation gating_act(nullptr, nullptr, 1, 1);

    // Apply the activation
    gating_act.apply(input, output);

    // Basic checks
    assert(output.rows() == 1);
    assert(output.cols() == 3);

    // The output should be element-wise multiplication of the two rows
    // after applying activations
    std::cout << "GatingActivation basic test passed" << std::endl;
  }

  static void test_with_custom_activations()
  {
    // Create custom activations
    nam::activations::ActivationLeakyReLU leaky_relu(0.01f);
    nam::activations::ActivationLeakyReLU leaky_relu2(0.05f);

    // Create test input data
    Eigen::MatrixXf input(2, 2);
    input << -1.0f, 1.0f, -2.0f, 0.5f;

    Eigen::MatrixXf output(1, 2);

    // Create gating activation with custom activations
    nam::gating_activations::GatingActivation gating_act(&leaky_relu, &leaky_relu2, 1, 1);

    // Apply the activation
    gating_act.apply(input, output);

    // Verify dimensions
    assert(output.rows() == 1);
    assert(output.cols() == 2);

    std::cout << "GatingActivation custom activations test passed" << std::endl;
  }

  static void test_error_handling()
  {
    // Test with insufficient rows
    Eigen::MatrixXf input(1, 3); // Only 1 row
    Eigen::MatrixXf output;

    nam::gating_activations::GatingActivation gating_act;

    try
    {
      gating_act.apply(input, output);
      assert(false); // Should not reach here
    }
    catch (const std::invalid_argument& e)
    {
      std::cout << "GatingActivation error handling test passed: " << e.what() << std::endl;
    }
  }
};

class TestBlendingActivation
{
public:
  static void test_basic_functionality()
  {
    // Create test input data (2 rows, 3 columns)
    Eigen::MatrixXf input(2, 3);
    input << 1.0f, -1.0f, 0.0f, 0.5f, 0.8f, 1.0f;

    Eigen::MatrixXf output(1, 3);

    // Create blending activation with default alpha (0.5)
    nam::gating_activations::BlendingActivation blending_act(nullptr, nullptr, 0.5f, 1, 1);

    // Apply the activation
    blending_act.apply(input, output);

    // Basic checks
    assert(output.rows() == 1);
    assert(output.cols() == 3);

    std::cout << "BlendingActivation basic test passed" << std::endl;
  }

  static void test_different_alpha_values()
  {
    // Test with alpha = 0.0 (should use only second row)
    Eigen::MatrixXf input(2, 2);
    input << 1.0f, -1.0f, 2.0f, 3.0f;

    Eigen::MatrixXf output(1, 2);

    nam::gating_activations::BlendingActivation blending_act(nullptr, nullptr, 0.0f, 1, 1);
    blending_act.apply(input, output);

    // With alpha=0.0, output should be close to second row
    assert(fabs(output(0, 0) - 2.0f) < 1e-6);
    assert(fabs(output(0, 1) - 3.0f) < 1e-6);

    // Test with alpha = 1.0 (should use only first row)
    nam::gating_activations::BlendingActivation blending_act2(nullptr, nullptr, 1.0f, 1, 1);
    blending_act2.apply(input, output);

    // With alpha=1.0, output should be close to first row
    assert(fabs(output(0, 0) - 1.0f) < 1e-6);
    assert(fabs(output(0, 1) - (-1.0f)) < 1e-6);

    // Test with alpha = 0.3
    nam::gating_activations::BlendingActivation blending_act3(nullptr, nullptr, 0.3f, 1, 1);
    blending_act3.apply(input, output);

    // With alpha=0.3, output should be 0.3*row1 + 0.7*row2
    float expected0 = 0.3f * 1.0f + 0.7f * 2.0f;
    float expected1 = 0.3f * (-1.0f) + 0.7f * 3.0f;

    assert(fabs(output(0, 0) - expected0) < 1e-6);
    assert(fabs(output(0, 1) - expected1) < 1e-6);

    std::cout << "BlendingActivation alpha values test passed" << std::endl;
  }

  static void test_with_custom_activations()
  {
    // Create custom activations
    nam::activations::ActivationLeakyReLU leaky_relu(0.01f);
    nam::activations::ActivationLeakyReLU leaky_relu2(0.05f);

    // Create test input data
    Eigen::MatrixXf input(2, 2);
    input << -1.0f, 1.0f, -2.0f, 0.5f;

    Eigen::MatrixXf output(1, 2);

    // Create blending activation with custom activations and alpha = 0.7
    nam::gating_activations::BlendingActivation blending_act(&leaky_relu, &leaky_relu2, 0.7f, 1, 1);

    // Apply the activation
    blending_act.apply(input, output);

    // Verify dimensions
    assert(output.rows() == 1);
    assert(output.cols() == 2);

    std::cout << "BlendingActivation custom activations test passed" << std::endl;
  }

  static void test_error_handling()
  {
    // Test with insufficient rows
    Eigen::MatrixXf input(1, 2); // Only 1 row
    Eigen::MatrixXf output;

    nam::gating_activations::BlendingActivation blending_act;

    try
    {
      blending_act.apply(input, output);
      assert(false); // Should not reach here
    }
    catch (const std::invalid_argument& e)
    {
      std::cout << "BlendingActivation error handling test passed: " << e.what() << std::endl;
    }

    // Test with invalid alpha value
    try
    {
      nam::gating_activations::BlendingActivation blending_act(nullptr, nullptr, 1.5f);
      assert(false); // Should not reach here
    }
    catch (const std::invalid_argument& e)
    {
      std::cout << "BlendingActivation alpha validation test passed: " << e.what() << std::endl;
    }

    try
    {
      nam::gating_activations::BlendingActivation blending_act(nullptr, nullptr, -0.1f);
      assert(false); // Should not reach here
    }
    catch (const std::invalid_argument& e)
    {
      std::cout << "BlendingActivation alpha validation test passed: " << e.what() << std::endl;
    }
  }

  static void test_edge_cases()
  {
    // Test with zero input
    Eigen::MatrixXf input(2, 1);
    input << 0.0f, 0.0f;

    Eigen::MatrixXf output(1, 1);

    nam::gating_activations::BlendingActivation blending_act(nullptr, nullptr, 0.5f, 1, 1);
    blending_act.apply(input, output);

    assert(fabs(output(0, 0) - 0.0f) < 1e-6);

    // Test with large values
    Eigen::MatrixXf input2(2, 1);
    input2 << 1000.0f, -1000.0f;

    blending_act.apply(input2, output);

    // Should handle large values without issues
    assert(output.rows() == 1);
    assert(output.cols() == 1);

    std::cout << "BlendingActivation edge cases test passed" << std::endl;
  }
};

}; // namespace test_gating_activations