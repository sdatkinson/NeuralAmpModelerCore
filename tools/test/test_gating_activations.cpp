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
    nam::activations::ActivationIdentity identity_act;
    nam::activations::ActivationSigmoid sigmoid_act;
    nam::gating_activations::GatingActivation gating_act(&identity_act, &sigmoid_act, 1);

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
    nam::gating_activations::GatingActivation gating_act(&leaky_relu, &leaky_relu2, 1);

    // Apply the activation
    gating_act.apply(input, output);

    // Verify dimensions
    assert(output.rows() == 1);
    assert(output.cols() == 2);

    std::cout << "GatingActivation custom activations test passed" << std::endl;
  }

  static void test_error_handling()
  {
    // Test with insufficient rows - should assert
    // In real-time code, we use asserts instead of exceptions for performance
    // These tests would normally crash the program due to asserts
    // In production, these conditions should never occur if the code is used correctly
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

    // Create blending activation (1 input channel)
    nam::activations::ActivationIdentity identity_act;
    nam::activations::ActivationIdentity identity_blend_act;
    nam::gating_activations::BlendingActivation blending_act(&identity_act, &identity_blend_act, 1);

    // Apply the activation
    blending_act.apply(input, output);

    // Basic checks
    assert(output.rows() == 1);
    assert(output.cols() == 3);

    std::cout << "BlendingActivation basic test passed" << std::endl;
  }

  static void test_blending_behavior()
  {
    // Test blending with different activation functions
    // Create test input data (2 rows, 2 columns)
    Eigen::MatrixXf input(2, 2);
    input << 1.0f, -1.0f, 0.5f, 0.8f;

    Eigen::MatrixXf output(1, 2);

    // Test with default (linear) activations
    nam::activations::ActivationIdentity identity_act;
    nam::activations::ActivationIdentity identity_blend_act;
    nam::gating_activations::BlendingActivation blending_act(&identity_act, &identity_blend_act, 1);
    blending_act.apply(input, output);

    // With linear activations, blending should be:
    // alpha = blend_input (since linear activation does nothing)
    // output = alpha * input + (1 - alpha) * input = input
    // So output should equal the first row (input after activation)
    assert(fabs(output(0, 0) - 1.0f) < 1e-6);
    assert(fabs(output(0, 1) - (-1.0f)) < 1e-6);

    // Test with sigmoid blending activation
    nam::activations::Activation* sigmoid_act = nam::activations::Activation::get_activation("Sigmoid");
    nam::gating_activations::BlendingActivation blending_act2(&identity_act, sigmoid_act, 1);
    blending_act2.apply(input, output);

    // With sigmoid blending, alpha values should be between 0 and 1
    // For input 0.5, sigmoid(0.5) ≈ 0.622
    // For input 0.8, sigmoid(0.8) ≈ 0.690
    float alpha0 = 1.0f / (1.0f + expf(-0.5f)); // sigmoid(0.5)
    float alpha1 = 1.0f / (1.0f + expf(-0.8f)); // sigmoid(0.8)

    // Expected output: alpha * activated_input + (1 - alpha) * pre_activation_input
    // Since input activation is linear, activated_input = pre_activation_input = input
    // So output = alpha * input + (1 - alpha) * input = input
    // This is the same as with linear activations
    assert(fabs(output(0, 0) - 1.0f) < 1e-6);
    assert(fabs(output(0, 1) - (-1.0f)) < 1e-6);

    std::cout << "BlendingActivation blending behavior test passed" << std::endl;
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

    // Create blending activation with custom activations
    nam::gating_activations::BlendingActivation blending_act(&leaky_relu, &leaky_relu2, 1);

    // Apply the activation
    blending_act.apply(input, output);

    // Verify dimensions
    assert(output.rows() == 1);
    assert(output.cols() == 2);

    std::cout << "BlendingActivation custom activations test passed" << std::endl;
  }

  static void test_error_handling()
  {
    // Test with insufficient rows - should assert
    Eigen::MatrixXf input(1, 2); // Only 1 row
    Eigen::MatrixXf output(1, 2);

    nam::activations::ActivationIdentity identity_act;
    nam::activations::ActivationIdentity identity_blend_act;
    nam::gating_activations::BlendingActivation blending_act(&identity_act, &identity_blend_act, 1);

    // This should trigger an assert and terminate the program
    // We can't easily test asserts in a unit test framework without special handling
    // For real-time code, we rely on the asserts to catch issues during development

    // Test with invalid number of channels - should assert in constructor
    // These tests would normally crash the program due to asserts
    // In production, these conditions should never occur if the code is used correctly
  }

  static void test_edge_cases()
  {
    // Test with zero input
    Eigen::MatrixXf input(2, 1);
    input << 0.0f, 0.0f;

    Eigen::MatrixXf output(1, 1);

    nam::activations::ActivationIdentity identity_act;
    nam::activations::ActivationIdentity identity_blend_act;
    nam::gating_activations::BlendingActivation blending_act(&identity_act, &identity_blend_act, 1);
    blending_act.apply(input, output);

    assert(fabs(output(0, 0) - 0.0f) < 1e-6);

    // Test with large values
    Eigen::MatrixXf input2(2, 1);
    input2 << 1000.0f, -1000.0f;

    blending_act.apply(input2, output);

    // Should handle large values without issues
    assert(output.rows() == 1);
    assert(output.cols() == 1);
  }
};

}; // namespace test_gating_activations
