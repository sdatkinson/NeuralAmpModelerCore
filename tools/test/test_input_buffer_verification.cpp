// Test to verify that input buffer correctly stores pre-activation values

#include <cassert>
#include <string>
#include <vector>
#include <cmath>
#include <iostream>

#include "NAM/gating_activations.h"
#include "NAM/activations.h"

namespace test_input_buffer_verification
{

class TestInputBufferVerification
{
public:
  static void test_buffer_stores_pre_activation_values()
  {
    // Create a test case where input activation changes the values
    Eigen::MatrixXf input(2, 1);
    input << -2.0f, 0.5f; // Negative input value

    Eigen::MatrixXf output(1, 1);

    // Use ReLU activation which will set negative values to 0
    nam::activations::ActivationReLU relu_act;
    nam::activations::ActivationIdentity identity_act;
    nam::gating_activations::BlendingActivation blending_act(&relu_act, &identity_act, 1);

    // Apply the activation
    blending_act.apply(input, output);

    // Expected behavior:
    // 1. Store pre-activation input in buffer: input_buffer = -2.0f
    // 2. Apply ReLU to input: activated_input = max(-2.0f, 0) = 0.0f
    // 3. Apply linear activation to blend: alpha = 0.5f (no change)
    // 4. Compute output: alpha * activated_input + (1 - alpha) * input_buffer
    //    = 0.5f * 0.0f + 0.5f * (-2.0f) = -1.0f

    float expected = 0.5f * 0.0f + 0.5f * (-2.0f); // = -1.0f
    assert(fabs(output(0, 0) - expected) < 1e-6);
  }

  static void test_buffer_with_different_activations()
  {
    // Test with LeakyReLU which modifies negative values differently
    Eigen::MatrixXf input(2, 1);
    input << -1.0f, 0.8f;

    Eigen::MatrixXf output(1, 1);

    // Use LeakyReLU with slope 0.1
    nam::activations::ActivationLeakyReLU leaky_relu(0.1f);
    nam::activations::ActivationIdentity identity_act;
    nam::gating_activations::BlendingActivation blending_act(&leaky_relu, &identity_act, 1);

    blending_act.apply(input, output);

    // Expected behavior:
    // 1. Store pre-activation input in buffer: input_buffer = -1.0f
    // 2. Apply LeakyReLU: activated_input = (-1.0f > 0) ? -1.0f : 0.1f * -1.0f = -0.1f
    // 3. Apply linear activation to blend: alpha = 0.8f
    // 4. Compute output: alpha * activated_input + (1 - alpha) * input_buffer
    //    = 0.8f * (-0.1f) + 0.2f * (-1.0f) = -0.08f - 0.2f = -0.28f

    float activated_input = (-1.0f > 0) ? -1.0f : 0.1f * -1.0f; // = -0.1f
    float expected = 0.8f * activated_input + 0.2f * (-1.0f); // = -0.28f

    assert(fabs(output(0, 0) - expected) < 1e-6);
  }
};

}; // namespace test_input_buffer_verification