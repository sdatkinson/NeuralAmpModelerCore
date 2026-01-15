// Detailed test for BlendingActivation behavior

#include <cassert>
#include <string>
#include <vector>
#include <cmath>
#include <iostream>

#include "NAM/gating_activations.h"
#include "NAM/activations.h"

namespace test_blending_detailed
{

class TestBlendingDetailed
{
public:
  static void test_blending_with_different_activations()
  {
    // Test case: 2 input channels, so we need 4 total input channels (2*channels in)
    Eigen::MatrixXf input(4, 2); // 4 rows (2 input + 2 blending), 2 samples
    input << 1.0f, 2.0f, // Input channel 1
      3.0f, 4.0f, // Input channel 2
      0.5f, 0.8f, // Blending channel 1
      0.3f, 0.6f; // Blending channel 2

    Eigen::MatrixXf output(2, 2); // 2 output channels, 2 samples

    // Test with default (linear) activations
    nam::activations::ActivationIdentity identity_act;
    nam::activations::ActivationIdentity identity_blend_act;
    nam::gating_activations::BlendingActivation blending_act(&identity_act, &identity_blend_act, 2);
    blending_act.apply(input, output);

    // With linear activations:
    // alpha = blend_input (since linear activation does nothing)
    // output = alpha * input + (1 - alpha) * input = input
    // So output should equal the input channels after activation (which is the same as input)
    assert(fabs(output(0, 0) - 1.0f) < 1e-6);
    assert(fabs(output(1, 0) - 3.0f) < 1e-6);
    assert(fabs(output(0, 1) - 2.0f) < 1e-6);
    assert(fabs(output(1, 1) - 4.0f) < 1e-6);

    // Test with sigmoid blending activation
    nam::activations::Activation* sigmoid_act = nam::activations::Activation::get_activation("Sigmoid");
    nam::gating_activations::BlendingActivation blending_act_sigmoid(&identity_act, sigmoid_act, 2);

    Eigen::MatrixXf output_sigmoid(2, 2);
    blending_act_sigmoid.apply(input, output_sigmoid);

    // With sigmoid blending, alpha values should be between 0 and 1
    // For blend input 0.5, sigmoid(0.5) ≈ 0.622
    // For blend input 0.8, sigmoid(0.8) ≈ 0.690
    // For blend input 0.3, sigmoid(0.3) ≈ 0.574
    // For blend input 0.6, sigmoid(0.6) ≈ 0.646

    float alpha0_0 = 1.0f / (1.0f + expf(-0.5f)); // sigmoid(0.5)
    float alpha1_0 = 1.0f / (1.0f + expf(-0.8f)); // sigmoid(0.8)
    float alpha0_1 = 1.0f / (1.0f + expf(-0.3f)); // sigmoid(0.3)
    float alpha1_1 = 1.0f / (1.0f + expf(-0.6f)); // sigmoid(0.6)

    // Expected output: alpha * activated_input + (1 - alpha) * pre_activation_input
    // Since input activation is linear, activated_input = pre_activation_input = input
    // So output = alpha * input + (1 - alpha) * input = input
    // This should be the same as with linear activations
    assert(fabs(output_sigmoid(0, 0) - 1.0f) < 1e-6);
    assert(fabs(output_sigmoid(1, 0) - 3.0f) < 1e-6);
    assert(fabs(output_sigmoid(0, 1) - 2.0f) < 1e-6);
    assert(fabs(output_sigmoid(1, 1) - 4.0f) < 1e-6);
  }

  static void test_input_buffer_usage()
  {
    // Test that the input buffer is correctly storing pre-activation values
    Eigen::MatrixXf input(2, 1);
    input << 2.0f, 0.5f;

    Eigen::MatrixXf output(1, 1);

    // Test with ReLU activation on input (which will change values < 0 to 0)
    nam::activations::ActivationReLU relu_act;
    nam::activations::ActivationIdentity identity_act;
    nam::gating_activations::BlendingActivation blending_act(&relu_act, &identity_act, 1);

    blending_act.apply(input, output);

    // With input=2.0, ReLU(2.0)=2.0, blend=0.5
    // output = 0.5 * 2.0 + (1 - 0.5) * 2.0 = 0.5 * 2.0 + 0.5 * 2.0 = 2.0
    assert(fabs(output(0, 0) - 2.0f) < 1e-6);

    // Test with negative input value
    Eigen::MatrixXf input2(2, 1);
    input2 << -1.0f, 0.5f;

    Eigen::MatrixXf output2(1, 1);
    blending_act.apply(input2, output2);

    // With input=-1.0, ReLU(-1.0)=0.0, blend=0.5
    // output = 0.5 * 0.0 + (1 - 0.5) * (-1.0) = 0.0 + 0.5 * (-1.0) = -0.5
    assert(fabs(output2(0, 0) - (-0.5f)) < 1e-6);
  }
};

}; // namespace test_blending_detailed