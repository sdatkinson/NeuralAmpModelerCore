// Test to verify that our gating implementation matches the wavenet behavior

#include <cassert>
#include <string>
#include <vector>
#include <cmath>
#include <iostream>

#include "NAM/gating_activations.h"
#include "NAM/activations.h"

namespace test_wavenet_gating_compatibility
{

class TestWavenetGatingCompatibility
{
public:
  static void test_wavenet_style_gating()
  {
    // Simulate wavenet scenario: 2 channels (input + gating), multiple samples
    const int channels = 2;
    const int num_samples = 3;

    // Create input matrix similar to wavenet's _z matrix
    // First 'channels' rows are input, next 'channels' rows are gating
    Eigen::MatrixXf input(2 * channels, num_samples);
    input << 1.0f, -0.5f, 0.2f, // Input channel 1
      0.3f, 0.1f, -0.4f, // Input channel 2
      0.8f, 0.6f, 0.9f, // Gating channel 1
      0.4f, 0.2f, 0.7f; // Gating channel 2

    Eigen::MatrixXf output(channels, num_samples);

    // Create gating activation that matches wavenet behavior
    // Wavenet uses: input activation (default/linear) and sigmoid for gating
    nam::gating_activations::GatingActivation gating_act(nullptr, nullptr, channels, channels);

    // Apply the activation
    gating_act.apply(input, output);

    // Verify dimensions
    assert(output.rows() == channels);
    assert(output.cols() == num_samples);

    // Verify that the output is the element-wise product of input and gating channels
    // after applying activations
    for (int c = 0; c < channels; c++)
    {
      for (int s = 0; s < num_samples; s++)
      {
        // Input channel value (no activation applied - linear)
        float input_val = input(c, s);

        // Gating channel value (sigmoid activation applied)
        float gating_val = input(channels + c, s);
        float sigmoid_gating = 1.0f / (1.0f + expf(-gating_val));

        // Expected output
        float expected = input_val * sigmoid_gating;

        // Check if they match
        if (fabs(output(c, s) - expected) > 1e-6)
        {
          std::cerr << "Mismatch at channel " << c << ", sample " << s << ": expected " << expected << ", got "
                    << output(c, s) << std::endl;
          assert(false);
        }
      }
    }

    std::cout << "Wavenet gating compatibility test passed" << std::endl;
  }

  static void test_column_by_column_processing()
  {
    // Test that our implementation processes column-by-column like wavenet
    const int channels = 1;
    const int num_samples = 4;

    Eigen::MatrixXf input(2, num_samples);
    input << 1.0f, 2.0f, 3.0f, 4.0f, 0.1f, 0.2f, 0.3f, 0.4f;

    Eigen::MatrixXf output(channels, num_samples);

    nam::gating_activations::GatingActivation gating_act(nullptr, nullptr, channels, channels);
    gating_act.apply(input, output);

    // Verify each column was processed independently
    for (int s = 0; s < num_samples; s++)
    {
      float input_val = input(0, s);
      float gating_val = input(1, s);
      float sigmoid_gating = 1.0f / (1.0f + expf(-gating_val));
      float expected = input_val * sigmoid_gating;

      assert(fabs(output(0, s) - expected) < 1e-6);
    }

    std::cout << "Column-by-column processing test passed" << std::endl;
  }

  static void test_memory_contiguity()
  {
    // Test that our implementation handles memory contiguity correctly
    // This is important for column-major matrices
    const int channels = 3;
    const int num_samples = 2;

    Eigen::MatrixXf input(2 * channels, num_samples);
    // Fill with some values
    for (int i = 0; i < 2 * channels; i++)
    {
      for (int j = 0; j < num_samples; j++)
      {
        input(i, j) = static_cast<float>(i * num_samples + j + 1);
      }
    }

    Eigen::MatrixXf output(channels, num_samples);

    nam::gating_activations::GatingActivation gating_act(nullptr, nullptr, channels, channels);

    // This should not crash or produce incorrect results due to memory contiguity issues
    gating_act.apply(input, output);

    // Verify the results are correct
    for (int c = 0; c < channels; c++)
    {
      for (int s = 0; s < num_samples; s++)
      {
        float input_val = input(c, s);
        float gating_val = input(channels + c, s);
        float sigmoid_gating = 1.0f / (1.0f + expf(-gating_val));
        float expected = input_val * sigmoid_gating;

        assert(fabs(output(c, s) - expected) < 1e-6);
      }
    }

    std::cout << "Memory contiguity test passed" << std::endl;
  }

  static void test_multiple_channels()
  {
    // Test with multiple equal input and gating channels (wavenet style)
    const int channels = 2;
    const int num_samples = 2;

    Eigen::MatrixXf input(2 * channels, num_samples);
    input << 1.0f, 2.0f, // Input channels
      3.0f, 4.0f, 5.0f, 6.0f, // Gating channels
      7.0f, 8.0f;

    Eigen::MatrixXf output(channels, num_samples);

    nam::gating_activations::GatingActivation gating_act(nullptr, nullptr, channels, channels);
    gating_act.apply(input, output);

    // Verify dimensions
    assert(output.rows() == channels);
    assert(output.cols() == num_samples);

    // Verify that each input channel is multiplied by corresponding gating channel
    for (int c = 0; c < channels; c++)
    {
      for (int s = 0; s < num_samples; s++)
      {
        float input_val = input(c, s);
        float gating_val = input(channels + c, s);
        float sigmoid_gating = 1.0f / (1.0f + expf(-gating_val));
        float expected = input_val * sigmoid_gating;

        assert(fabs(output(c, s) - expected) < 1e-6);
      }
    }

    std::cout << "Multiple channels test passed" << std::endl;
  }
};

}; // namespace test_wavenet_gating_compatibility
