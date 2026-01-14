#pragma once

#include <string>
#include <cmath> // expf
#include <unordered_map>
#include <Eigen/Dense>
#include <functional>
#include <stdexcept>
#include "activations.h"

namespace nam
{
namespace gating_activations
{

// Default linear activation (identity function)
class IdentityActivation : public nam::activations::Activation
{
public:
  IdentityActivation() = default;
  ~IdentityActivation() = default;
  // Inherit the default apply methods which do nothing (linear/identity)
};

// Static instance for default activation
static IdentityActivation default_activation;

class GatingActivation
{
public:
  /**
   * Constructor for GatingActivation
   * @param input_act Activation function for input channels (default: linear)
   * @param gating_act Activation function for gating channels (default: sigmoid)
   * @param input_channels Number of input channels (default: 1)
   * @param gating_channels Number of gating channels (default: 1)
   */
  GatingActivation(activations::Activation* input_act = nullptr, activations::Activation* gating_act = nullptr,
                   int input_channels = 1, int gating_channels = 1)
  : input_activation(input_act ? input_act : &default_activation)
  , gating_activation(gating_act ? gating_act : activations::Activation::get_activation("Sigmoid"))
  , num_input_channels(input_channels)
  {
    assert(num_input_channels > 0);
  }

  ~GatingActivation() = default;

  /**
   * Apply gating activation to input matrix
   * @param input Input matrix with shape (input_channels + gating_channels) x num_samples
   * @param output Output matrix with shape input_channels x num_samples
   */
  void apply(Eigen::MatrixXf& input, Eigen::MatrixXf& output)
  {
    // Validate input dimensions (assert for real-time performance)
    const int total_channels = 2 * num_input_channels;
    assert(input.rows() == total_channels);
    assert(output.rows() == num_input_channels);
    assert(output.cols() == input.cols());

    // Process column-by-column to ensure memory contiguity (important for column-major matrices)
    const int num_samples = input.cols();
    for (int i = 0; i < num_samples; i++)
    {
      // Apply activation to input channels
      Eigen::MatrixXf input_block = input.block(0, i, num_input_channels, 1);
      input_activation->apply(input_block);

      // Apply activation to gating channels
      Eigen::MatrixXf gating_block = input.block(num_input_channels, i, num_input_channels, 1);
      gating_activation->apply(gating_block);

      // Element-wise multiplication and store result
      // For wavenet compatibility, we assume one-to-one mapping
      output.block(0, i, num_input_channels, 1) = input_block.array() * gating_block.array();
    }
  }

  /**
   * Get the total number of input channels required
   */
  int get_total_input_channels() const { return 2 * num_input_channels; }

  /**
   * Get the number of output channels
   */
  int get_output_channels() const { return num_input_channels; }

private:
  activations::Activation* input_activation;
  activations::Activation* gating_activation;
  int num_input_channels;
};

class BlendingActivation
{
public:
  /**
   * Constructor for BlendingActivation
   * @param input_act Activation function for input channels
   * @param blend_act Activation function for blending channels
   * @param input_channels Number of input channels
   */
  BlendingActivation(activations::Activation* input_act = nullptr, activations::Activation* blend_act = nullptr,
                     int input_channels = 1)
  : input_activation(input_act ? input_act : &default_activation)
  , blending_activation(blend_act ? blend_act : &default_activation)
  , num_input_channels(input_channels)
  {
    if (num_input_channels <= 0)
    {
      throw std::invalid_argument("BlendingActivation: number of input channels must be positive");
    }
    // Initialize input buffer with correct size
    // Note: current code copies column-by-column so we only need (num_input_channels, 1)
    input_buffer.resize(num_input_channels, 1);
  }

  ~BlendingActivation() = default;

  /**
   * Apply blending activation to input matrix
   * @param input Input matrix with shape (input_channels + blend_channels) x num_samples
   * @param output Output matrix with shape input_channels x num_samples
   */
  void apply(Eigen::MatrixXf& input, Eigen::MatrixXf& output)
  {
    // Validate input dimensions (assert for real-time performance)
    const int total_channels = num_input_channels * 2; // 2*channels in, channels out
    assert(input.rows() == total_channels);
    assert(output.rows() == num_input_channels);
    assert(output.cols() == input.cols());

    // Process column-by-column to ensure memory contiguity
    const int num_samples = input.cols();
    for (int i = 0; i < num_samples; i++)
    {
      // Store pre-activation input values in buffer
      input_buffer = input.block(0, i, num_input_channels, 1);

      // Apply activation to input channels
      Eigen::MatrixXf input_block = input.block(0, i, num_input_channels, 1);
      input_activation->apply(input_block);

      // Apply activation to blend channels to compute alpha
      Eigen::MatrixXf blend_block = input.block(num_input_channels, i, num_input_channels, 1);
      blending_activation->apply(blend_block);

      // Weighted blending: alpha * activated_input + (1 - alpha) * pre_activation_input
      output.block(0, i, num_input_channels, 1) =
        blend_block.array() * input_block.array() + (1.0f - blend_block.array()) * input_buffer.array();
    }
  }

  /**
   * Get the total number of input channels required
   */
  int get_total_input_channels() const { return 2 * num_input_channels; }

  /**
   * Get the number of output channels
   */
  int get_output_channels() const { return num_input_channels; }

private:
  activations::Activation* input_activation;
  activations::Activation* blending_activation;
  int num_input_channels;
  Eigen::MatrixXf input_buffer;
};


}; // namespace gating_activations
}; // namespace nam
