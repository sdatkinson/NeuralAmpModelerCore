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
class LinearActivation : public nam::activations::Activation
{
public:
  LinearActivation() = default;
  ~LinearActivation() = default;
  // Inherit the default apply methods which do nothing (linear/identity)
};

// Static instance for default activation
static LinearActivation default_activation;

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
  , num_gating_channels(gating_channels)
  {
    if (num_input_channels <= 0 || num_gating_channels <= 0)
    {
      throw std::invalid_argument("GatingActivation: number of channels must be positive");
    }
  }

  ~GatingActivation() = default;

  /**
   * Apply gating activation to input matrix
   * @param input Input matrix with shape (input_channels + gating_channels) x num_samples
   * @param output Output matrix with shape input_channels x num_samples
   */
  void apply(Eigen::MatrixXf& input, Eigen::MatrixXf& output)
  {
    // Validate input dimensions
    const int total_channels = num_input_channels + num_gating_channels;
    if (input.rows() != total_channels)
    {
      throw std::invalid_argument("GatingActivation: input matrix must have " + std::to_string(total_channels)
                                  + " rows");
    }

    // Validate output dimensions
    if (output.rows() != num_input_channels || output.cols() != input.cols())
    {
      throw std::invalid_argument("GatingActivation: output matrix must have " + std::to_string(num_input_channels)
                                  + " rows and " + std::to_string(input.cols()) + " columns");
    }

    // Process column-by-column to ensure memory contiguity (important for column-major matrices)
    const int num_samples = input.cols();
    for (int i = 0; i < num_samples; i++)
    {
      // Apply activation to input channels
      Eigen::MatrixXf input_block = input.block(0, i, num_input_channels, 1);
      input_activation->apply(input_block);

      // Apply activation to gating channels
      Eigen::MatrixXf gating_block = input.block(num_input_channels, i, num_gating_channels, 1);
      gating_activation->apply(gating_block);

      // Element-wise multiplication and store result
      // For wavenet compatibility, we assume one-to-one mapping
      assert(num_input_channels == num_gating_channels);
      output.block(0, i, num_input_channels, 1) = input_block.array() * gating_block.array();
    }
  }

  /**
   * Get the total number of input channels required
   */
  int get_total_input_channels() const { return num_input_channels + num_gating_channels; }

  /**
   * Get the number of output channels
   */
  int get_output_channels() const { return num_input_channels; }

private:
  activations::Activation* input_activation;
  activations::Activation* gating_activation;
  int num_input_channels;
  int num_gating_channels;
};

class BlendingActivation
{
public:
  /**
   * Constructor for BlendingActivation
   * @param input_act Activation function for input channels
   * @param blend_act Activation function for blending channels
   * @param alpha_val Blending factor (0.0 to 1.0)
   * @param input_channels Number of input channels
   * @param blend_channels Number of blending channels
   */
  BlendingActivation(activations::Activation* input_act = nullptr, activations::Activation* blend_act = nullptr,
                     float alpha_val = 0.5f, int input_channels = 1, int blend_channels = 1)
  : input_activation(input_act ? input_act : &default_activation)
  , blending_activation(blend_act ? blend_act : &default_activation)
  , alpha(alpha_val)
  , num_input_channels(input_channels)
  , num_blend_channels(blend_channels)
  {
    // Validate alpha is in valid range
    if (alpha < 0.0f || alpha > 1.0f)
    {
      throw std::invalid_argument("BlendingActivation: alpha must be between 0.0 and 1.0");
    }
    if (num_input_channels <= 0 || num_blend_channels <= 0)
    {
      throw std::invalid_argument("BlendingActivation: number of channels must be positive");
    }
  }

  ~BlendingActivation() = default;

  /**
   * Apply blending activation to input matrix
   * @param input Input matrix with shape (input_channels + blend_channels) x num_samples
   * @param output Output matrix with shape input_channels x num_samples
   */
  void apply(Eigen::MatrixXf& input, Eigen::MatrixXf& output)
  {
    // Validate input dimensions
    const int total_channels = num_input_channels + num_blend_channels;
    if (input.rows() != total_channels)
    {
      throw std::invalid_argument("BlendingActivation: input matrix must have " + std::to_string(total_channels)
                                  + " rows");
    }

    // Validate output dimensions
    if (output.rows() != num_input_channels || output.cols() != input.cols())
    {
      throw std::invalid_argument("BlendingActivation: output matrix must have " + std::to_string(num_input_channels)
                                  + " rows and " + std::to_string(input.cols()) + " columns");
    }

    // Process column-by-column to ensure memory contiguity
    const int num_samples = input.cols();
    for (int i = 0; i < num_samples; i++)
    {
      // Apply activation to input channels
      Eigen::MatrixXf input_block = input.block(0, i, num_input_channels, 1);
      input_activation->apply(input_block);

      // Apply activation to blend channels
      Eigen::MatrixXf blend_block = input.block(num_input_channels, i, num_blend_channels, 1);
      blending_activation->apply(blend_block);

      // Weighted blending
      // For wavenet compatibility, we assume one-to-one mapping
      assert(num_input_channels == num_blend_channels);
      output.block(0, i, num_input_channels, 1) = alpha * input_block + (1.0f - alpha) * blend_block;
    }
  }

  /**
   * Get the total number of input channels required
   */
  int get_total_input_channels() const { return num_input_channels + num_blend_channels; }

  /**
   * Get the number of output channels
   */
  int get_output_channels() const { return num_input_channels; }

private:
  activations::Activation* input_activation;
  activations::Activation* blending_activation;
  float alpha;
  int num_input_channels;
  int num_blend_channels;
};


}; // namespace gating_activations
}; // namespace nam
