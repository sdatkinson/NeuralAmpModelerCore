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

class GatingActivation
{
public:
  /**
   * Constructor for GatingActivation
   * @param input_act Activation function for input channels
   * @param gating_act Activation function for gating channels
   * @param input_channels Number of input channels (default: 1)
   * @param gating_channels Number of gating channels (default: 1)
   */
  GatingActivation(activations::Activation::Ptr input_act, activations::Activation::Ptr gating_act,
                   int input_channels = 1)
  : input_activation(input_act)
  , gating_activation(gating_act)
  , num_channels(input_channels)
  {
    if (num_channels <= 0)
    {
      throw std::invalid_argument("GatingActivation: number of input channels must be positive");
    }
    // Initialize buffers with correct size
    // Note: current code copies column-by-column so we only need (num_channels, 1)
    input_buffer.resize(num_channels, 1);
    gating_buffer.resize(num_channels, 1);
  }

  ~GatingActivation() = default;

  /**
   * Apply gating activation to input matrix
   * @param input Input matrix with shape (input_channels + gating_channels) x num_samples
   * @param output Output matrix with shape input_channels x num_samples
   */
  template <typename InputDerived, typename OutputDerived>
  void apply(const Eigen::MatrixBase<InputDerived>& input, Eigen::MatrixBase<OutputDerived>& output)
  {
    // Validate input dimensions (assert for real-time performance)
    assert(input.rows() == 2 * num_channels);
    assert(output.rows() == num_channels);
    assert(output.cols() == input.cols());

    const int num_samples = input.cols();
    const int input_rows = input.rows();  // 2 * num_channels

#ifdef NAM_USE_INLINE_GEMM
    // Optimized path: direct memory access with activation applied per-element
    // Note: output may be a block expression with outer stride != num_channels
    const float* __restrict__ input_ptr = input.derived().data();
    float* __restrict__ output_ptr = output.derived().data();
    const int output_stride = (int)output.outerStride();  // Column stride for output

    for (int f = 0; f < num_samples; f++)
    {
      const float* __restrict__ in_col = input_ptr + f * input_rows;
      float* __restrict__ out_col = output_ptr + f * output_stride;

      // Copy input and gating channels to buffers, apply activations, multiply
      for (int c = 0; c < num_channels; c++)
      {
        input_buffer(c, 0) = in_col[c];
        gating_buffer(c, 0) = in_col[c + num_channels];
      }

      input_activation->apply(input_buffer);
      gating_activation->apply(gating_buffer);

      // Element-wise multiply and store
      for (int c = 0; c < num_channels; c++)
      {
        out_col[c] = input_buffer(c, 0) * gating_buffer(c, 0);
      }
    }
#else
    // Original Eigen path
    for (int i = 0; i < num_samples; i++)
    {
      // Copy to pre-allocated buffers and apply activations in-place
      input_buffer = input.block(0, i, num_channels, 1);
      input_activation->apply(input_buffer);

      gating_buffer = input.block(num_channels, i, num_channels, 1);
      gating_activation->apply(gating_buffer);

      // Element-wise multiplication and store result
      output.block(0, i, num_channels, 1) = input_buffer.array() * gating_buffer.array();
    }
#endif
  }

  /**
   * Get the total number of input channels required
   */
  int get_input_channels() const { return 2 * num_channels; }

  /**
   * Get the number of output channels
   */
  int get_output_channels() const { return num_channels; }

private:
  activations::Activation::Ptr input_activation;
  activations::Activation::Ptr gating_activation;
  int num_channels;
  Eigen::MatrixXf input_buffer;
  Eigen::MatrixXf gating_buffer;
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
  BlendingActivation(activations::Activation::Ptr input_act, activations::Activation::Ptr blend_act,
                     int input_channels = 1)
  : input_activation(input_act)
  , blending_activation(blend_act)
  , num_channels(input_channels)
  {
    assert(num_channels > 0);

    // Initialize buffers with correct size
    // Note: current code copies column-by-column so we only need (num_channels, 1)
    pre_activation_buffer.resize(num_channels, 1);
    input_buffer.resize(num_channels, 1);
    blend_buffer.resize(num_channels, 1);
  }

  ~BlendingActivation() = default;

  /**
   * Apply blending activation to input matrix
   * @param input Input matrix with shape (input_channels + blend_channels) x num_samples
   * @param output Output matrix with shape input_channels x num_samples
   */
  template <typename InputDerived, typename OutputDerived>
  void apply(const Eigen::MatrixBase<InputDerived>& input, Eigen::MatrixBase<OutputDerived>& output)
  {
    // Validate input dimensions (assert for real-time performance)
    assert(input.rows() == num_channels * 2);
    assert(output.rows() == num_channels);
    assert(output.cols() == input.cols());

    const int num_samples = input.cols();
    const int input_rows = input.rows();  // 2 * num_channels

#ifdef NAM_USE_INLINE_GEMM
    // Optimized path: direct memory access
    // Note: output may be a block expression with outer stride != num_channels
    const float* __restrict__ input_ptr = input.derived().data();
    float* __restrict__ output_ptr = output.derived().data();
    const int output_stride = (int)output.outerStride();  // Column stride for output

    for (int f = 0; f < num_samples; f++)
    {
      const float* __restrict__ in_col = input_ptr + f * input_rows;
      float* __restrict__ out_col = output_ptr + f * output_stride;

      // Copy channels to buffers
      for (int c = 0; c < num_channels; c++)
      {
        pre_activation_buffer(c, 0) = in_col[c];
        input_buffer(c, 0) = in_col[c];
        blend_buffer(c, 0) = in_col[c + num_channels];
      }

      // Apply activations
      input_activation->apply(input_buffer);
      blending_activation->apply(blend_buffer);

      // Weighted blending: alpha * activated + (1 - alpha) * pre_activation
      for (int c = 0; c < num_channels; c++)
      {
        const float alpha = blend_buffer(c, 0);
        out_col[c] = alpha * input_buffer(c, 0) + (1.0f - alpha) * pre_activation_buffer(c, 0);
      }
    }
#else
    // Original Eigen path
    for (int i = 0; i < num_samples; i++)
    {
      // Store pre-activation input values in buffer
      pre_activation_buffer = input.block(0, i, num_channels, 1);

      // Copy to pre-allocated buffer and apply activation to input channels
      input_buffer = input.block(0, i, num_channels, 1);
      input_activation->apply(input_buffer);

      // Copy to pre-allocated buffer and apply activation to blend channels to compute alpha
      blend_buffer = input.block(num_channels, i, num_channels, 1);
      blending_activation->apply(blend_buffer);

      // Weighted blending: alpha * activated_input + (1 - alpha) * pre_activation_input
      output.block(0, i, num_channels, 1) =
        blend_buffer.array() * input_buffer.array() + (1.0f - blend_buffer.array()) * pre_activation_buffer.array();
    }
#endif
  }

  /**
   * Get the total number of input channels required
   */
  int get_input_channels() const { return 2 * num_channels; }

  /**
   * Get the number of output channels
   */
  int get_output_channels() const { return num_channels; }

private:
  activations::Activation::Ptr input_activation;
  activations::Activation::Ptr blending_activation;
  int num_channels;
  Eigen::MatrixXf pre_activation_buffer;
  Eigen::MatrixXf input_buffer;
  Eigen::MatrixXf blend_buffer;
};


}; // namespace gating_activations
}; // namespace nam
