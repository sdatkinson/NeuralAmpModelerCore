#pragma once

#include "params.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "../conv1d.h"
#include "../gating_activations.h"
#include "../film.h"

namespace nam
{
namespace wavenet
{
namespace detail
{

/// \brief A single WaveNet layer block
///
/// A WaveNet layer performs the following operations:
/// 1. Dilated convolution on the input (with optional pre/post-FiLM)
/// 2. Input mixin (conditioning input processing, with optional pre/post-FiLM)
/// 3. Sum of conv and input mixin outputs
/// 4. Activation (with optional gating/blending and pre/post FiLM)
/// 5. Optional layer1x1 convolution for the next layer (with optional post-FiLM)
/// 6. Optional 1x1 convolution for the head output (with optional post-FiLM)
/// 7. Residual connection (input + layer1x1 output, or just input if layer1x1 inactive) and skip connection (to next
/// layer)
///
/// The layer supports multiple gating modes and FiLM at various points in the computation.
/// See the walkthrough documentation for detailed step-by-step explanation.
class Layer
{
public:
  /// \brief Constructor with LayerParams
  /// \param params Parameters for constructing the layer
  /// \throws std::invalid_argument If head1x1_post_film_params is active but head1x1 is not, or if layer1x1 is inactive
  /// but bottleneck != channels
  Layer(const LayerParams& params)
  : _conv(params.channels, (params.gating_mode != GatingMode::NONE) ? 2 * params.bottleneck : params.bottleneck,
          params.kernel_size, true, params.dilation, params.groups_input)
  , _input_mixin(params.condition_size,
                 (params.gating_mode != GatingMode::NONE) ? 2 * params.bottleneck : params.bottleneck, false,
                 params.groups_input_mixin)
  , _activation(activations::Activation::get_activation(params.activation_config))
  , _gating_mode(params.gating_mode)
  , _bottleneck(params.bottleneck)
  {
    if (params.layer1x1_params.active)
    {
      _layer1x1 = std::make_unique<Conv1x1>(params.bottleneck, params.channels, true, params.layer1x1_params.groups);
    }
    else
    {
      // Validation: if layer1x1 is inactive, bottleneck must equal channels
      if (params.bottleneck != params.channels)
      {
        throw std::invalid_argument("When layer1x1.active is false, bottleneck (" + std::to_string(params.bottleneck)
                                    + ") must equal channels (" + std::to_string(params.channels) + ")");
      }
      // If there's a post-layer1x1 FiLM but no layer1x1, this is redundant--don't allow it
      if (params._layer1x1_post_film_params.active)
      {
        throw std::invalid_argument("layer1x1_post_film cannot be active when layer1x1 is not active");
      }
    }

    if (params.head1x1_params.active)
    {
      _head1x1 = std::make_unique<Conv1x1>(
        params.bottleneck, params.head1x1_params.out_channels, true, params.head1x1_params.groups);
    }
    else
    {
      // If there's a post-head 1x1 FiLM but no head 1x1, this is redundant--don't allow it
      if (params.head1x1_post_film_params.active)
      {
        throw std::invalid_argument("Do not use post-head 1x1 FiLM if there is no head 1x1");
      }
    }

    // When no head1x1 and no gating, _output_head would be a straight copy of _z.
    // Skip the copy and return _z directly from GetOutputHead().
    _skip_head_copy = !params.head1x1_params.active && params.gating_mode == GatingMode::NONE;

    // Validate & initialize gating/blending activation
    if (params.gating_mode == GatingMode::GATED)
    {
      _gating_activation = std::make_unique<gating_activations::GatingActivation>(
        _activation, activations::Activation::get_activation(params.secondary_activation_config), params.bottleneck);
    }
    else if (params.gating_mode == GatingMode::BLENDED)
    {
      _blending_activation = std::make_unique<gating_activations::BlendingActivation>(
        _activation, activations::Activation::get_activation(params.secondary_activation_config), params.bottleneck);
    }

    // Initialize FiLM objects
    if (params.conv_pre_film_params.active)
    {
      _conv_pre_film = std::make_unique<FiLM>(
        params.condition_size, params.channels, params.conv_pre_film_params.shift, params.conv_pre_film_params.groups);
    }
    if (params.conv_post_film_params.active)
    {
      const int conv_out_channels =
        (params.gating_mode != GatingMode::NONE) ? 2 * params.bottleneck : params.bottleneck;
      _conv_post_film = std::make_unique<FiLM>(params.condition_size, conv_out_channels,
                                               params.conv_post_film_params.shift, params.conv_post_film_params.groups);
    }
    if (params.input_mixin_pre_film_params.active)
    {
      _input_mixin_pre_film =
        std::make_unique<FiLM>(params.condition_size, params.condition_size, params.input_mixin_pre_film_params.shift,
                               params.input_mixin_pre_film_params.groups);
    }
    if (params.input_mixin_post_film_params.active)
    {
      const int input_mixin_out_channels =
        (params.gating_mode != GatingMode::NONE) ? 2 * params.bottleneck : params.bottleneck;
      _input_mixin_post_film =
        std::make_unique<FiLM>(params.condition_size, input_mixin_out_channels,
                               params.input_mixin_post_film_params.shift, params.input_mixin_post_film_params.groups);
    }
    if (params.activation_pre_film_params.active)
    {
      const int z_channels = (params.gating_mode != GatingMode::NONE) ? 2 * params.bottleneck : params.bottleneck;
      _activation_pre_film =
        std::make_unique<FiLM>(params.condition_size, z_channels, params.activation_pre_film_params.shift,
                               params.activation_pre_film_params.groups);
    }
    if (params.activation_post_film_params.active)
    {
      _activation_post_film =
        std::make_unique<FiLM>(params.condition_size, params.bottleneck, params.activation_post_film_params.shift,
                               params.activation_post_film_params.groups);
    }
    if (params._layer1x1_post_film_params.active && params.layer1x1_params.active)
    {
      _layer1x1_post_film =
        std::make_unique<FiLM>(params.condition_size, params.channels, params._layer1x1_post_film_params.shift,
                               params._layer1x1_post_film_params.groups);
    }
    if (params.head1x1_post_film_params.active && params.head1x1_params.active)
    {
      _head1x1_post_film =
        std::make_unique<FiLM>(params.condition_size, params.head1x1_params.out_channels,
                               params.head1x1_post_film_params.shift, params.head1x1_post_film_params.groups);
    }
  };

  /// \brief Resize all arrays to be able to process maxBufferSize frames
  /// \param maxBufferSize Maximum number of frames to process in a single call
  void SetMaxBufferSize(const int maxBufferSize);

  /// \brief Set the parameters (weights) of this module
  /// \param weights Iterator to the weights vector. Will be advanced as weights are consumed.
  void set_weights_(std::vector<float>::iterator& weights);

  /// \brief Process a block of frames
  ///
  /// Performs the complete layer computation:
  /// 1. Input convolution (with optional pre/post-FiLM)
  /// 2. Input mixin processing (with optional pre/post-FiLM)
  /// 3. Sum and activation (with optional gating/blending and pre/post-FiLM)
  /// 4. Optional layer1x1 convolution toward the skip connection for next layer (with optional post-FiLM)
  /// 5. Optional 1x1 convolution for the head output (with optional post-FiLM)
  /// 6. Store outputs for next layer and the layer array head
  ///
  /// \param input Input from previous layer (channels x num_frames)
  /// \param condition Conditioning input (condition_size x num_frames)
  /// \param num_frames Number of frames to process
  ///
  /// Outputs are stored internally and accessible via GetOutputNextLayer() and GetOutputHead().
  /// Only the first num_frames columns of the output buffers are valid.
  void Process(const Eigen::MatrixXf& input, const Eigen::MatrixXf& condition, const int num_frames);

  /// \brief Get the number of channels expected as input/output from this layer
  /// \return Number of channels
  long get_channels() const { return this->_conv.get_in_channels(); };

  /// \brief Get the dilation of the input convolution layer
  /// \return Dilation factor
  int get_dilation() const { return this->_conv.get_dilation(); };

  /// \brief Get the kernel size of the input convolution layer
  /// \return Kernel size
  long get_kernel_size() const { return this->_conv.get_kernel_size(); };

  /// \brief Get output to next layer (residual connection: input + layer1x1 output)
  ///
  /// Returns the full pre-allocated buffer; only the first num_frames columns
  /// are valid for a given processing call. Slice with .leftCols(num_frames) as needed.
  /// \return Reference to the output buffer (channels x maxBufferSize)
  Eigen::MatrixXf& GetOutputNextLayer() { return this->_output_next_layer; }

  /// \brief Get output to next layer (const version)
  /// \return Const reference to the output buffer
  const Eigen::MatrixXf& GetOutputNextLayer() const { return this->_output_next_layer; }

  /// \brief Get output to head (skip connection: activated conv output)
  ///
  /// Returns the full pre-allocated buffer; only the first num_frames columns
  /// are valid for a given processing call. Slice with .leftCols(num_frames) as needed.
  /// When _skip_head_copy is true (no head1x1, no gating), returns _z directly
  /// to avoid a redundant memcpy.
  /// \return Reference to the head output buffer
  Eigen::MatrixXf& GetOutputHead() { return _skip_head_copy ? this->_z : this->_output_head; }

  /// \brief Get output to head (const version)
  /// \return Const reference to the head output buffer
  const Eigen::MatrixXf& GetOutputHead() const { return _skip_head_copy ? this->_z : this->_output_head; }

  /// \brief Access Conv1D for Reset() propagation (needed for LayerArray)
  /// \return Reference to the internal Conv1D object
  Conv1D& get_conv() { return _conv; }

  /// \brief Access Conv1D (const version)
  /// \return Const reference to the internal Conv1D object
  const Conv1D& get_conv() const { return _conv; }

private:
  // The dilated convolution at the front of the block
  Conv1D _conv;
  // Input mixin
  Conv1x1 _input_mixin;
  // The post-activation layer1x1 convolution (optional)
  std::unique_ptr<Conv1x1> _layer1x1;
  // The post-activation 1x1 convolution outputting to the head, optional
  std::unique_ptr<Conv1x1> _head1x1;
  // The internal state
  Eigen::MatrixXf _z;
  // Output to next layer (residual connection: input + layer1x1 output, or just input if layer1x1 inactive)
  Eigen::MatrixXf _output_next_layer;
  // Output to head (skip connection: activated conv output)
  Eigen::MatrixXf _output_head;

  activations::Activation::Ptr _activation;
  const GatingMode _gating_mode;
  const int _bottleneck; // Internal channel count (not doubled when gated)
  bool _skip_head_copy = false; // When true, GetOutputHead() returns _z directly (no head1x1, no gating)

  // Gating/blending activation objects
  std::unique_ptr<gating_activations::GatingActivation> _gating_activation;
  std::unique_ptr<gating_activations::BlendingActivation> _blending_activation;

  // FiLM objects for feature-wise linear modulation
  std::unique_ptr<FiLM> _conv_pre_film;
  std::unique_ptr<FiLM> _conv_post_film;
  std::unique_ptr<FiLM> _input_mixin_pre_film;
  std::unique_ptr<FiLM> _input_mixin_post_film;
  std::unique_ptr<FiLM> _activation_pre_film;
  std::unique_ptr<FiLM> _activation_post_film;
  std::unique_ptr<FiLM> _layer1x1_post_film;
  std::unique_ptr<FiLM> _head1x1_post_film;
};

/// \brief An array of layers with the same channels, kernel sizes, and activations
///
/// A LayerArray chains multiple Layer objects together, processing them sequentially.
/// Each layer processes the output of the previous layer (residual connection).
/// All layers contribute to a shared head output (skip connection) that is accumulated
/// and then projected to the final head size.
///
/// The LayerArray handles:
/// - Input projection to match layer channel count
/// - Processing layers in sequence with residual connections
/// - Accumulating head outputs from all layers
/// - Projecting the accumulated head output to the final head size
class LayerArray
{
public:
  /// \brief Constructor with LayerArrayParams
  /// \param params Parameters for constructing the layer array
  LayerArray(const LayerArrayParams& params);

  /// \brief Resize all arrays to be able to process maxBufferSize frames
  /// \param maxBufferSize Maximum number of frames to process in a single call
  void SetMaxBufferSize(const int maxBufferSize);

  /// \brief Process without a given previous head input (first layer array)
  ///
  /// Zeros head accumulated output before proceeding. Used for the first layer array in a WaveNet.
  /// \param layer_inputs Input to the layer array (input_size x num_frames)
  /// \param condition Conditioning input (condition_size x num_frames)
  /// \param num_frames Number of frames to process
  void Process(const Eigen::MatrixXf& layer_inputs, const Eigen::MatrixXf& condition, const int num_frames);

  /// \brief Process with a given previous head input (subsequent layer arrays)
  ///
  /// Copies head input before proceeding. Used for subsequent layer arrays that accumulate
  /// head outputs from previous arrays.
  /// \param layer_inputs Input to the layer array (input_size x num_frames)
  /// \param condition Conditioning input (condition_size x num_frames)
  /// \param head_inputs Head input from previous layer array (head_input_size x num_frames)
  /// \param num_frames Number of frames to process
  void Process(const Eigen::MatrixXf& layer_inputs, const Eigen::MatrixXf& condition,
               const Eigen::MatrixXf& head_inputs, const int num_frames);

  /// \brief Get output from last layer (for next layer array)
  ///
  /// Returns the full pre-allocated buffer; only the first num_frames columns
  /// are valid for a given processing call. Slice with .leftCols(num_frames) as needed.
  /// \return Reference to the layer output buffer (channels x maxBufferSize)
  Eigen::MatrixXf& GetLayerOutputs() { return this->_layer_outputs; }

  /// \brief Get output from last layer (const version)
  /// \return Const reference to the layer output buffer
  const Eigen::MatrixXf& GetLayerOutputs() const { return this->_layer_outputs; }

  /// \brief Get head outputs (post head-rechannel)
  ///
  /// Returns the full pre-allocated buffer; only the first num_frames columns
  /// are valid for a given processing call. Slice with .leftCols(num_frames) as needed.
  /// \return Reference to the head output buffer (head_size x maxBufferSize)
  Eigen::MatrixXf& GetHeadOutputs();

  /// \brief Get head outputs (const version)
  /// \return Const reference to the head output buffer
  const Eigen::MatrixXf& GetHeadOutputs() const;

  /// \brief Set the parameters (weights) of this module
  /// \param it Iterator to the weights vector. Will be advanced as weights are consumed.
  void set_weights_(std::vector<float>::iterator& it);

  /// \brief Get the "zero-indexed" receptive field
  ///
  /// The receptive field is the number of input samples that affect the output.
  /// A 1x1 convolution is defined to have a zero-indexed receptive field of zero.
  /// \return Receptive field size
  long get_receptive_field() const;

private:
  // The rechannel before the layers
  Conv1x1 _rechannel;

  // The layer objects
  std::vector<Layer> _layers;
  // Output from last layer (for next layer array)
  Eigen::MatrixXf _layer_outputs;
  // Accumulated head inputs from all layers
  // Size is _head_output_size (= head1x1.out_channels if head1x1 active, else bottleneck)
  Eigen::MatrixXf _head_inputs;

  // Rechannel for the head (_head_output_size -> head_size), causal Conv1D (dilation 1)
  Conv1D _head_rechannel;

  // Head output size from each layer (head1x1.out_channels if active, else bottleneck)
  const int _head_output_size;

  long _get_channels() const;
  // Common processing logic after head inputs are set
  void ProcessInner(const Eigen::MatrixXf& layer_inputs, const Eigen::MatrixXf& condition, const int num_frames);
};

/// \brief Post-stack head: repeated (activation → Conv1D) with dilation 1, stride 1, valid (causal streaming) conv.
class Head
{
public:
  explicit Head(const HeadParams& params);

  void set_weights_(std::vector<float>::iterator& weights);
  void SetMaxBufferSize(int maxBufferSize);
  long receptive_field() const;
  int in_channels() const { return _in_channels; }
  int out_channels() const { return _out_channels; }

  /// \param work Input buffer (in_channels × maxBufferSize); first in_channels×num_frames scaled by head_scale;
  ///             may be modified in place.
  void process(Eigen::MatrixXf& work, int num_frames);

  const Eigen::MatrixXf& get_last_output() const { return _convs.back().GetOutput(); }

private:
  std::vector<nam::Conv1D> _convs;
  std::vector<nam::activations::Activation::Ptr> _activations;
  int _in_channels;
  int _out_channels;
};

} // namespace detail
} // namespace wavenet
} // namespace nam
