#pragma once

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "activations.h"
#include "conv1d.h"
#include "dsp.h"
#include "gating_activations.h"
#include "film.h"
#include "json.hpp"

namespace nam
{
namespace wavenet
{

/// \brief Gating mode for WaveNet layers
///
/// Determines how the layer processes the doubled bottleneck channels when gating is enabled.
enum class GatingMode
{
  NONE, ///< No gating or blending - standard activation
  GATED, ///< Traditional gating (element-wise multiplication of activated channels)
  BLENDED ///< Blending (weighted average between activated and pre-activated values)
};

/// \brief Helper function for backward compatibility with boolean gated parameter
/// \param gated Boolean indicating whether gating should be enabled
/// \return GatingMode::GATED if gated is true, GatingMode::NONE otherwise
inline GatingMode gating_mode_from_bool(bool gated)
{
  return gated ? GatingMode::GATED : GatingMode::NONE;
}

/// \brief Parameters for head1x1 configuration
///
/// Configures an optional 1x1 convolution that outputs directly to the head (skip connection)
/// instead of using the activation output directly.
struct Head1x1Params
{
  /// \brief Constructor
  /// \param active_ Whether the head1x1 convolution is active
  /// \param out_channels_ Number of output channels for the head1x1 convolution
  /// \param groups_ Number of groups for grouped convolution
  Head1x1Params(bool active_, int out_channels_, int groups_)
  : active(active_)
  , out_channels(out_channels_)
  , groups(groups_)
  {
  }

  const bool active; ///< Whether the head1x1 convolution is active
  const int out_channels; ///< Number of output channels
  const int groups; ///< Number of groups for grouped convolution
};

/// \brief Parameters for FiLM (Feature-wise Linear Modulation) configuration
///
/// FiLM applies per-channel scaling and optional shifting based on conditioning input.
struct _FiLMParams
{
  /// \brief Constructor
  /// \param active_ Whether FiLM is active at this location
  /// \param shift_ Whether to apply both scale and shift (true) or only scale (false)
  _FiLMParams(bool active_, bool shift_)
  : active(active_)
  , shift(shift_)
  {
  }
  const bool active; ///< Whether FiLM is active
  const bool shift; ///< Whether to apply shift in addition to scale
};

/// \brief A single WaveNet layer block
///
/// A WaveNet layer performs the following operations:
/// 1. Dilated convolution on the input (with optional pre/post-FiLM)
/// 2. Input mixin (conditioning input processing, with optional pre/post-FiLM)
/// 3. Sum of conv and input mixin outputs
/// 4. Activation (with optional gating/blending and pre/post FiLM)
/// 5. 1x1 convolution for the next layer (with optional post-FiLM)
/// 6. Optional 1x1 convolution for the head output (with optional post-FiLM)
/// 7. Residual connection (input + 1x1 output) and skip connection (to next layer)
///
/// The layer supports multiple gating modes and FiLM at various points in the computation.
/// See the walkthrough documentation for detailed step-by-step explanation.
class _Layer
{
public:
  /// \brief Constructor with GatingMode enum and typed ActivationConfig
  /// \param condition_size Size of the conditioning input
  /// \param channels Number of input/output channels from layer to layer
  /// \param bottleneck Internal channel count
  /// \param kernel_size Kernel size for the dilated convolution
  /// \param dilation Dilation factor for the convolution
  /// \param activation_config Primary activation function configuration
  /// \param gating_mode Gating mode (NONE, GATED, or BLENDED)
  /// \param groups_input Number of groups for the input convolution
  /// \param groups_input_mixin Number of groups for the input mixin convolution
  /// \param groups_1x1 Number of groups for the 1x1 convolution
  /// \param head1x1_params Configuration of the optional head1x1 convolution
  /// \param secondary_activation_config Secondary activation (for gating/blending)
  /// \param conv_pre_film_params FiLM parameters before the input convolution
  /// \param conv_post_film_params FiLM parameters after the input convolution
  /// \param input_mixin_pre_film_params FiLM parameters before the input mixin
  /// \param input_mixin_post_film_params FiLM parameters after the input mixin
  /// \param activation_pre_film_params FiLM parameters after the input/mixin summed output before activation
  /// \param activation_post_film_params FiLM parameters after the activation output before the 1x1 convolution
  /// \param _1x1_post_film_params FiLM parameters after the 1x1 convolution
  /// \param head1x1_post_film_params FiLM parameters after the head1x1 convolution
  /// \throws std::invalid_argument If head1x1_post_film_params is active but head1x1 is not
  _Layer(const int condition_size, const int channels, const int bottleneck, const int kernel_size, const int dilation,
         const activations::ActivationConfig& activation_config, const GatingMode gating_mode, const int groups_input,
         const int groups_input_mixin, const int groups_1x1, const Head1x1Params& head1x1_params,
         const activations::ActivationConfig& secondary_activation_config, const _FiLMParams& conv_pre_film_params,
         const _FiLMParams& conv_post_film_params, const _FiLMParams& input_mixin_pre_film_params,
         const _FiLMParams& input_mixin_post_film_params, const _FiLMParams& activation_pre_film_params,
         const _FiLMParams& activation_post_film_params, const _FiLMParams& _1x1_post_film_params,
         const _FiLMParams& head1x1_post_film_params)
  : _conv(channels, (gating_mode != GatingMode::NONE) ? 2 * bottleneck : bottleneck, kernel_size, true, dilation,
          groups_input)
  , _input_mixin(
      condition_size, (gating_mode != GatingMode::NONE) ? 2 * bottleneck : bottleneck, false, groups_input_mixin)
  , _1x1(bottleneck, channels, true, groups_1x1)
  , _activation(activations::Activation::get_activation(activation_config))
  , _gating_mode(gating_mode)
  , _bottleneck(bottleneck)
  {
    if (head1x1_params.active)
    {
      _head1x1 = std::make_unique<Conv1x1>(bottleneck, head1x1_params.out_channels, true, head1x1_params.groups);
    }
    else
    {
      // If there's a post-head 1x1 FiLM but no head 1x1, this is redundant--don't allow it
      if (head1x1_post_film_params.active)
      {
        throw std::invalid_argument("Do not use post-head 1x1 FiLM if there is no head 1x1");
      }
    }

    // Validate & initialize gating/blending activation
    if (gating_mode == GatingMode::GATED)
    {
      _gating_activation = std::make_unique<gating_activations::GatingActivation>(
        _activation, activations::Activation::get_activation(secondary_activation_config), bottleneck);
    }
    else if (gating_mode == GatingMode::BLENDED)
    {
      _blending_activation = std::make_unique<gating_activations::BlendingActivation>(
        _activation, activations::Activation::get_activation(secondary_activation_config), bottleneck);
    }

    // Initialize FiLM objects
    if (conv_pre_film_params.active)
    {
      _conv_pre_film = std::make_unique<FiLM>(condition_size, channels, conv_pre_film_params.shift);
    }
    if (conv_post_film_params.active)
    {
      const int conv_out_channels = (gating_mode != GatingMode::NONE) ? 2 * bottleneck : bottleneck;
      _conv_post_film = std::make_unique<FiLM>(condition_size, conv_out_channels, conv_post_film_params.shift);
    }
    if (input_mixin_pre_film_params.active)
    {
      _input_mixin_pre_film = std::make_unique<FiLM>(condition_size, condition_size, input_mixin_pre_film_params.shift);
    }
    if (input_mixin_post_film_params.active)
    {
      const int input_mixin_out_channels = (gating_mode != GatingMode::NONE) ? 2 * bottleneck : bottleneck;
      _input_mixin_post_film =
        std::make_unique<FiLM>(condition_size, input_mixin_out_channels, input_mixin_post_film_params.shift);
    }
    if (activation_pre_film_params.active)
    {
      const int z_channels = (gating_mode != GatingMode::NONE) ? 2 * bottleneck : bottleneck;
      _activation_pre_film = std::make_unique<FiLM>(condition_size, z_channels, activation_pre_film_params.shift);
    }
    if (activation_post_film_params.active)
    {
      _activation_post_film = std::make_unique<FiLM>(condition_size, bottleneck, activation_post_film_params.shift);
    }
    if (_1x1_post_film_params.active)
    {
      _1x1_post_film = std::make_unique<FiLM>(condition_size, channels, _1x1_post_film_params.shift);
    }
    if (head1x1_post_film_params.active && head1x1_params.active)
    {
      _head1x1_post_film =
        std::make_unique<FiLM>(condition_size, head1x1_params.out_channels, head1x1_post_film_params.shift);
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
  /// 4. 1x1 convolution toward the skip connection for next layer (with optional post-FiLM)
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

  /// \brief Get output to next layer (residual connection: input + _1x1 output)
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
  /// \return Reference to the head output buffer
  Eigen::MatrixXf& GetOutputHead() { return this->_output_head; }

  /// \brief Get output to head (const version)
  /// \return Const reference to the head output buffer
  const Eigen::MatrixXf& GetOutputHead() const { return this->_output_head; }

  /// \brief Access Conv1D for Reset() propagation (needed for _LayerArray)
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
  // The post-activation 1x1 convolution
  Conv1x1 _1x1;
  // The post-activation 1x1 convolution outputting to the head, optional
  std::unique_ptr<Conv1x1> _head1x1;
  // The internal state
  Eigen::MatrixXf _z;
  // Output to next layer (residual connection: input + _1x1 output)
  Eigen::MatrixXf _output_next_layer;
  // Output to head (skip connection: activated conv output)
  Eigen::MatrixXf _output_head;

  activations::Activation::Ptr _activation;
  const GatingMode _gating_mode;
  const int _bottleneck; // Internal channel count (not doubled when gated)

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
  std::unique_ptr<FiLM> _1x1_post_film;
  std::unique_ptr<FiLM> _head1x1_post_film;
};

/// \brief Parameters for constructing a LayerArray
///
/// Contains all configuration needed to construct a _LayerArray with multiple layers
/// sharing the same channel count and kernel size. Each layer can have its own activation configuration.
class LayerArrayParams
{
public:
  /// \brief Constructor
  /// \param input_size_ Input size (number of channels) to the layer array
  /// \param condition_size_ Size of the conditioning input
  /// \param head_size_ Size of the head output (after head rechannel)
  /// \param channels_ Number of channels in each layer
  /// \param bottleneck_ Bottleneck size (internal channel count)
  /// \param kernel_size_ Kernel size for dilated convolutions
  /// \param dilations_ Vector of dilation factors, one per layer
  /// \param activation_configs_ Vector of primary activation configurations, one per layer
  /// \param gating_mode_ Gating mode for all layers
  /// \param head_bias_ Whether to use bias in the head rechannel
  /// \param groups_input Number of groups for input convolutions
  /// \param groups_input_mixin_ Number of groups for input mixin convolutions
  /// \param groups_1x1_ Number of groups for 1x1 convolutions
  /// \param head1x1_params_ Parameters for optional head1x1 convolutions
  /// \param secondary_activation_config_ Secondary activation for gating/blending
  /// \param conv_pre_film_params_ FiLM parameters before input convolutions
  /// \param conv_post_film_params_ FiLM parameters after input convolutions
  /// \param input_mixin_pre_film_params_ FiLM parameters before input mixin
  /// \param input_mixin_post_film_params_ FiLM parameters after input mixin
  /// \param activation_pre_film_params_ FiLM parameters before activation
  /// \param activation_post_film_params_ FiLM parameters after activation
  /// \param _1x1_post_film_params_ FiLM parameters after 1x1 convolutions
  /// \param head1x1_post_film_params_ FiLM parameters after head1x1 convolutions
  /// \throws std::invalid_argument If dilations and activation_configs sizes don't match
  LayerArrayParams(const int input_size_, const int condition_size_, const int head_size_, const int channels_,
                   const int bottleneck_, const int kernel_size_, const std::vector<int>&& dilations_,
                   const std::vector<activations::ActivationConfig>&& activation_configs_,
                   const GatingMode gating_mode_, const bool head_bias_, const int groups_input,
                   const int groups_input_mixin_, const int groups_1x1_, const Head1x1Params& head1x1_params_,
                   const activations::ActivationConfig& secondary_activation_config_,
                   const _FiLMParams& conv_pre_film_params_, const _FiLMParams& conv_post_film_params_,
                   const _FiLMParams& input_mixin_pre_film_params_, const _FiLMParams& input_mixin_post_film_params_,
                   const _FiLMParams& activation_pre_film_params_, const _FiLMParams& activation_post_film_params_,
                   const _FiLMParams& _1x1_post_film_params_, const _FiLMParams& head1x1_post_film_params_)
  : input_size(input_size_)
  , condition_size(condition_size_)
  , head_size(head_size_)
  , channels(channels_)
  , bottleneck(bottleneck_)
  , kernel_size(kernel_size_)
  , dilations(std::move(dilations_))
  , activation_configs(std::move(activation_configs_))
  , gating_mode(gating_mode_)
  , head_bias(head_bias_)
  , groups_input(groups_input)
  , groups_input_mixin(groups_input_mixin_)
  , groups_1x1(groups_1x1_)
  , head1x1_params(head1x1_params_)
  , secondary_activation_config(secondary_activation_config_)
  , conv_pre_film_params(conv_pre_film_params_)
  , conv_post_film_params(conv_post_film_params_)
  , input_mixin_pre_film_params(input_mixin_pre_film_params_)
  , input_mixin_post_film_params(input_mixin_post_film_params_)
  , activation_pre_film_params(activation_pre_film_params_)
  , activation_post_film_params(activation_post_film_params_)
  , _1x1_post_film_params(_1x1_post_film_params_)
  , head1x1_post_film_params(head1x1_post_film_params_)
  {
    if (dilations.size() != activation_configs.size())
    {
      throw std::invalid_argument("LayerArrayParams: dilations size (" + std::to_string(dilations.size())
                                  + ") must match activation_configs size (" + std::to_string(activation_configs.size())
                                  + ")");
    }
  }

  const int input_size; ///< Input size (number of channels)
  const int condition_size; ///< Size of conditioning input
  const int head_size; ///< Size of head output (after rechannel)
  const int channels; ///< Number of channels in each layer
  const int bottleneck; ///< Bottleneck size (internal channel count)
  const int kernel_size; ///< Kernel size for dilated convolutions
  std::vector<int> dilations; ///< Dilation factors, one per layer
  std::vector<activations::ActivationConfig> activation_configs; ///< Primary activation configurations, one per layer
  const GatingMode gating_mode; ///< Gating mode for all layers
  const bool head_bias; ///< Whether to use bias in head rechannel
  const int groups_input; ///< Number of groups for input convolutions
  const int groups_input_mixin; ///< Number of groups for input mixin
  const int groups_1x1; ///< Number of groups for 1x1 convolutions
  const Head1x1Params head1x1_params; ///< Parameters for optional head1x1
  const activations::ActivationConfig secondary_activation_config; ///< Secondary activation for gating/blending
  const _FiLMParams conv_pre_film_params; ///< FiLM params before input conv
  const _FiLMParams conv_post_film_params; ///< FiLM params after input conv
  const _FiLMParams input_mixin_pre_film_params; ///< FiLM params before input mixin
  const _FiLMParams input_mixin_post_film_params; ///< FiLM params after input mixin
  const _FiLMParams activation_pre_film_params; ///< FiLM params before activation
  const _FiLMParams activation_post_film_params; ///< FiLM params after activation
  const _FiLMParams _1x1_post_film_params; ///< FiLM params after 1x1 conv
  const _FiLMParams head1x1_post_film_params; ///< FiLM params after head1x1 conv
};

/// \brief An array of layers with the same channels, kernel sizes, and activations
///
/// A LayerArray chains multiple _Layer objects together, processing them sequentially.
/// Each layer processes the output of the previous layer (residual connection).
/// All layers contribute to a shared head output (skip connection) that is accumulated
/// and then projected to the final head size.
///
/// The LayerArray handles:
/// - Input projection to match layer channel count
/// - Processing layers in sequence with residual connections
/// - Accumulating head outputs from all layers
/// - Projecting the accumulated head output to the final head size
class _LayerArray
{
public:
  /// \brief Constructor with GatingMode enum and typed ActivationConfig
  /// \param input_size Input size (number of channels) to the layer array
  /// \param condition_size Size of the conditioning input
  /// \param head_size Size of the head output (after head rechannel)
  /// \param channels Number of channels in each layer
  /// \param bottleneck Bottleneck size (internal channel count)
  /// \param kernel_size Kernel size for dilated convolutions
  /// \param dilations Vector of dilation factors, one per layer
  /// \param activation_configs Vector of primary activation configurations, one per layer
  /// \param gating_mode Gating mode for all layers
  /// \param head_bias Whether to use bias in the head rechannel
  /// \param groups_input Number of groups for input convolutions
  /// \param groups_input_mixin Number of groups for input mixin
  /// \param groups_1x1 Number of groups for 1x1 convolutions
  /// \param head1x1_params Parameters for optional head1x1 convolutions
  /// \param secondary_activation_config Secondary activation for gating/blending
  /// \param conv_pre_film_params FiLM parameters before input convolutions
  /// \param conv_post_film_params FiLM parameters after input convolutions
  /// \param input_mixin_pre_film_params FiLM parameters before input mixin
  /// \param input_mixin_post_film_params FiLM parameters after input mixin
  /// \param activation_pre_film_params FiLM parameters before activation
  /// \param activation_post_film_params FiLM parameters after activation
  /// \param _1x1_post_film_params FiLM parameters after 1x1 convolutions
  /// \param head1x1_post_film_params FiLM parameters after head1x1 convolutions
  _LayerArray(const int input_size, const int condition_size, const int head_size, const int channels,
              const int bottleneck, const int kernel_size, const std::vector<int>& dilations,
              const std::vector<activations::ActivationConfig>& activation_configs, const GatingMode gating_mode,
              const bool head_bias, const int groups_input, const int groups_input_mixin, const int groups_1x1,
              const Head1x1Params& head1x1_params, const activations::ActivationConfig& secondary_activation_config,
              const _FiLMParams& conv_pre_film_params, const _FiLMParams& conv_post_film_params,
              const _FiLMParams& input_mixin_pre_film_params, const _FiLMParams& input_mixin_post_film_params,
              const _FiLMParams& activation_pre_film_params, const _FiLMParams& activation_post_film_params,
              const _FiLMParams& _1x1_post_film_params, const _FiLMParams& head1x1_post_film_params);

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
  std::vector<_Layer> _layers;
  // Output from last layer (for next layer array)
  Eigen::MatrixXf _layer_outputs;
  // Accumulated head inputs from all layers
  // Size is _head_output_size (= head1x1.out_channels if head1x1 active, else bottleneck)
  Eigen::MatrixXf _head_inputs;

  // Rechannel for the head (_head_output_size -> head_size)
  Conv1x1 _head_rechannel;

  // Head output size from each layer (head1x1.out_channels if active, else bottleneck)
  const int _head_output_size;

  long _get_channels() const;
  // Common processing logic after head inputs are set
  void ProcessInner(const Eigen::MatrixXf& layer_inputs, const Eigen::MatrixXf& condition, const int num_frames);
};

/// \brief The main WaveNet model
///
/// WaveNet is a dilated convolutional neural network architecture for audio processing.
/// It consists of multiple LayerArrays, each containing multiple layers with increasing
/// dilation factors. The model processes audio through:
///
/// 1. Condition DSP (optional) - processes input to generate conditioning signal
/// 2. LayerArrays - sequential processing with residual and skip connections
/// 3. Head scaling - final output scaling
///
/// The model supports real-time audio processing with pre-allocated buffers.
class WaveNet : public DSP
{
public:
  /// \brief Constructor
  /// \param in_channels Number of input channels
  /// \param layer_array_params Parameters for each layer array
  /// \param head_scale Scaling factor applied to the final head output
  /// \param with_head Whether to use a custom "head" module that further processes the output (not currently supported)
  /// \param weights Model weights (will be consumed during construction)
  /// \param condition_dsp Optional DSP module for processing the conditioning input
  /// \param expected_sample_rate Expected sample rate in Hz (-1.0 if unknown)
  WaveNet(const int in_channels, const std::vector<LayerArrayParams>& layer_array_params, const float head_scale,
          const bool with_head, std::vector<float> weights, std::unique_ptr<DSP> condition_dsp,
          const double expected_sample_rate = -1.0);

  /// \brief Destructor
  ~WaveNet() = default;

  /// \brief Process audio frames
  ///
  /// Implements the DSP::process() interface. Processes input audio through the
  /// complete WaveNet pipeline and writes to output.
  /// \param input Input audio buffers (in_channels x frames)
  /// \param output Output audio buffers (out_channels x frames)
  /// \param num_frames Number of frames to process
  void process(NAM_SAMPLE** input, NAM_SAMPLE** output, const int num_frames) override;

  /// \brief Set model weights from a vector
  /// \param weights Vector containing all model weights
  void set_weights_(std::vector<float>& weights);

  /// \brief Set model weights from an iterator
  /// \param weights Iterator to the weights vector. Will be advanced as weights are consumed.
  void set_weights_(std::vector<float>::iterator& weights);

protected:
  // Element-wise arrays:
  Eigen::MatrixXf _condition_input;
  Eigen::MatrixXf _condition_output;
  std::unique_ptr<DSP> _condition_dsp;
  // Temporary buffers for condition DSP processing (to avoid allocations in _process_condition)
  std::vector<std::vector<NAM_SAMPLE>> _condition_dsp_input_buffers;
  std::vector<std::vector<NAM_SAMPLE>> _condition_dsp_output_buffers;
  std::vector<NAM_SAMPLE*> _condition_dsp_input_ptrs;
  std::vector<NAM_SAMPLE*> _condition_dsp_output_ptrs;

  /// \brief Resize all buffers to handle maxBufferSize frames
  /// \param maxBufferSize Maximum number of frames to process in a single call
  void SetMaxBufferSize(const int maxBufferSize) override;

  /// \brief Compute the conditioning array to be given to the layer arrays
  ///
  /// Processes the condition input through the condition DSP (if present) or
  /// passes it through directly.
  /// \param num_frames Number of frames to process
  virtual void _process_condition(const int num_frames);

  /// \brief Fill in the "condition" array that's fed into the various parts of the net
  ///
  /// Copies input audio into the condition buffer for processing.
  /// \param input Input audio buffers
  /// \param num_frames Number of frames to process
  virtual void _set_condition_array(NAM_SAMPLE** input, const int num_frames);

  /// \brief Get the number of conditioning inputs
  ///
  /// For standard WaveNet, this is just the audio input (same as input channels).
  /// \return Number of conditioning input channels
  virtual int _get_condition_dim() const { return NumInputChannels(); };

private:
  std::vector<_LayerArray> _layer_arrays;

  float _head_scale;

  int mPrewarmSamples = 0; // Pre-compute during initialization
  int PrewarmSamples() override { return mPrewarmSamples; };
};

/// \brief Factory function to instantiate WaveNet from JSON configuration
/// \param config JSON configuration object
/// \param weights Model weights vector
/// \param expectedSampleRate Expected sample rate in Hz (-1.0 if unknown)
/// \return Unique pointer to a DSP object (WaveNet instance)
std::unique_ptr<DSP> Factory(const nlohmann::json& config, std::vector<float>& weights,
                             const double expectedSampleRate);
}; // namespace wavenet
}; // namespace nam
