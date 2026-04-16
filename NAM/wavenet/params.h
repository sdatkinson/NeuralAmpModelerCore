#pragma once

#include <stdexcept>
#include <string>
#include <vector>

#include "../activations.h"

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

/// \brief Parameters for layer1x1 configuration
///
/// Configures an optional 1x1 convolution that processes the activation output
/// for the residual connection to the next layer.
struct Layer1x1Params
{
  /// \brief Constructor
  /// \param active_ Whether the layer1x1 convolution is active
  /// \param groups_ Number of groups for grouped convolution
  Layer1x1Params(bool active_, int groups_)
  : active(active_)
  , groups(groups_)
  {
  }

  const bool active; ///< Whether the layer1x1 convolution is active
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
  /// \param groups_ Number of groups for grouped convolution in the condition-to-scale-shift submodule (default: 1)
  _FiLMParams(bool active_, bool shift_, int groups_ = 1)
  : active(active_)
  , shift(shift_)
  , groups(groups_)
  {
  }
  const bool active; ///< Whether FiLM is active
  const bool shift; ///< Whether to apply shift in addition to scale
  const int groups; ///< Number of groups for grouped convolution in the condition-to-scale-shift submodule
};

/// \brief Parameters for constructing a single Layer
///
/// Contains all configuration needed to construct a detail::Layer
struct LayerParams
{
  /// \brief Constructor
  /// \param condition_size_ Size of the conditioning input
  /// \param channels_ Number of input/output channels from layer to layer
  /// \param bottleneck_ Internal channel count
  /// \param kernel_size_ Kernel size for the dilated convolution
  /// \param dilation_ Dilation factor for the convolution
  /// \param activation_config_ Primary activation function configuration
  /// \param gating_mode_ Gating mode (NONE, GATED, or BLENDED)
  /// \param groups_input_ Number of groups for the input convolution
  /// \param groups_input_mixin_ Number of groups for the input mixin convolution
  /// \param layer1x1_params_ Configuration of the optional layer1x1 convolution
  /// \param head1x1_params_ Configuration of the optional head1x1 convolution
  /// \param secondary_activation_config_ Secondary activation (for gating/blending)
  /// \param conv_pre_film_params_ FiLM parameters before the input convolution
  /// \param conv_post_film_params_ FiLM parameters after the input convolution
  /// \param input_mixin_pre_film_params_ FiLM parameters before the input mixin
  /// \param input_mixin_post_film_params_ FiLM parameters after the input mixin
  /// \param activation_pre_film_params_ FiLM parameters after the input/mixin summed output before activation
  /// \param activation_post_film_params_ FiLM parameters after the activation output before the layer1x1 convolution
  /// \param _layer1x1_post_film_params_ FiLM parameters after the layer1x1 convolution
  /// \param head1x1_post_film_params_ FiLM parameters after the head1x1 convolution
  LayerParams(const int condition_size_, const int channels_, const int bottleneck_, const int kernel_size_,
              const int dilation_, const activations::ActivationConfig& activation_config_,
              const GatingMode gating_mode_, const int groups_input_, const int groups_input_mixin_,
              const Layer1x1Params& layer1x1_params_, const Head1x1Params& head1x1_params_,
              const activations::ActivationConfig& secondary_activation_config_,
              const _FiLMParams& conv_pre_film_params_, const _FiLMParams& conv_post_film_params_,
              const _FiLMParams& input_mixin_pre_film_params_, const _FiLMParams& input_mixin_post_film_params_,
              const _FiLMParams& activation_pre_film_params_, const _FiLMParams& activation_post_film_params_,
              const _FiLMParams& _layer1x1_post_film_params_, const _FiLMParams& head1x1_post_film_params_)
  : condition_size(condition_size_)
  , channels(channels_)
  , bottleneck(bottleneck_)
  , kernel_size(kernel_size_)
  , dilation(dilation_)
  , activation_config(activation_config_)
  , gating_mode(gating_mode_)
  , groups_input(groups_input_)
  , groups_input_mixin(groups_input_mixin_)
  , layer1x1_params(layer1x1_params_)
  , head1x1_params(head1x1_params_)
  , secondary_activation_config(secondary_activation_config_)
  , conv_pre_film_params(conv_pre_film_params_)
  , conv_post_film_params(conv_post_film_params_)
  , input_mixin_pre_film_params(input_mixin_pre_film_params_)
  , input_mixin_post_film_params(input_mixin_post_film_params_)
  , activation_pre_film_params(activation_pre_film_params_)
  , activation_post_film_params(activation_post_film_params_)
  , _layer1x1_post_film_params(_layer1x1_post_film_params_)
  , head1x1_post_film_params(head1x1_post_film_params_)
  {
  }

  const int condition_size; ///< Size of the conditioning input
  const int channels; ///< Number of input/output channels from layer to layer
  const int bottleneck; ///< Internal channel count
  const int kernel_size; ///< Kernel size for the dilated convolution
  const int dilation; ///< Dilation factor for the convolution
  const activations::ActivationConfig activation_config; ///< Primary activation function configuration
  const GatingMode gating_mode; ///< Gating mode (NONE, GATED, or BLENDED)
  const int groups_input; ///< Number of groups for the input convolution
  const int groups_input_mixin; ///< Number of groups for the input mixin convolution
  const Layer1x1Params layer1x1_params; ///< Configuration of the optional layer1x1 convolution
  const Head1x1Params head1x1_params; ///< Configuration of the optional head1x1 convolution
  const activations::ActivationConfig secondary_activation_config; ///< Secondary activation (for gating/blending)
  const _FiLMParams conv_pre_film_params; ///< FiLM parameters before the input convolution
  const _FiLMParams conv_post_film_params; ///< FiLM parameters after the input convolution
  const _FiLMParams input_mixin_pre_film_params; ///< FiLM parameters before the input mixin
  const _FiLMParams input_mixin_post_film_params; ///< FiLM parameters after the input mixin
  const _FiLMParams activation_pre_film_params; ///< FiLM parameters before activation
  const _FiLMParams activation_post_film_params; ///< FiLM parameters after activation
  const _FiLMParams _layer1x1_post_film_params; ///< FiLM parameters after the layer1x1 convolution (layer1x1_post_film)
  const _FiLMParams head1x1_post_film_params; ///< FiLM parameters after the head1x1 convolution
};

/// \brief Parameters for constructing a LayerArray
///
/// Contains all configuration needed to construct a detail::LayerArray with multiple layers
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
  /// \param kernel_sizes_ Per-layer kernel sizes, one per layer
  /// \param dilations_ Vector of dilation factors, one per layer
  /// \param activation_configs_ Vector of primary activation configurations, one per layer
  /// \param gating_modes_ Vector of gating modes, one per layer
  /// \param head_kernel_size_ Kernel size of the head rechannel conv (>= 1)
  /// \param head_bias_ Whether to use bias in the head rechannel
  /// \param groups_input Number of groups for input convolutions
  /// \param groups_input_mixin_ Number of groups for input mixin convolutions
  /// \param layer1x1_params_ Parameters for optional layer1x1 convolutions
  /// \param head1x1_params_ Parameters for optional head1x1 convolutions
  /// \param secondary_activation_configs_ Vector of secondary activation configs for gating/blending, one per layer
  /// \param conv_pre_film_params_ FiLM parameters before input convolutions
  /// \param conv_post_film_params_ FiLM parameters after input convolutions
  /// \param input_mixin_pre_film_params_ FiLM parameters before input mixin
  /// \param input_mixin_post_film_params_ FiLM parameters after input mixin
  /// \param activation_pre_film_params_ FiLM parameters before activation
  /// \param activation_post_film_params_ FiLM parameters after activation
  /// \param _layer1x1_post_film_params_ FiLM parameters after layer1x1 convolutions
  /// \param head1x1_post_film_params_ FiLM parameters after head1x1 convolutions
  /// \throws std::invalid_argument If dilations, activation_configs, gating_modes, or secondary_activation_configs
  /// sizes don't match
  LayerArrayParams(const int input_size_, const int condition_size_, const int head_size_, const int head_kernel_size_,
                   const int channels_, const int bottleneck_, const std::vector<int>&& kernel_sizes_,
                   const std::vector<int>&& dilations_,
                   const std::vector<activations::ActivationConfig>&& activation_configs_,
                   const std::vector<GatingMode>&& gating_modes_, const bool head_bias_, const int groups_input,
                   const int groups_input_mixin_, const Layer1x1Params& layer1x1_params_,
                   const Head1x1Params& head1x1_params_,
                   const std::vector<activations::ActivationConfig>&& secondary_activation_configs_,
                   const _FiLMParams& conv_pre_film_params_, const _FiLMParams& conv_post_film_params_,
                   const _FiLMParams& input_mixin_pre_film_params_, const _FiLMParams& input_mixin_post_film_params_,
                   const _FiLMParams& activation_pre_film_params_, const _FiLMParams& activation_post_film_params_,
                   const _FiLMParams& _layer1x1_post_film_params_, const _FiLMParams& head1x1_post_film_params_)
  : input_size(input_size_)
  , condition_size(condition_size_)
  , head_size(head_size_)
  , head_kernel_size(head_kernel_size_)
  , channels(channels_)
  , bottleneck(bottleneck_)
  , kernel_sizes(std::move(kernel_sizes_))
  , dilations(std::move(dilations_))
  , activation_configs(std::move(activation_configs_))
  , gating_modes(std::move(gating_modes_))
  , head_bias(head_bias_)
  , groups_input(groups_input)
  , groups_input_mixin(groups_input_mixin_)
  , layer1x1_params(layer1x1_params_)
  , head1x1_params(head1x1_params_)
  , secondary_activation_configs(std::move(secondary_activation_configs_))
  , conv_pre_film_params(conv_pre_film_params_)
  , conv_post_film_params(conv_post_film_params_)
  , input_mixin_pre_film_params(input_mixin_pre_film_params_)
  , input_mixin_post_film_params(input_mixin_post_film_params_)
  , activation_pre_film_params(activation_pre_film_params_)
  , activation_post_film_params(activation_post_film_params_)
  , _layer1x1_post_film_params(_layer1x1_post_film_params_)
  , head1x1_post_film_params(head1x1_post_film_params_)
  {
    if (head_kernel_size < 1)
    {
      throw std::invalid_argument("LayerArrayParams: head_kernel_size must be >= 1");
    }
    const size_t num_layers = dilations.size();
    if (kernel_sizes.empty())
    {
      throw std::invalid_argument("LayerArrayParams: kernel_sizes must not be empty");
    }
    if (kernel_sizes.size() != num_layers)
    {
      throw std::invalid_argument("LayerArrayParams: dilations size (" + std::to_string(num_layers)
                                  + ") must match kernel_sizes size (" + std::to_string(kernel_sizes.size()) + ")");
    }
    if (activation_configs.size() != num_layers)
    {
      throw std::invalid_argument("LayerArrayParams: dilations size (" + std::to_string(num_layers)
                                  + ") must match activation_configs size (" + std::to_string(activation_configs.size())
                                  + ")");
    }
    if (gating_modes.size() != num_layers)
    {
      throw std::invalid_argument("LayerArrayParams: dilations size (" + std::to_string(num_layers)
                                  + ") must match gating_modes size (" + std::to_string(gating_modes.size()) + ")");
    }
    if (secondary_activation_configs.size() != num_layers)
    {
      throw std::invalid_argument("LayerArrayParams: dilations size (" + std::to_string(num_layers)
                                  + ") must match secondary_activation_configs size ("
                                  + std::to_string(secondary_activation_configs.size()) + ")");
    }
  }

  const int input_size; ///< Input size (number of channels)
  const int condition_size; ///< Size of conditioning input
  const int head_size; ///< Size of head output (after rechannel)
  const int head_kernel_size; ///< Kernel size of head rechannel convolution (>= 1)
  const int channels; ///< Number of channels in each layer
  const int bottleneck; ///< Bottleneck size (internal channel count)
  std::vector<int> kernel_sizes; ///< Per-layer kernel sizes, one per layer
  std::vector<int> dilations; ///< Dilation factors, one per layer
  std::vector<activations::ActivationConfig> activation_configs; ///< Primary activation configurations, one per layer
  std::vector<GatingMode> gating_modes; ///< Gating modes, one per layer
  const bool head_bias; ///< Whether to use bias in head rechannel
  const int groups_input; ///< Number of groups for input convolutions
  const int groups_input_mixin; ///< Number of groups for input mixin
  const Layer1x1Params layer1x1_params; ///< Parameters for optional layer1x1
  const Head1x1Params head1x1_params; ///< Parameters for optional head1x1
  std::vector<activations::ActivationConfig>
    secondary_activation_configs; ///< Secondary activation configs for gating/blending, one per layer
  const _FiLMParams conv_pre_film_params; ///< FiLM params before input conv
  const _FiLMParams conv_post_film_params; ///< FiLM params after input conv
  const _FiLMParams input_mixin_pre_film_params; ///< FiLM params before input mixin
  const _FiLMParams input_mixin_post_film_params; ///< FiLM params after input mixin
  const _FiLMParams activation_pre_film_params; ///< FiLM params before activation
  const _FiLMParams activation_post_film_params; ///< FiLM params after activation
  const _FiLMParams _layer1x1_post_film_params; ///< FiLM params after layer1x1 conv
  const _FiLMParams head1x1_post_film_params; ///< FiLM params after head1x1 conv
};

/// \brief Parameters for the optional post-stack head (matches Python ``nam.models.wavenet._head.Head``).
/// JSON export omits ``in_channels`` (implied by last layer array ``head_size``); load sets it from there.
struct HeadParams
{
  int in_channels;
  int channels;
  int out_channels;
  std::vector<int> kernel_sizes;
  activations::ActivationConfig activation_config;
};

} // namespace wavenet
} // namespace nam
