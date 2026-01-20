#pragma once

#include <string>
#include <vector>
#include <memory>
#include <stdexcept>

#include "json.hpp"
#include <Eigen/Dense>

#include "dsp.h"
#include "conv1d.h"
#include "gating_activations.h"

namespace nam
{
namespace wavenet
{

// Gating mode for WaveNet layers
enum class GatingMode
{
  NONE, // No gating or blending
  GATED, // Traditional gating (element-wise multiplication)
  BLENDED // Blending (weighted average)
};

// Helper function for backward compatibility with boolean gated parameter
inline GatingMode gating_mode_from_bool(bool gated)
{
  return gated ? GatingMode::GATED : GatingMode::NONE;
}
// Parameters for head1x1 configuration
struct Head1x1Params
{
  Head1x1Params(bool active_, int out_channels_, int groups_)
  : active(active_)
  , out_channels(out_channels_)
  , groups(groups_)
  {
  }

  const bool active;
  const int out_channels;
  const int groups;
};

class _Layer
{
public:
  // New constructor with GatingMode enum and configurable activations
  _Layer(const int condition_size, const int channels, const int bottleneck, const int kernel_size, const int dilation,
         const nlohmann::json activation_config, const GatingMode gating_mode, const int groups_input, const int groups_1x1,
         const Head1x1Params& head1x1_params, const std::string& secondary_activation)
  : _conv(channels, (gating_mode != GatingMode::NONE) ? 2 * bottleneck : bottleneck, kernel_size, true, dilation)
  , _input_mixin(condition_size, (gating_mode != GatingMode::NONE) ? 2 * bottleneck : bottleneck, false)
  , _1x1(bottleneck, channels, groups_1x1)
  , _activation(activations::Activation::get_activation(activation_config)) // now supports activations with parameters
  , _gating_mode(gating_mode)
  , _bottleneck(bottleneck)
  {
    if (head1x1_params.active)
    {
      _head1x1 = std::make_unique<Conv1x1>(bottleneck, head1x1_params.out_channels, true, head1x1_params.groups);
    }

    // Validate & initialize gating/blending activation
    if (gating_mode == GatingMode::GATED)
    {
      if (secondary_activation.empty())
        throw std::invalid_argument("secondary_activation must be provided for gated mode");
      _gating_activation = std::make_unique<gating_activations::GatingActivation>(
        _activation, activations::Activation::get_activation(secondary_activation), bottleneck);
    }
    else if (gating_mode == GatingMode::BLENDED)
    {
      if (secondary_activation.empty())
        throw std::invalid_argument("secondary_activation must be provided for blended mode");
      _blending_activation = std::make_unique<gating_activations::BlendingActivation>(
        _activation, activations::Activation::get_activation(secondary_activation), bottleneck);
    }
    else
    {
      if (!secondary_activation.empty())
        throw std::invalid_argument("secondary_activation provided for none mode");
    }
  };

  // Resize all arrays to be able to process `maxBufferSize` frames.
  void SetMaxBufferSize(const int maxBufferSize);
  // Set the parameters of this module
  void set_weights_(std::vector<float>::iterator& weights);
  // Process a block of frames.
  // :param `input`: from previous layer
  // :param `condition`: conditioning input (input to the WaveNet / "skip-in")
  // :param `num_frames`: number of frames to process
  // Outputs are stored internally and accessible via GetOutputNextLayer() and GetOutputHead()
  void Process(const Eigen::MatrixXf& input, const Eigen::MatrixXf& condition, const int num_frames);
  // The number of channels expected as input/output from this layer
  long get_channels() const { return this->_conv.get_in_channels(); };
  // Dilation of the input convolution layer
  int get_dilation() const { return this->_conv.get_dilation(); };
  // Kernel size of the input convolution layer
  long get_kernel_size() const { return this->_conv.get_kernel_size(); };

  // Get output to next layer (residual connection: input + _1x1 output)
  // Returns the full pre-allocated buffer; only the first `num_frames` columns
  // are valid for a given processing call. Slice with .leftCols(num_frames) as needed.
  Eigen::MatrixXf& GetOutputNextLayer() { return this->_output_next_layer; }
  const Eigen::MatrixXf& GetOutputNextLayer() const { return this->_output_next_layer; }
  // Get output to head (skip connection: activated conv output)
  // Returns the full pre-allocated buffer; only the first `num_frames` columns
  // are valid for a given processing call. Slice with .leftCols(num_frames) as needed.
  Eigen::MatrixXf& GetOutputHead() { return this->_output_head; }
  const Eigen::MatrixXf& GetOutputHead() const { return this->_output_head; }

  // Access Conv1D for Reset() propagation (needed for _LayerArray)
  Conv1D& get_conv() { return _conv; }
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

  activations::Activation* _activation;
  const GatingMode _gating_mode;
  const int _bottleneck; // Internal channel count (not doubled when gated)

  // Gating/blending activation objects
  std::unique_ptr<gating_activations::GatingActivation> _gating_activation;
  std::unique_ptr<gating_activations::BlendingActivation> _blending_activation;
};

class LayerArrayParams
{
public:
  LayerArrayParams(const int input_size_, const int condition_size_, const int head_size_, const int channels_,
                   const int bottleneck_, const int kernel_size_, const std::vector<int>&& dilations_,
                   const nlohmann::json activation_, const GatingMode gating_mode_, const bool head_bias_,
                   const int groups_input, const int groups_1x1_, const Head1x1Params& head1x1_params_,
                   const std::string& secondary_activation_)
  : input_size(input_size_)
  , condition_size(condition_size_)
  , head_size(head_size_)
  , channels(channels_)
  , bottleneck(bottleneck_)
  , kernel_size(kernel_size_)
  , dilations(std::move(dilations_))
  , activation_config(activation_)
  , gating_mode(gating_mode_)
  , head_bias(head_bias_)
  , groups_input(groups_input)
  , groups_1x1(groups_1x1_)
  , head1x1_params(head1x1_params_)
  , secondary_activation(secondary_activation_)
  {
  }

  const int input_size;
  const int condition_size;
  const int head_size;
  const int channels;
  const int bottleneck;
  const int kernel_size;
  std::vector<int> dilations;
  const nlohmann::json activation_config;
  const GatingMode gating_mode;
  const bool head_bias;
  const int groups_input;
  const int groups_1x1;
  const Head1x1Params head1x1_params;
  const std::string secondary_activation;
};

// An array of layers with the same channels, kernel sizes, activations.
class _LayerArray
{
public:
  // New constructor with GatingMode enum and configurable activations
  _LayerArray(const int input_size, const int condition_size, const int head_size, const int channels,
              const int bottleneck, const int kernel_size, const std::vector<int>& dilations,
              const nlohmann::json activation_config, const GatingMode gating_mode, const bool head_bias, const int groups_input,
              const int groups_1x1, const Head1x1Params& head1x1_params, const std::string& secondary_activation);

  void SetMaxBufferSize(const int maxBufferSize);

  // All arrays are "short".
  // Process without head input (first layer array) - zeros head inputs before proceeding
  void Process(const Eigen::MatrixXf& layer_inputs, // Short
               const Eigen::MatrixXf& condition, // Short
               const int num_frames);
  // Process with head input (subsequent layer arrays) - copies head input before proceeding
  void Process(const Eigen::MatrixXf& layer_inputs, // Short
               const Eigen::MatrixXf& condition, // Short
               const Eigen::MatrixXf& head_inputs, // Short - from previous layer array
               const int num_frames);
  // Get output from last layer (for next layer array)
  // Returns the full pre-allocated buffer; only the first `num_frames` columns
  // are valid for a given processing call. Slice with .leftCols(num_frames) as needed.
  Eigen::MatrixXf& GetLayerOutputs() { return this->_layer_outputs; }
  const Eigen::MatrixXf& GetLayerOutputs() const { return this->_layer_outputs; }
  // Get head outputs (post head-rechannel)
  // Returns the full pre-allocated buffer; only the first `num_frames` columns
  // are valid for a given processing call. Slice with .leftCols(num_frames) as needed.
  Eigen::MatrixXf& GetHeadOutputs();
  const Eigen::MatrixXf& GetHeadOutputs() const;
  void set_weights_(std::vector<float>::iterator& it);

  // "Zero-indexed" receptive field.
  // E.g. a 1x1 convolution has a z.i.r.f. of zero.
  long get_receptive_field() const;

private:
  // The rechannel before the layers
  Conv1x1 _rechannel;

  // The layer objects
  std::vector<_Layer> _layers;
  // Output from last layer (for next layer array)
  Eigen::MatrixXf _layer_outputs;
  // Accumulated head inputs from all layers (bottleneck channels)
  Eigen::MatrixXf _head_inputs;

  // Rechannel for the head (bottleneck -> head_size)
  Conv1x1 _head_rechannel;

  // Bottleneck size (internal channel count)
  const int _bottleneck;

  long _get_channels() const;
  // Common processing logic after head inputs are set
  void ProcessInner(const Eigen::MatrixXf& layer_inputs, const Eigen::MatrixXf& condition, const int num_frames);
};

// The main WaveNet model
class WaveNet : public DSP
{
public:
  WaveNet(const int in_channels, const std::vector<LayerArrayParams>& layer_array_params, const float head_scale,
          const bool with_head, std::vector<float> weights, std::unique_ptr<DSP> condition_dsp,
          const double expected_sample_rate = -1.0);
  ~WaveNet() = default;
  void process(NAM_SAMPLE** input, NAM_SAMPLE** output, const int num_frames) override;
  void set_weights_(std::vector<float>& weights);
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

  void SetMaxBufferSize(const int maxBufferSize) override;
  // Compute the conditioning array to be given to the layer arrays
  virtual void _process_condition(const int num_frames);
  // Fill in the "condition" array that's fed into the various parts of the net.
  virtual void _set_condition_array(NAM_SAMPLE** input, const int num_frames);
  // How many conditioning inputs are there.
  // Just one--the audio.
  virtual int _get_condition_dim() const { return NumInputChannels(); };

private:
  std::vector<_LayerArray> _layer_arrays;

  float _head_scale;

  int mPrewarmSamples = 0; // Pre-compute during initialization
  int PrewarmSamples() override { return mPrewarmSamples; };
};

// Factory to instantiate from nlohmann json
std::unique_ptr<DSP> Factory(const nlohmann::json& config, std::vector<float>& weights,
                             const double expectedSampleRate);
}; // namespace wavenet
}; // namespace nam
