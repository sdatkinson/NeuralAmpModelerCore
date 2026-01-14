#pragma once

#include <string>
#include <vector>

#include "json.hpp"
#include <Eigen/Dense>

#include "dsp.h"
#include "conv1d.h"

namespace nam
{
namespace wavenet
{
class _Layer
{
public:
  _Layer(const int condition_size, const int channels, const int kernel_size, const int dilation,
         const std::string activation, const bool gated)
  : _conv(channels, gated ? 2 * channels : channels, kernel_size, true, dilation)
  , _input_mixin(condition_size, gated ? 2 * channels : channels, false)
  , _1x1(channels, channels, true)
  , _activation(activations::Activation::get_activation(activation))
  , _gated(gated) {};
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
  // The internal state
  Eigen::MatrixXf _z;
  // Output to next layer (residual connection: input + _1x1 output)
  Eigen::MatrixXf _output_next_layer;
  // Output to head (skip connection: activated conv output)
  Eigen::MatrixXf _output_head;

  activations::Activation* _activation;
  const bool _gated;
};

class LayerArrayParams
{
public:
  LayerArrayParams(const int input_size_, const int condition_size_, const int head_size_, const int channels_,
                   const int kernel_size_, const std::vector<int>&& dilations_, const std::string activation_,
                   const bool gated_, const bool head_bias_)
  : input_size(input_size_)
  , condition_size(condition_size_)
  , head_size(head_size_)
  , channels(channels_)
  , kernel_size(kernel_size_)
  , dilations(std::move(dilations_))
  , activation(activation_)
  , gated(gated_)
  , head_bias(head_bias_)
  {
  }

  const int input_size;
  const int condition_size;
  const int head_size;
  const int channels;
  const int kernel_size;
  std::vector<int> dilations;
  const std::string activation;
  const bool gated;
  const bool head_bias;
};

// An array of layers with the same channels, kernel sizes, activations.
class _LayerArray
{
public:
  _LayerArray(const int input_size, const int condition_size, const int head_size, const int channels,
              const int kernel_size, const std::vector<int>& dilations, const std::string activation, const bool gated,
              const bool head_bias);

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
  // Accumulated head inputs from all layers
  Eigen::MatrixXf _head_inputs;

  // Rechannel for the head
  Conv1x1 _head_rechannel;

  long _get_channels() const;
  // Common processing logic after head inputs are set
  void ProcessInner(const Eigen::MatrixXf& layer_inputs, const Eigen::MatrixXf& condition, const int num_frames);
};

// The main WaveNet model
class WaveNet : public DSP
{
public:
  WaveNet(const std::vector<LayerArrayParams>& layer_array_params, const float head_scale, const bool with_head,
          std::vector<float> weights, const double expected_sample_rate = -1.0);
  ~WaveNet() = default;
  void process(NAM_SAMPLE* input, NAM_SAMPLE* output, const int num_frames) override;
  void set_weights_(std::vector<float>& weights);

protected:
  // Element-wise arrays:
  Eigen::MatrixXf _condition;

  void SetMaxBufferSize(const int maxBufferSize) override;
  // Fill in the "condition" array that's fed into the various parts of the net.
  virtual void _set_condition_array(NAM_SAMPLE* input, const int num_frames);
  // How many conditioning inputs are there.
  // Just one--the audio.
  virtual int _get_condition_dim() const { return 1; };

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
