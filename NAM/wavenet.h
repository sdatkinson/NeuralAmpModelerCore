#pragma once

#include <string>
#include <vector>

#include "json.hpp"
#include <Eigen/Dense>

#include "dsp.h"

namespace nam
{
namespace wavenet
{
// Rework the initialization API slightly. Merge w/ dsp.h later.
class _DilatedConv : public Conv1D
{
public:
  _DilatedConv(const int in_channels, const int out_channels, const int kernel_size, const int bias,
               const int dilation);
};

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
  // :param `head_input`: input to the head ("skip-out")
  // :param `output`: to next layer
  // :param `i_start`: Index of the first column of the input samples that the conv layer's first kernel will process
  // :param `j_start`: Index of the first column of the output block that will be written to
  // :param `num_frames`: number of frames to process
  void process_(const Eigen::MatrixXf& input, const Eigen::MatrixXf& condition, Eigen::MatrixXf& head_input,
                Eigen::MatrixXf& output, const long i_start, const long j_start, const int num_frames);
  // DEPRECATED - use SetMaxBufferSize() instead
  void set_num_frames_(const long num_frames);
  // The number of channels expected as input/output from this layer
  long get_channels() const { return this->_conv.get_in_channels(); };
  // Dilation of the input convolution layer
  int get_dilation() const { return this->_conv.get_dilation(); };
  // Kernel size of the input convolution layer
  long get_kernel_size() const { return this->_conv.get_kernel_size(); };

private:
  // The dilated convolution at the front of the block
  _DilatedConv _conv;
  // Input mixin
  Conv1x1 _input_mixin;
  // The post-activation 1x1 convolution
  Conv1x1 _1x1;
  // The internal state
  Eigen::MatrixXf _z;

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

  void advance_buffers_(const int num_frames);

  // Preparing for frames:
  // Rewind buffers if needed
  // Shift index to prepare
  //
  void prepare_for_frames_(const long num_frames);

  // All arrays are "short".
  void process_(const Eigen::MatrixXf& layer_inputs, // Short
                const Eigen::MatrixXf& condition, // Short
                Eigen::MatrixXf& layer_outputs, // Short
                Eigen::MatrixXf& head_inputs, // Sum up on this.
                Eigen::MatrixXf& head_outputs, // post head-rechannel
                const int num_frames);
  void set_num_frames_(const long num_frames);
  void set_weights_(std::vector<float>::iterator& it);

  // "Zero-indexed" receptive field.
  // E.g. a 1x1 convolution has a z.i.r.f. of zero.
  long get_receptive_field() const;

private:
  long _buffer_start;
  // The rechannel before the layers
  Conv1x1 _rechannel;

  // Buffers in between layers.
  // buffer [i] is the input to layer [i].
  // the last layer outputs to a short array provided by outside.
  std::vector<Eigen::MatrixXf> _layer_buffers;
  // The layer objects
  std::vector<_Layer> _layers;

  // Rechannel for the head
  Conv1x1 _head_rechannel;

  long _get_buffer_size() const { return this->_layer_buffers.size() > 0 ? this->_layer_buffers[0].cols() : 0; };
  long _get_channels() const;
  // "One-indexed" receptive field
  // TODO remove!
  // E.g. a 1x1 convolution has a o.i.r.f. of one.
  long _get_receptive_field() const;
  void _rewind_buffers_();
};

// The head module
// [Act->Conv] x L
class _Head
{
public:
  _Head(const int input_size, const int num_layers, const int channels, const std::string activation);
  void Reset(const double sampleRate, const int maxBufferSize);
  void set_weights_(std::vector<float>::iterator& weights);
  // NOTE: the head transforms the provided input by applying a nonlinearity
  // to it in-place!
  void process_(Eigen::MatrixXf& inputs, Eigen::MatrixXf& outputs);
  void set_num_frames_(const long num_frames);

private:
  int _channels;
  std::vector<Conv1x1> _layers;
  Conv1x1 _head;
  activations::Activation* _activation;

  // Stores the outputs of the convs *except* the last one, which goes in
  // The array `outputs` provided to .process_()
  std::vector<Eigen::MatrixXf> _buffers;

  // Apply the activation to the provided array, in-place
  void _apply_activation_(Eigen::MatrixXf& x);
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
  // Their outputs
  std::vector<Eigen::MatrixXf> _layer_array_outputs;
  // _Head _head;

  // One more than total layer arrays
  std::vector<Eigen::MatrixXf> _head_arrays;
  float _head_scale;
  Eigen::MatrixXf _head_output;

  void _advance_buffers_(const int num_frames);
  void _prepare_for_frames_(const long num_frames);

  // Ensure that all buffer arrays are the right size for this num_frames
  void _set_num_frames_(const long num_frames);

  int mPrewarmSamples = 0; // Pre-compute during initialization
  int PrewarmSamples() override { return mPrewarmSamples; };
};
}; // namespace wavenet
}; // namespace nam
