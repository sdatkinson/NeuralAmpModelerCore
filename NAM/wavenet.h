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
  _DilatedConv(const int inChannels, const int outChannels, const int kernelSize, const int bias,
               const int dilation);
};

class _Layer
{
public:
  _Layer(const int condition_size, const int channels, const int kernelSize, const int dilation,
         const std::string activation, const bool gated)
  : _conv(channels, gated ? 2 * channels : channels, kernelSize, true, dilation)
  , _input_mixin(condition_size, gated ? 2 * channels : channels, false)
  , _1x1(channels, channels, true)
  , _activation(activations::Activation::GetActivation(activation))
  , _gated(gated){};
  void SetWeights(weights_it& weights);
  // :param `input`: from previous layer
  // :param `output`: to next layer
  void Process(const Eigen::Ref<const Eigen::MatrixXf> input, const Eigen::Ref<const Eigen::MatrixXf> condition, Eigen::Ref<Eigen::MatrixXf> head_input,
                Eigen::Ref<Eigen::MatrixXf> output, const long i_start, const long j_start);
  void SetNumFrames(const long numFrames);
  long get_channels() const { return this->_conv.GetInChannels(); };
  int GetDilation() const { return this->_conv.GetDilation(); };
  long GetKernelSize() const { return this->_conv.GetKernelSize(); };

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
                   const int kernel_size_, const std::vector<int>& dilations_, const std::string activation_,
                   const bool gated_, const bool head_bias_)
  : inputSize(input_size_)
  , condition_size(condition_size_)
  , head_size(head_size_)
  , channels(channels_)
  , kernelSize(kernel_size_)
  , activation(activation_)
  , gated(gated_)
  , head_bias(head_bias_)
  {
    for (size_t i = 0; i < dilations_.size(); i++)
      this->dilations.push_back(dilations_[i]);
  };

  const int inputSize;
  const int condition_size;
  const int head_size;
  const int channels;
  const int kernelSize;
  std::vector<int> dilations;
  const std::string activation;
  const bool gated;
  const bool head_bias;
};

// An array of layers with the same channels, kernel sizes, activations.
class LayerArray
{
public:
  LayerArray(const int inputSize, const int condition_size, const int head_size, const int channels,
              const int kernelSize, const std::vector<int>& dilations, const std::string activation, const bool gated,
              const bool head_bias);

  void advance_buffers_(const int numFrames);

  // Preparing for frames:
  // Rewind buffers if needed
  // Shift index to prepare
  //
  void prepare_for_frames_(const long numFrames);

  // All arrays are "short".
  void Process(const Eigen::Ref<const Eigen::MatrixXf> layer_inputs, // Short
                const Eigen::Ref<const Eigen::MatrixXf> condition, // Short
                Eigen::Ref<Eigen::MatrixXf> layer_outputs, // Short
                Eigen::Ref<Eigen::MatrixXf> head_inputs, // Sum up on this.
                Eigen::Ref<Eigen::MatrixXf> head_outputs // post head-rechannel
  );
  void SetNumFrames(const long numFrames);
  void SetWeights(weights_it& it);

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
  void RewindBuffers();
};

// The head module
// [Act->Conv] x L
class Head
{
public:
  Head(const int inputSize, const int numLayers, const int channels, const std::string activation);
  void SetWeights(weights_it& weights);
  // NOTE: the head transforms the provided input by applying a nonlinearity
  // to it in-place!
  void Process(Eigen::Ref<Eigen::MatrixXf> inputs, Eigen::Ref<Eigen::MatrixXf> outputs);
  void SetNumFrames(const long numFrames);

private:
  int mChannels;
  std::vector<Conv1x1> mLayers;
  Conv1x1 mHead;
  activations::Activation* mActivation;

  // Stores the outputs of the convs *except* the last one, which goes in
  // The array `outputs` provided to .Process()
  std::vector<Eigen::MatrixXf> mBuffers;

  // Apply the activation to the provided array, in-place
  void ApplyActivation(Eigen::Ref<Eigen::MatrixXf> x);
};

// The main WaveNet model
class WaveNet : public DSP
{
public:
  WaveNet(const std::vector<LayerArrayParams>& layerArrayParams, const float headScale, const bool withHead,
          const std::vector<float>& weights, const double expectedSampleRate = -1.0);
  ~WaveNet() = default;

  void Finalize(const int numFrames) override;
  void SetWeights(const std::vector<float>& weights);

private:
  long mNumFrames;
  std::vector<LayerArray> mLayerArrays;
  // Their outputs
  std::vector<Eigen::MatrixXf> mLayerArrayOutputs;
  // Head _head;

  // Element-wise arrays:
  Eigen::MatrixXf mCondition;
  // One more than total layer arrays
  std::vector<Eigen::MatrixXf> mHeadArrays;
  float mHeadScale;
  Eigen::MatrixXf mHeadOutput;

  void AdvanceBuffers(const int numFrames);
  void PrepareForFrames(const long numFrames);
  void Process(float* input, float* output, const int numFrames) override;

  virtual int GetConditionDim() const { return 1; };
  // Fill in the "condition" array that's fed into the various parts of the net.
  virtual void SetConditionArray(float* input, const int numFrames);
  // Ensure that all buffer arrays are the right size for this numFrames
  void SetNumFrames(const long numFrames);
};
}; // namespace wavenet
}; // namespace nam
