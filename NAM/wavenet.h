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
class DilatedConv : public Conv1D
{
public:
  DilatedConv(const int inChannels, const int outChannels, const int kernelSize, const int bias,
               const int dilation);
};

class Layer
{
public:
  Layer(const int condition_size, const int channels, const int kernelSize, const int dilation,
         const std::string activation, const bool gated)
  : mConv(channels, gated ? 2 * channels : channels, kernelSize, true, dilation)
  , mInputMixin(condition_size, gated ? 2 * channels : channels, false)
  , _1x1(channels, channels, true)
  , mActivation(activations::Activation::GetActivation(activation))
  , mGated(gated){};
  void SetWeights(weights_it& weights);
  // :param `input`: from previous layer
  // :param `output`: to next layer
  void Process(const Eigen::Ref<const Eigen::MatrixXf> input, const Eigen::Ref<const Eigen::MatrixXf> condition, Eigen::Ref<Eigen::MatrixXf> head_input,
                Eigen::Ref<Eigen::MatrixXf> output, const long i_start, const long j_start);
  void SetNumFrames(const long numFrames);
  long GetChannels() const { return this->mConv.GetInChannels(); };
  int GetDilation() const { return this->mConv.GetDilation(); };
  long GetKernelSize() const { return this->mConv.GetKernelSize(); };

private:
  // The dilated convolution at the front of the block
  DilatedConv mConv;
  // Input mixin
  Conv1x1 mInputMixin;
  // The post-activation 1x1 convolution
  Conv1x1 _1x1;
  // The internal state
  Eigen::MatrixXf _z;

  activations::Activation* mActivation;
  const bool mGated;
};

class LayerArrayParams
{
public:
  LayerArrayParams(const int inputSize, const int conditionSize, const int headSize, const int channels,
                   const int kernelSize, const std::vector<int>& dilations, const std::string activation,
                   const bool gated, const bool headBias)
  : mInputSize(inputSize)
  , mConditionSize(conditionSize)
  , mHeadSize(headSize)
  , mChannels(channels)
  , mKernelSize(kernelSize)
  , mActivation(activation)
  , mGated(gated)
  , mHeadBias(headBias)
  {
    for (size_t i = 0; i < dilations.size(); i++)
      this->mDilations.push_back(dilations[i]);
  };

  const int mInputSize;
  const int mConditionSize;
  const int mHeadSize;
  const int mChannels;
  const int mKernelSize;
  std::vector<int> mDilations;
  const std::string mActivation;
  const bool mGated;
  const bool mHeadBias;
};

// An array of layers with the same channels, kernel sizes, activations.
class LayerArray
{
public:
  LayerArray(const int inputSize, const int condition_size, const int head_size, const int channels,
              const int kernelSize, const std::vector<int>& dilations, const std::string activation, const bool gated,
              const bool head_bias);

  void AdvanceBuffers(const int numFrames);

  // Preparing for frames:
  // Rewind buffers if needed
  // Shift index to prepare
  //
  void PrepareForFrames(const long numFrames);

  // All arrays are "short".
  void Process(const Eigen::Ref<const Eigen::MatrixXf> layerInputs, // Short
                const Eigen::Ref<const Eigen::MatrixXf> condition, // Short
                Eigen::Ref<Eigen::MatrixXf> layerOutputs, // Short
                Eigen::Ref<Eigen::MatrixXf> headInputs, // Sum up on this.
                Eigen::Ref<Eigen::MatrixXf> headOutputs // post head-rechannel
  );
  void SetNumFrames(const long numFrames);
  void SetWeights(weights_it& it);

  // "Zero-indexed" receptive field.
  // E.g. a 1x1 convolution has a z.i.r.f. of zero.
  long GetReceptiveField() const;

private:
  long mBufferStart;
  // The rechannel before the layers
  Conv1x1 mReChannel;

  // Buffers in between layers.
  // buffer [i] is the input to layer [i].
  // the last layer outputs to a short array provided by outside.
  std::vector<Eigen::MatrixXf> mLayerBuffers;
  // The layer objects
  std::vector<Layer> mLayers;

  // Rechannel for the head
  Conv1x1 mHeadRechannel;

  long GetBufferSize() const { return this->mLayerBuffers.size() > 0 ? this->mLayerBuffers[0].cols() : 0; };
  long GetChannels() const;
  // "One-indexed" receptive field
  // TODO remove!
  // E.g. a 1x1 convolution has a o.i.r.f. of one.
  long _GetReceptiveField() const; // TODO: why two!
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
