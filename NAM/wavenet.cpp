#include <algorithm>
#include <iostream>
#include <math.h>

#include <Eigen/Dense>

#include "wavenet.h"

nam::wavenet::DilatedConv::DilatedConv(const int inChannels, const int outChannels, const int kernelSize,
                                         const int bias, const int dilation)
{
  SetSize(inChannels, outChannels, kernelSize, bias, dilation);
}

void nam::wavenet::Layer::SetWeights(weights_it& weights)
{
  mConv.SetWeights(weights);
  mInputMixin.SetWeights(weights);
  _1x1.SetWeights(weights);
}

void nam::wavenet::Layer::Process(const Eigen::Ref<const Eigen::MatrixXf> input, const Eigen::Ref<const Eigen::MatrixXf> condition,
                                    Eigen::Ref<Eigen::MatrixXf> head_input, Eigen::Ref<Eigen::MatrixXf> output, const long i_start,
                                    const long j_start)
{
  const long ncols = condition.cols();
  const long channels = GetChannels();
  // Input dilated conv
  mConv.Process(input, _z, i_start, ncols, 0);
  // Mix-in condition
  _z += mInputMixin.Process(condition);

  mActivation->Apply(_z);

  if (mGated)
  {
    activations::Activation::GetActivation("Sigmoid")->Apply(_z.block(channels, 0, channels, _z.cols()));

    _z.topRows(channels).array() *= _z.bottomRows(channels).array();
    // _z.topRows(channels) = _z.topRows(channels).cwiseProduct(
    //   _z.bottomRows(channels)
    // );
  }

  head_input += _z.topRows(channels);
  output.middleCols(j_start, ncols) = input.middleCols(i_start, ncols) + _1x1.Process(_z.topRows(channels));
}

void nam::wavenet::Layer::SetNumFrames(const long numFrames)
{
  if (_z.rows() == mConv.GetOutChannels() && _z.cols() == numFrames)
    return; // Already has correct size

  _z.resize(mConv.GetOutChannels(), numFrames);
  _z.setZero();
}

// LayerArray =================================================================

#define LAYER_ARRAY_BUFFER_SIZE 65536

nam::wavenet::LayerArray::LayerArray(const int inputSize, const int condition_size, const int head_size,
                                       const int channels, const int kernelSize, const std::vector<int>& dilations,
                                       const std::string activation, const bool gated, const bool head_bias)
: mReChannel(inputSize, channels, false)
, mHeadRechannel(channels, head_size, head_bias)
{
  for (size_t i = 0; i < dilations.size(); i++)
    mLayers.push_back(Layer(condition_size, channels, kernelSize, dilations[i], activation, gated));
  const long receptiveField = GetReceptiveField();
  for (size_t i = 0; i < dilations.size(); i++)
  {
    mLayerBuffers.push_back(Eigen::MatrixXf(channels, LAYER_ARRAY_BUFFER_SIZE + receptiveField - 1));
    mLayerBuffers[i].setZero();
  }
  mBufferStart = GetReceptiveField() - 1;
}

void nam::wavenet::LayerArray::AdvanceBuffers(const int numFrames)
{
  mBufferStart += numFrames;
}

long nam::wavenet::LayerArray::GetReceptiveField() const
{
  long result = 0;
  for (size_t i = 0; i < mLayers.size(); i++)
    result += mLayers[i].GetDilation() * (mLayers[i].GetKernelSize() - 1);
  return result;
}

void nam::wavenet::LayerArray::PrepareForFrames(const long numFrames)
{
  // Example:
  // _buffer_start = 0
  // numFrames = 64
  // buffer_size = 64
  // -> this will write on indices 0 through 63, inclusive.
  // -> No illegal writes.
  // -> no rewind needed.
  if (mBufferStart + numFrames > GetBufferSize())
    RewindBuffers();
}

void nam::wavenet::LayerArray::Process(const Eigen::Ref<const Eigen::MatrixXf> layerInputs, const Eigen::Ref<const Eigen::MatrixXf> condition,
                                         Eigen::Ref<Eigen::MatrixXf> headInputs, Eigen::Ref<Eigen::MatrixXf> layerOutputs,
                                         Eigen::Ref<Eigen::MatrixXf> headOutputs)
{
  mLayerBuffers[0].middleCols(mBufferStart, layerInputs.cols()) = mReChannel.Process(layerInputs);
  const size_t last_layer = mLayers.size() - 1;
  for (size_t i = 0; i < mLayers.size(); i++)
  {
    if (i == last_layer)
    {
      mLayers[i].Process(mLayerBuffers[i], condition, headInputs,
                                layerOutputs, mBufferStart,
                                0);
    }
    else
    {
      mLayers[i].Process(mLayerBuffers[i], condition, headInputs,
                                mLayerBuffers[i + 1], mBufferStart,
                                mBufferStart);
    }

  }
  headOutputs = mHeadRechannel.Process(headInputs);
}

void nam::wavenet::LayerArray::SetNumFrames(const long numFrames)
{
  // Wavenet checks for unchanged numFrames; if we made it here, there's
  // something to do.
  if (LAYER_ARRAY_BUFFER_SIZE - numFrames < GetReceptiveField())
  {
    std::stringstream ss;
    ss << "Asked to accept a buffer of " << numFrames << " samples, but the buffer is too short ("
       << LAYER_ARRAY_BUFFER_SIZE << ") to get out of the recptive field (" << GetReceptiveField()
       << "); copy errors could occur!\n";
    throw std::runtime_error(ss.str().c_str());
  }
  for (size_t i = 0; i < mLayers.size(); i++)
    mLayers[i].SetNumFrames(numFrames);
}

void nam::wavenet::LayerArray::SetWeights(weights_it& weights)
{
  mReChannel.SetWeights(weights);
  for (size_t i = 0; i < mLayers.size(); i++)
    mLayers[i].SetWeights(weights);
  mHeadRechannel.SetWeights(weights);
}

long nam::wavenet::LayerArray::GetChannels() const
{
  return mLayers.size() > 0 ? mLayers[0].GetChannels() : 0;
}

long nam::wavenet::LayerArray::_GetReceptiveField() const // TODO: why two?
{
  // TODO remove this and use GetReceptiveField() instead!
  long res = 1;
  for (size_t i = 0; i < mLayers.size(); i++)
    res += (mLayers[i].GetKernelSize() - 1) * mLayers[i].GetDilation();
  return res;
}

void nam::wavenet::LayerArray::RewindBuffers()
// Consider wrapping instead...
// Can make this smaller--largest dilation, not receptive field!
{
  const long start = GetReceptiveField() - 1;
  for (size_t i = 0; i < mLayerBuffers.size(); i++)
  {
    const long d = (mLayers[i].GetKernelSize() - 1) * mLayers[i].GetDilation();
    mLayerBuffers[i].middleCols(start - d, d) = mLayerBuffers[i].middleCols(mBufferStart - d, d);
  }
  mBufferStart = start;
}

// Head =======================================================================

nam::wavenet::Head::Head(const int inputSize, const int numLayers, const int channels, const std::string activation)
: mChannels(channels)
, mHead(numLayers > 0 ? channels : inputSize, 1, true)
, mActivation(activations::Activation::GetActivation(activation))
{
  assert(numLayers > 0);
  int dx = inputSize;
  for (int i = 0; i < numLayers; i++)
  {
    mLayers.push_back(Conv1x1(dx, i == numLayers - 1 ? 1 : channels, true));
    dx = channels;
    if (i < numLayers - 1)
      mBuffers.push_back(Eigen::MatrixXf());
  }
}

void nam::wavenet::Head::SetWeights(weights_it& weights)
{
  for (size_t i = 0; i < mLayers.size(); i++)
    mLayers[i].SetWeights(weights);
}

void nam::wavenet::Head::Process(Eigen::Ref<Eigen::MatrixXf> inputs, Eigen::Ref<Eigen::MatrixXf> outputs)
{
  const size_t numLayers = mLayers.size();
  ApplyActivation(inputs);
  if (numLayers == 1)
    outputs = mLayers[0].Process(inputs);
  else
  {
    mBuffers[0] = mLayers[0].Process(inputs);
    for (size_t i = 1; i < numLayers; i++)
    { // Asserted > 0 layers
      ApplyActivation(mBuffers[i - 1]);
      if (i < numLayers - 1)
        mBuffers[i] = mLayers[i].Process(mBuffers[i - 1]);
      else
        outputs = mLayers[i].Process(mBuffers[i - 1]);
    }
  }
}

void nam::wavenet::Head::SetNumFrames(const long numFrames)
{
  for (size_t i = 0; i < mBuffers.size(); i++)
  {
    if (mBuffers[i].rows() == mChannels && mBuffers[i].cols() == numFrames)
      continue; // Already has correct size
    mBuffers[i].resize(mChannels, numFrames);
    mBuffers[i].setZero();
  }
}

void nam::wavenet::Head::ApplyActivation(Eigen::Ref<Eigen::MatrixXf> x)
{
  mActivation->Apply(x);
}

// WaveNet ====================================================================

nam::wavenet::WaveNet::WaveNet(const std::vector<nam::wavenet::LayerArrayParams>& layerArrayParams,
                               const float headScale, const bool withHead, const std::vector<float>& weights,
                               const double expectedSampleRate)
: DSP(expectedSampleRate)
, mNumFrames(0)
, mHeadScale(headScale)
{
  if (withHead)
    throw std::runtime_error("Head not implemented!");
  for (size_t i = 0; i < layerArrayParams.size(); i++)
  {
    mLayerArrays.push_back(nam::wavenet::LayerArray(
      layerArrayParams[i].mInputSize, layerArrayParams[i].mConditionSize, layerArrayParams[i].mHeadSize,
      layerArrayParams[i].mChannels, layerArrayParams[i].mKernelSize, layerArrayParams[i].mDilations,
      layerArrayParams[i].mActivation, layerArrayParams[i].mGated, layerArrayParams[i].mHeadBias));
    mLayerArrayOutputs.push_back(Eigen::MatrixXf(layerArrayParams[i].mChannels, 0));
    if (i == 0)
      mHeadArrays.push_back(Eigen::MatrixXf(layerArrayParams[i].mChannels, 0));
    if (i > 0)
      if (layerArrayParams[i].mChannels != layerArrayParams[i - 1].mHeadSize)
      {
        std::stringstream ss;
        ss << "channels of layer " << i << " (" << layerArrayParams[i].mChannels
           << ") doesn't match head_size of preceding layer (" << layerArrayParams[i - 1].mHeadSize << "!\n";
        throw std::runtime_error(ss.str().c_str());
      }
    mHeadArrays.push_back(Eigen::MatrixXf(layerArrayParams[i].mHeadSize, 0));
  }
  mHeadOutput.resize(1, 0); // Mono output!
  SetWeights(weights);

  mPrewarmSamples = 1;
  for (size_t i = 0; i < mLayerArrays.size(); i++)
    mPrewarmSamples += mLayerArrays[i].GetReceptiveField();
}

void nam::wavenet::WaveNet::Finalize(const int numFrames)
{
  DSP::Finalize(numFrames);
  AdvanceBuffers(numFrames);
}

void nam::wavenet::WaveNet::SetWeights(const std::vector<float>& weights)
{
  weights_it it = weights.begin();
  for (size_t i = 0; i < mLayerArrays.size(); i++)
    mLayerArrays[i].SetWeights(it);
  // _head.set_params_(it);
  mHeadScale = *(it++);
  if (it != weights.end())
  {
    std::stringstream ss;
    for (size_t i = 0; i < weights.size(); i++)
      if (weights[i] == *it)
      {
        ss << "Weight mismatch: assigned " << i + 1 << " weights, but " << weights.size() << " were provided.";
        throw std::runtime_error(ss.str().c_str());
      }
    ss << "Weight mismatch: provided " << weights.size() << " weights, but the model expects more.";
    throw std::runtime_error(ss.str().c_str());
  }
}

void nam::wavenet::WaveNet::AdvanceBuffers(const int numFrames)
{
  for (size_t i = 0; i < mLayerArrays.size(); i++)
    mLayerArrays[i].AdvanceBuffers(numFrames);
}

void nam::wavenet::WaveNet::PrepareForFrames(const long numFrames)
{
  for (size_t i = 0; i < mLayerArrays.size(); i++)
    mLayerArrays[i].PrepareForFrames(numFrames);
}

void nam::wavenet::WaveNet::SetConditionArray(float* input, const int numFrames)
{
  for (int j = 0; j < numFrames; j++)
  {
    mCondition(0, j) = input[j];
  }
}

void nam::wavenet::WaveNet::Process(float* input, float* output, const int numFrames)
{
  SetNumFrames(numFrames);
  PrepareForFrames(numFrames);
  SetConditionArray(input, numFrames);

  // Main layer arrays:
  // Layer-to-layer
  // Sum on head output
  mHeadArrays[0].setZero();
  for (size_t i = 0; i < mLayerArrays.size(); i++)
    mLayerArrays[i].Process(i == 0 ? mCondition : mLayerArrayOutputs[i - 1], mCondition,
                                    mHeadArrays[i], mLayerArrayOutputs[i], mHeadArrays[i + 1]);
  // _head.Process(
  //   _head_input,
  //   _head_output
  //);
  //  Copy to required output array
  //  Hack: apply head scale here; revisit when/if I activate the head.
  //  assert(_head_output.rows() == 1);

  const long final_head_array = mHeadArrays.size() - 1;
  assert(mHeadArrays[final_head_array].rows() == 1);
  for (int s = 0; s < numFrames; s++)
  {
    float out = mHeadScale * mHeadArrays[final_head_array](0, s);
    output[s] = out;
  }
}

void nam::wavenet::WaveNet::SetNumFrames(const long numFrames)
{
  if (numFrames == mNumFrames)
    return;

  mCondition.resize(GetConditionDim(), numFrames);
  for (size_t i = 0; i < mHeadArrays.size(); i++)
    mHeadArrays[i].resize(mHeadArrays[i].rows(), numFrames);
  for (size_t i = 0; i < mLayerArrayOutputs.size(); i++)
    mLayerArrayOutputs[i].resize(mLayerArrayOutputs[i].rows(), numFrames);
  mHeadOutput.resize(mHeadOutput.rows(), numFrames);
  mHeadOutput.setZero();

  for (size_t i = 0; i < mLayerArrays.size(); i++)
    mLayerArrays[i].SetNumFrames(numFrames);
  // _head.SetNumFrames(numFrames);
  mNumFrames = numFrames;
}
