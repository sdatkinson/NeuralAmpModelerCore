#include <algorithm>
#include <iostream>
#include <math.h>

#include <Eigen/Dense>

#include "wavenet.h"

nam::wavenet::DilatedConv::DilatedConv(const int inChannels, const int outChannels, const int kernelSize,
                                         const int bias, const int dilation)
{
  this->SetSize(inChannels, outChannels, kernelSize, bias, dilation);
}

void nam::wavenet::Layer::SetWeights(weights_it& weights)
{
  this->mConv.SetWeights(weights);
  this->mInputMixin.SetWeights(weights);
  this->_1x1.SetWeights(weights);
}

void nam::wavenet::Layer::Process(const Eigen::Ref<const Eigen::MatrixXf> input, const Eigen::Ref<const Eigen::MatrixXf> condition,
                                    Eigen::Ref<Eigen::MatrixXf> head_input, Eigen::Ref<Eigen::MatrixXf> output, const long i_start,
                                    const long j_start)
{
  const long ncols = condition.cols();
  const long channels = this->GetChannels();
  // Input dilated conv
  this->mConv.Process(input, this->_z, i_start, ncols, 0);
  // Mix-in condition
  this->_z += this->mInputMixin.Process(condition);

  this->mActivation->Apply(this->_z);

  if (this->mGated)
  {
    activations::Activation::GetActivation("Sigmoid")->Apply(this->_z.block(channels, 0, channels, this->_z.cols()));

    this->_z.topRows(channels).array() *= this->_z.bottomRows(channels).array();
    // this->_z.topRows(channels) = this->_z.topRows(channels).cwiseProduct(
    //   this->_z.bottomRows(channels)
    // );
  }

  head_input += this->_z.topRows(channels);
  output.middleCols(j_start, ncols) = input.middleCols(i_start, ncols) + this->_1x1.Process(this->_z.topRows(channels));
}

void nam::wavenet::Layer::SetNumFrames(const long numFrames)
{
  if (this->_z.rows() == this->mConv.GetOutChannels() && this->_z.cols() == numFrames)
    return; // Already has correct size

  this->_z.resize(this->mConv.GetOutChannels(), numFrames);
  this->_z.setZero();
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
    this->mLayers.push_back(Layer(condition_size, channels, kernelSize, dilations[i], activation, gated));
  const long receptiveField = this->GetReceptiveField();
  for (size_t i = 0; i < dilations.size(); i++)
  {
    this->mLayerBuffers.push_back(Eigen::MatrixXf(channels, LAYER_ARRAY_BUFFER_SIZE + receptiveField - 1));
    this->mLayerBuffers[i].setZero();
  }
  this->mBufferStart = this->GetReceptiveField() - 1;
}

void nam::wavenet::LayerArray::AdvanceBuffers(const int numFrames)
{
  this->mBufferStart += numFrames;
}

long nam::wavenet::LayerArray::GetReceptiveField() const
{
  long result = 0;
  for (size_t i = 0; i < this->mLayers.size(); i++)
    result += this->mLayers[i].GetDilation() * (this->mLayers[i].GetKernelSize() - 1);
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
  if (this->mBufferStart + numFrames > this->GetBufferSize())
    this->RewindBuffers();
}

void nam::wavenet::LayerArray::Process(const Eigen::Ref<const Eigen::MatrixXf> layerInputs, const Eigen::Ref<const Eigen::MatrixXf> condition,
                                         Eigen::Ref<Eigen::MatrixXf> headInputs, Eigen::Ref<Eigen::MatrixXf> layerOutputs,
                                         Eigen::Ref<Eigen::MatrixXf> headOutputs)
{
  this->mLayerBuffers[0].middleCols(this->mBufferStart, layerInputs.cols()) = this->mReChannel.Process(layerInputs);
  const size_t last_layer = this->mLayers.size() - 1;
  for (size_t i = 0; i < this->mLayers.size(); i++)
  {
    if (i == last_layer)
    {
      this->mLayers[i].Process(this->mLayerBuffers[i], condition, headInputs,
                                layerOutputs, this->mBufferStart,
                                0);
    }
    else
    {
      this->mLayers[i].Process(this->mLayerBuffers[i], condition, headInputs,
                                this->mLayerBuffers[i + 1], this->mBufferStart,
                                this->mBufferStart);
    }

  }
  headOutputs = this->mHeadRechannel.Process(headInputs);
}

void nam::wavenet::LayerArray::SetNumFrames(const long numFrames)
{
  // Wavenet checks for unchanged numFrames; if we made it here, there's
  // something to do.
  if (LAYER_ARRAY_BUFFER_SIZE - numFrames < this->GetReceptiveField())
  {
    std::stringstream ss;
    ss << "Asked to accept a buffer of " << numFrames << " samples, but the buffer is too short ("
       << LAYER_ARRAY_BUFFER_SIZE << ") to get out of the recptive field (" << this->GetReceptiveField()
       << "); copy errors could occur!\n";
    throw std::runtime_error(ss.str().c_str());
  }
  for (size_t i = 0; i < this->mLayers.size(); i++)
    this->mLayers[i].SetNumFrames(numFrames);
}

void nam::wavenet::LayerArray::SetWeights(weights_it& weights)
{
  this->mReChannel.SetWeights(weights);
  for (size_t i = 0; i < this->mLayers.size(); i++)
    this->mLayers[i].SetWeights(weights);
  this->mHeadRechannel.SetWeights(weights);
}

long nam::wavenet::LayerArray::GetChannels() const
{
  return this->mLayers.size() > 0 ? this->mLayers[0].GetChannels() : 0;
}

long nam::wavenet::LayerArray::_GetReceptiveField() const // TODO: why two?
{
  // TODO remove this and use GetReceptiveField() instead!
  long res = 1;
  for (size_t i = 0; i < this->mLayers.size(); i++)
    res += (this->mLayers[i].GetKernelSize() - 1) * this->mLayers[i].GetDilation();
  return res;
}

void nam::wavenet::LayerArray::RewindBuffers()
// Consider wrapping instead...
// Can make this smaller--largest dilation, not receptive field!
{
  const long start = this->GetReceptiveField() - 1;
  for (size_t i = 0; i < this->mLayerBuffers.size(); i++)
  {
    const long d = (this->mLayers[i].GetKernelSize() - 1) * this->mLayers[i].GetDilation();
    this->mLayerBuffers[i].middleCols(start - d, d) = this->mLayerBuffers[i].middleCols(this->mBufferStart - d, d);
  }
  this->mBufferStart = start;
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
    this->mLayers.push_back(Conv1x1(dx, i == numLayers - 1 ? 1 : channels, true));
    dx = channels;
    if (i < numLayers - 1)
      this->mBuffers.push_back(Eigen::MatrixXf());
  }
}

void nam::wavenet::Head::SetWeights(weights_it& weights)
{
  for (size_t i = 0; i < this->mLayers.size(); i++)
    this->mLayers[i].SetWeights(weights);
}

void nam::wavenet::Head::Process(Eigen::Ref<Eigen::MatrixXf> inputs, Eigen::Ref<Eigen::MatrixXf> outputs)
{
  const size_t numLayers = this->mLayers.size();
  this->ApplyActivation(inputs);
  if (numLayers == 1)
    outputs = this->mLayers[0].Process(inputs);
  else
  {
    this->mBuffers[0] = this->mLayers[0].Process(inputs);
    for (size_t i = 1; i < numLayers; i++)
    { // Asserted > 0 layers
      this->ApplyActivation(this->mBuffers[i - 1]);
      if (i < numLayers - 1)
        this->mBuffers[i] = this->mLayers[i].Process(this->mBuffers[i - 1]);
      else
        outputs = this->mLayers[i].Process(this->mBuffers[i - 1]);
    }
  }
}

void nam::wavenet::Head::SetNumFrames(const long numFrames)
{
  for (size_t i = 0; i < this->mBuffers.size(); i++)
  {
    if (this->mBuffers[i].rows() == this->mChannels && this->mBuffers[i].cols() == numFrames)
      continue; // Already has correct size
    this->mBuffers[i].resize(this->mChannels, numFrames);
    this->mBuffers[i].setZero();
  }
}

void nam::wavenet::Head::ApplyActivation(Eigen::Ref<Eigen::MatrixXf> x)
{
  this->mActivation->Apply(x);
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
    this->mLayerArrays.push_back(nam::wavenet::LayerArray(
      layerArrayParams[i].mInputSize, layerArrayParams[i].mConditionSize, layerArrayParams[i].mHeadSize,
      layerArrayParams[i].mChannels, layerArrayParams[i].mKernelSize, layerArrayParams[i].mDilations,
      layerArrayParams[i].mActivation, layerArrayParams[i].mGated, layerArrayParams[i].mHeadBias));
    this->mLayerArrayOutputs.push_back(Eigen::MatrixXf(layerArrayParams[i].mChannels, 0));
    if (i == 0)
      this->mHeadArrays.push_back(Eigen::MatrixXf(layerArrayParams[i].mChannels, 0));
    if (i > 0)
      if (layerArrayParams[i].mChannels != layerArrayParams[i - 1].mHeadSize)
      {
        std::stringstream ss;
        ss << "channels of layer " << i << " (" << layerArrayParams[i].mChannels
           << ") doesn't match head_size of preceding layer (" << layerArrayParams[i - 1].mHeadSize << "!\n";
        throw std::runtime_error(ss.str().c_str());
      }
    this->mHeadArrays.push_back(Eigen::MatrixXf(layerArrayParams[i].mHeadSize, 0));
  }
  this->mHeadOutput.resize(1, 0); // Mono output!
  this->SetWeights(weights);

  mPrewarmSamples = 1;
  for (size_t i = 0; i < this->mLayerArrays.size(); i++)
    mPrewarmSamples += this->mLayerArrays[i].GetReceptiveField();
}

void nam::wavenet::WaveNet::Finalize(const int numFrames)
{
  this->DSP::Finalize(numFrames);
  this->AdvanceBuffers(numFrames);
}

void nam::wavenet::WaveNet::SetWeights(const std::vector<float>& weights)
{
  weights_it it = weights.begin();
  for (size_t i = 0; i < this->mLayerArrays.size(); i++)
    this->mLayerArrays[i].SetWeights(it);
  // this->_head.set_params_(it);
  this->mHeadScale = *(it++);
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
  for (size_t i = 0; i < this->mLayerArrays.size(); i++)
    this->mLayerArrays[i].AdvanceBuffers(numFrames);
}

void nam::wavenet::WaveNet::PrepareForFrames(const long numFrames)
{
  for (size_t i = 0; i < this->mLayerArrays.size(); i++)
    this->mLayerArrays[i].PrepareForFrames(numFrames);
}

void nam::wavenet::WaveNet::SetConditionArray(float* input, const int numFrames)
{
  for (int j = 0; j < numFrames; j++)
  {
    this->mCondition(0, j) = input[j];
  }
}

void nam::wavenet::WaveNet::Process(float* input, float* output, const int numFrames)
{
  this->SetNumFrames(numFrames);
  this->PrepareForFrames(numFrames);
  this->SetConditionArray(input, numFrames);

  // Main layer arrays:
  // Layer-to-layer
  // Sum on head output
  this->mHeadArrays[0].setZero();
  for (size_t i = 0; i < this->mLayerArrays.size(); i++)
    this->mLayerArrays[i].Process(i == 0 ? this->mCondition : this->mLayerArrayOutputs[i - 1], this->mCondition,
                                    this->mHeadArrays[i], this->mLayerArrayOutputs[i], this->mHeadArrays[i + 1]);
  // this->_head.Process(
  //   this->_head_input,
  //   this->_head_output
  //);
  //  Copy to required output array
  //  Hack: apply head scale here; revisit when/if I activate the head.
  //  assert(this->_head_output.rows() == 1);

  const long final_head_array = this->mHeadArrays.size() - 1;
  assert(this->mHeadArrays[final_head_array].rows() == 1);
  for (int s = 0; s < numFrames; s++)
  {
    float out = this->mHeadScale * this->mHeadArrays[final_head_array](0, s);
    output[s] = out;
  }
}

void nam::wavenet::WaveNet::SetNumFrames(const long numFrames)
{
  if (numFrames == this->mNumFrames)
    return;

  this->mCondition.resize(this->GetConditionDim(), numFrames);
  for (size_t i = 0; i < this->mHeadArrays.size(); i++)
    this->mHeadArrays[i].resize(this->mHeadArrays[i].rows(), numFrames);
  for (size_t i = 0; i < this->mLayerArrayOutputs.size(); i++)
    this->mLayerArrayOutputs[i].resize(this->mLayerArrayOutputs[i].rows(), numFrames);
  this->mHeadOutput.resize(this->mHeadOutput.rows(), numFrames);
  this->mHeadOutput.setZero();

  for (size_t i = 0; i < this->mLayerArrays.size(); i++)
    this->mLayerArrays[i].SetNumFrames(numFrames);
  // this->_head.SetNumFrames(numFrames);
  this->mNumFrames = numFrames;
}
