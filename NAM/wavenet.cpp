#include <algorithm>
#include <iostream>
#include <math.h>

#include <Eigen/Dense>

#include "wavenet.h"

nam::wavenet::_DilatedConv::_DilatedConv(const int inChannels, const int outChannels, const int kernelSize,
                                         const int bias, const int dilation)
{
  this->SetSize(inChannels, outChannels, kernelSize, bias, dilation);
}

void nam::wavenet::_Layer::SetWeights(weights_it& weights)
{
  this->_conv.SetWeights(weights);
  this->_input_mixin.SetWeights(weights);
  this->_1x1.SetWeights(weights);
}

void nam::wavenet::_Layer::Process(const Eigen::Ref<const Eigen::MatrixXf> input, const Eigen::Ref<const Eigen::MatrixXf> condition,
                                    Eigen::Ref<Eigen::MatrixXf> head_input, Eigen::Ref<Eigen::MatrixXf> output, const long i_start,
                                    const long j_start)
{
  const long ncols = condition.cols();
  const long channels = this->get_channels();
  // Input dilated conv
  this->_conv.Process(input, this->_z, i_start, ncols, 0);
  // Mix-in condition
  this->_z += this->_input_mixin.Process(condition);

  this->_activation->Apply(this->_z);

  if (this->_gated)
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

void nam::wavenet::_Layer::set_num_frames_(const long numFrames)
{
  if (this->_z.rows() == this->_conv.GetOutChannels() && this->_z.cols() == numFrames)
    return; // Already has correct size

  this->_z.resize(this->_conv.GetOutChannels(), numFrames);
  this->_z.setZero();
}

// LayerArray =================================================================

#define LAYER_ARRAY_BUFFER_SIZE 65536

nam::wavenet::_LayerArray::_LayerArray(const int inputSize, const int condition_size, const int head_size,
                                       const int channels, const int kernelSize, const std::vector<int>& dilations,
                                       const std::string activation, const bool gated, const bool head_bias)
: _rechannel(inputSize, channels, false)
, _head_rechannel(channels, head_size, head_bias)
{
  for (size_t i = 0; i < dilations.size(); i++)
    this->_layers.push_back(_Layer(condition_size, channels, kernelSize, dilations[i], activation, gated));
  const long receptiveField = this->_get_receptive_field();
  for (size_t i = 0; i < dilations.size(); i++)
  {
    this->_layer_buffers.push_back(Eigen::MatrixXf(channels, LAYER_ARRAY_BUFFER_SIZE + receptiveField - 1));
    this->_layer_buffers[i].setZero();
  }
  this->_buffer_start = this->_get_receptive_field() - 1;
}

void nam::wavenet::_LayerArray::advance_buffers_(const int numFrames)
{
  this->_buffer_start += numFrames;
}

long nam::wavenet::_LayerArray::get_receptive_field() const
{
  long result = 0;
  for (size_t i = 0; i < this->_layers.size(); i++)
    result += this->_layers[i].GetDilation() * (this->_layers[i].GetKernelSize() - 1);
  return result;
}

void nam::wavenet::_LayerArray::prepare_for_frames_(const long numFrames)
{
  // Example:
  // _buffer_start = 0
  // numFrames = 64
  // buffer_size = 64
  // -> this will write on indices 0 through 63, inclusive.
  // -> No illegal writes.
  // -> no rewind needed.
  if (this->_buffer_start + numFrames > this->_get_buffer_size())
    this->RewindBuffers();
}

void nam::wavenet::_LayerArray::Process(const Eigen::Ref<const Eigen::MatrixXf> layer_inputs, const Eigen::Ref<const Eigen::MatrixXf> condition,
                                         Eigen::Ref<Eigen::MatrixXf> head_inputs, Eigen::Ref<Eigen::MatrixXf> layer_outputs,
                                         Eigen::Ref<Eigen::MatrixXf> head_outputs)
{
  this->_layer_buffers[0].middleCols(this->_buffer_start, layer_inputs.cols()) = this->_rechannel.Process(layer_inputs);
  const size_t last_layer = this->_layers.size() - 1;
  for (size_t i = 0; i < this->_layers.size(); i++)
  {
    if (i == last_layer)
    {
      this->_layers[i].Process(this->_layer_buffers[i], condition, head_inputs,
                                layer_outputs, this->_buffer_start,
                                0);
    }
    else
    {
      this->_layers[i].Process(this->_layer_buffers[i], condition, head_inputs,
                                this->_layer_buffers[i + 1], this->_buffer_start,
                                this->_buffer_start);
    }

  }
  head_outputs = this->_head_rechannel.Process(head_inputs);
}

void nam::wavenet::_LayerArray::set_num_frames_(const long numFrames)
{
  // Wavenet checks for unchanged numFrames; if we made it here, there's
  // something to do.
  if (LAYER_ARRAY_BUFFER_SIZE - numFrames < this->_get_receptive_field())
  {
    std::stringstream ss;
    ss << "Asked to accept a buffer of " << numFrames << " samples, but the buffer is too short ("
       << LAYER_ARRAY_BUFFER_SIZE << ") to get out of the recptive field (" << this->_get_receptive_field()
       << "); copy errors could occur!\n";
    throw std::runtime_error(ss.str().c_str());
  }
  for (size_t i = 0; i < this->_layers.size(); i++)
    this->_layers[i].set_num_frames_(numFrames);
}

void nam::wavenet::_LayerArray::SetWeights(weights_it& weights)
{
  this->_rechannel.SetWeights(weights);
  for (size_t i = 0; i < this->_layers.size(); i++)
    this->_layers[i].SetWeights(weights);
  this->_head_rechannel.SetWeights(weights);
}

long nam::wavenet::_LayerArray::_get_channels() const
{
  return this->_layers.size() > 0 ? this->_layers[0].get_channels() : 0;
}

long nam::wavenet::_LayerArray::_get_receptive_field() const
{
  // TODO remove this and use get_receptive_field() instead!
  long res = 1;
  for (size_t i = 0; i < this->_layers.size(); i++)
    res += (this->_layers[i].GetKernelSize() - 1) * this->_layers[i].GetDilation();
  return res;
}

void nam::wavenet::_LayerArray::RewindBuffers()
// Consider wrapping instead...
// Can make this smaller--largest dilation, not receptive field!
{
  const long start = this->_get_receptive_field() - 1;
  for (size_t i = 0; i < this->_layer_buffers.size(); i++)
  {
    const long d = (this->_layers[i].GetKernelSize() - 1) * this->_layers[i].GetDilation();
    this->_layer_buffers[i].middleCols(start - d, d) = this->_layer_buffers[i].middleCols(this->_buffer_start - d, d);
  }
  this->_buffer_start = start;
}

// Head =======================================================================

nam::wavenet::Head::Head(const int inputSize, const int numLayers, const int channels, const std::string activation)
: _channels(channels)
, _head(numLayers > 0 ? channels : inputSize, 1, true)
, _activation(activations::Activation::GetActivation(activation))
{
  assert(numLayers > 0);
  int dx = inputSize;
  for (int i = 0; i < numLayers; i++)
  {
    this->_layers.push_back(Conv1x1(dx, i == numLayers - 1 ? 1 : channels, true));
    dx = channels;
    if (i < numLayers - 1)
      this->_buffers.push_back(Eigen::MatrixXf());
  }
}

void nam::wavenet::Head::SetWeights(weights_it& weights)
{
  for (size_t i = 0; i < this->_layers.size(); i++)
    this->_layers[i].SetWeights(weights);
}

void nam::wavenet::Head::Process(Eigen::Ref<Eigen::MatrixXf> inputs, Eigen::Ref<Eigen::MatrixXf> outputs)
{
  const size_t numLayers = this->_layers.size();
  this->_apply_activation_(inputs);
  if (numLayers == 1)
    outputs = this->_layers[0].Process(inputs);
  else
  {
    this->_buffers[0] = this->_layers[0].Process(inputs);
    for (size_t i = 1; i < numLayers; i++)
    { // Asserted > 0 layers
      this->_apply_activation_(this->_buffers[i - 1]);
      if (i < numLayers - 1)
        this->_buffers[i] = this->_layers[i].Process(this->_buffers[i - 1]);
      else
        outputs = this->_layers[i].Process(this->_buffers[i - 1]);
    }
  }
}

void nam::wavenet::Head::set_num_frames_(const long numFrames)
{
  for (size_t i = 0; i < this->_buffers.size(); i++)
  {
    if (this->_buffers[i].rows() == this->_channels && this->_buffers[i].cols() == numFrames)
      continue; // Already has correct size
    this->_buffers[i].resize(this->_channels, numFrames);
    this->_buffers[i].setZero();
  }
}

void nam::wavenet::Head::_apply_activation_(Eigen::Ref<Eigen::MatrixXf> x)
{
  this->_activation->Apply(x);
}

// WaveNet ====================================================================

nam::wavenet::WaveNet::WaveNet(const std::vector<nam::wavenet::LayerArrayParams>& layer_array_params,
                               const float head_scale, const bool with_head, const std::vector<float>& weights,
                               const double expectedSampleRate)
: DSP(expectedSampleRate)
, _num_frames(0)
, _head_scale(head_scale)
{
  if (with_head)
    throw std::runtime_error("Head not implemented!");
  for (size_t i = 0; i < layer_array_params.size(); i++)
  {
    this->_layer_arrays.push_back(nam::wavenet::_LayerArray(
      layer_array_params[i].inputSize, layer_array_params[i].condition_size, layer_array_params[i].head_size,
      layer_array_params[i].channels, layer_array_params[i].kernelSize, layer_array_params[i].dilations,
      layer_array_params[i].activation, layer_array_params[i].gated, layer_array_params[i].head_bias));
    this->_layer_array_outputs.push_back(Eigen::MatrixXf(layer_array_params[i].channels, 0));
    if (i == 0)
      this->_head_arrays.push_back(Eigen::MatrixXf(layer_array_params[i].channels, 0));
    if (i > 0)
      if (layer_array_params[i].channels != layer_array_params[i - 1].head_size)
      {
        std::stringstream ss;
        ss << "channels of layer " << i << " (" << layer_array_params[i].channels
           << ") doesn't match head_size of preceding layer (" << layer_array_params[i - 1].head_size << "!\n";
        throw std::runtime_error(ss.str().c_str());
      }
    this->_head_arrays.push_back(Eigen::MatrixXf(layer_array_params[i].head_size, 0));
  }
  this->mHeadOutput.resize(1, 0); // Mono output!
  this->SetWeights(weights);

  mPrewarmSamples = 1;
  for (size_t i = 0; i < this->_layer_arrays.size(); i++)
    mPrewarmSamples += this->_layer_arrays[i].get_receptive_field();
}

void nam::wavenet::WaveNet::Finalize(const int numFrames)
{
  this->DSP::Finalize(numFrames);
  this->AdvanceBuffers(numFrames);
}

void nam::wavenet::WaveNet::SetWeights(const std::vector<float>& weights)
{
  weights_it it = weights.begin();
  for (size_t i = 0; i < this->_layer_arrays.size(); i++)
    this->_layer_arrays[i].SetWeights(it);
  // this->_head.set_params_(it);
  this->_head_scale = *(it++);
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
  for (size_t i = 0; i < this->_layer_arrays.size(); i++)
    this->_layer_arrays[i].advance_buffers_(numFrames);
}

void nam::wavenet::WaveNet::PrepareForFrames(const long numFrames)
{
  for (size_t i = 0; i < this->_layer_arrays.size(); i++)
    this->_layer_arrays[i].prepare_for_frames_(numFrames);
}

void nam::wavenet::WaveNet::SetConditionArray(float* input, const int numFrames)
{
  for (int j = 0; j < numFrames; j++)
  {
    this->_condition(0, j) = input[j];
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
  this->_head_arrays[0].setZero();
  for (size_t i = 0; i < this->_layer_arrays.size(); i++)
    this->_layer_arrays[i].Process(i == 0 ? this->_condition : this->_layer_array_outputs[i - 1], this->_condition,
                                    this->_head_arrays[i], this->_layer_array_outputs[i], this->_head_arrays[i + 1]);
  // this->_head.Process(
  //   this->_head_input,
  //   this->_head_output
  //);
  //  Copy to required output array
  //  Hack: apply head scale here; revisit when/if I activate the head.
  //  assert(this->_head_output.rows() == 1);

  const long final_head_array = this->_head_arrays.size() - 1;
  assert(this->_head_arrays[final_head_array].rows() == 1);
  for (int s = 0; s < numFrames; s++)
  {
    float out = this->_head_scale * this->_head_arrays[final_head_array](0, s);
    output[s] = out;
  }
}

void nam::wavenet::WaveNet::SetNumFrames(const long numFrames)
{
  if (numFrames == this->_num_frames)
    return;

  this->_condition.resize(this->GetConditionDim(), numFrames);
  for (size_t i = 0; i < this->_head_arrays.size(); i++)
    this->_head_arrays[i].resize(this->_head_arrays[i].rows(), numFrames);
  for (size_t i = 0; i < this->_layer_array_outputs.size(); i++)
    this->_layer_array_outputs[i].resize(this->_layer_array_outputs[i].rows(), numFrames);
  this->mHeadOutput.resize(this->mHeadOutput.rows(), numFrames);
  this->mHeadOutput.setZero();

  for (size_t i = 0; i < this->_layer_arrays.size(); i++)
    this->_layer_arrays[i].set_num_frames_(numFrames);
  // this->_head.set_num_frames_(numFrames);
  this->_num_frames = numFrames;
}
