#include <algorithm>
#include <iostream>
#include <math.h>

#include <Eigen/Dense>

#include "registry.h"
#include "wavenet.h"

// Layer ======================================================================

void nam::wavenet::_Layer::SetMaxBufferSize(const int maxBufferSize)
{
  _conv.SetMaxBufferSize(maxBufferSize);
  _input_mixin.SetMaxBufferSize(maxBufferSize);
  _z.resize(this->_conv.get_out_channels(), maxBufferSize);
  _z.setZero();
  _1x1.SetMaxBufferSize(maxBufferSize);
  // Pre-allocate output buffers
  const long channels = this->get_channels();
  this->_output_next_layer.resize(channels, maxBufferSize);
  this->_output_head.resize(channels, maxBufferSize);
}

void nam::wavenet::_Layer::set_weights_(std::vector<float>::iterator& weights)
{
  this->_conv.set_weights_(weights);
  this->_input_mixin.set_weights_(weights);
  this->_1x1.set_weights_(weights);
}

void nam::wavenet::_Layer::Process(const Eigen::MatrixXf& input, const Eigen::MatrixXf& condition, const int num_frames)
{
  const long channels = this->get_channels();

  // Step 1: input convolutions
  this->_conv.Process(input, num_frames);
  this->_input_mixin.process_(condition, num_frames);
  this->_z.leftCols(num_frames) = this->_conv.GetOutput(num_frames) + _input_mixin.GetOutput(num_frames);

  // Step 2 & 3: activation and 1x1
  if (!this->_gated)
  {
    this->_activation->apply(this->_z.leftCols(num_frames));
    _1x1.process_(_z, num_frames);
  }
  else
  {
    // CAREFUL: .topRows() and .bottomRows() won't be memory-contiguous for a column-major matrix (Issue 125). Need to
    // do this column-wise:
    for (int i = 0; i < num_frames; i++)
    {
      this->_activation->apply(this->_z.block(0, i, channels, 1));
      activations::Activation::get_activation("Sigmoid")->apply(this->_z.block(channels, i, channels, 1));
    }
    this->_z.block(0, 0, channels, num_frames).array() *= this->_z.block(channels, 0, channels, num_frames).array();
    _1x1.process_(_z.topRows(channels), num_frames); // Might not be RT safe
  }

  // Store output to head (skip connection: activated conv output)
  this->_output_head.leftCols(num_frames) = this->_z.leftCols(num_frames);
  // Store output to next layer (residual connection: input + _1x1 output)
  this->_output_next_layer.leftCols(num_frames).noalias() = input.leftCols(num_frames) + _1x1.GetOutput(num_frames);
}

Eigen::Block<Eigen::MatrixXf> nam::wavenet::_Layer::GetOutputNextLayer(const int num_frames)
{
  // FIXME use leftCols?
  return this->_output_next_layer.block(0, 0, this->_output_next_layer.rows(), num_frames);
}

Eigen::Block<Eigen::MatrixXf> nam::wavenet::_Layer::GetOutputHead(const int num_frames)
{
  return this->_output_head.block(0, 0, this->_output_head.rows(), num_frames);
}

// LayerArray =================================================================

nam::wavenet::_LayerArray::_LayerArray(const int input_size, const int condition_size, const int head_size,
                                       const int channels, const int kernel_size, const std::vector<int>& dilations,
                                       const std::string activation, const bool gated, const bool head_bias)
: _rechannel(input_size, channels, false)
, _head_rechannel(channels, head_size, head_bias)
{
  for (size_t i = 0; i < dilations.size(); i++)
    this->_layers.push_back(_Layer(condition_size, channels, kernel_size, dilations[i], activation, gated));
}

void nam::wavenet::_LayerArray::SetMaxBufferSize(const int maxBufferSize)
{
  _rechannel.SetMaxBufferSize(maxBufferSize);
  _head_rechannel.SetMaxBufferSize(maxBufferSize);
  for (auto it = _layers.begin(); it != _layers.end(); ++it)
  {
    it->SetMaxBufferSize(maxBufferSize);
  }
  // Pre-allocate output buffers
  const long channels = this->_get_channels();
  this->_layer_outputs.resize(channels, maxBufferSize);
  this->_head_inputs.resize(channels, maxBufferSize);
}


long nam::wavenet::_LayerArray::get_receptive_field() const
{
  long result = 0;
  for (size_t i = 0; i < this->_layers.size(); i++)
    result += this->_layers[i].get_dilation() * (this->_layers[i].get_kernel_size() - 1);
  return result;
}


void nam::wavenet::_LayerArray::Process(const Eigen::MatrixXf& layer_inputs, const Eigen::MatrixXf& condition,
                                        const int num_frames)
{
  // Zero head inputs accumulator (first layer array)
  this->_head_inputs.setZero();
  ProcessInner(layer_inputs, condition, num_frames);
}

void nam::wavenet::_LayerArray::Process(const Eigen::MatrixXf& layer_inputs, const Eigen::MatrixXf& condition,
                                        const Eigen::MatrixXf& head_inputs, const int num_frames)
{
  // Copy head inputs from previous layer array
  this->_head_inputs.leftCols(num_frames) = head_inputs.leftCols(num_frames);
  ProcessInner(layer_inputs, condition, num_frames);
}

void nam::wavenet::_LayerArray::ProcessInner(const Eigen::MatrixXf& layer_inputs, const Eigen::MatrixXf& condition,
                                             const int num_frames)
{
  // Process rechannel and get output
  this->_rechannel.process_(layer_inputs, num_frames);
  auto rechannel_output = _rechannel.GetOutput(num_frames);

  // Process layers
  for (size_t i = 0; i < this->_layers.size(); i++)
  {
    // Get input for this layer
    Eigen::MatrixXf layer_input = i == 0 ? rechannel_output : this->_layers[i - 1].GetOutputNextLayer(num_frames);
    // Process layer
    this->_layers[i].Process(layer_input, condition, num_frames);

    // Accumulate head output from this layer
    this->_head_inputs.leftCols(num_frames).noalias() += this->_layers[i].GetOutputHead(num_frames);
  }

  // Store output from last layer
  const size_t last_layer = this->_layers.size() - 1;
  this->_layer_outputs.leftCols(num_frames) = this->_layers[last_layer].GetOutputNextLayer(num_frames);

  // Process head rechannel
  _head_rechannel.process_(this->_head_inputs, num_frames);
}

Eigen::Block<Eigen::MatrixXf> nam::wavenet::_LayerArray::GetLayerOutputs(const int num_frames)
{
  return this->_layer_outputs.block(0, 0, this->_layer_outputs.rows(), num_frames);
}

Eigen::Block<Eigen::MatrixXf> nam::wavenet::_LayerArray::GetHeadOutputs(const int num_frames)
{
  return this->_head_rechannel.GetOutput(num_frames);
}


void nam::wavenet::_LayerArray::set_weights_(std::vector<float>::iterator& weights)
{
  this->_rechannel.set_weights_(weights);
  for (size_t i = 0; i < this->_layers.size(); i++)
    this->_layers[i].set_weights_(weights);
  this->_head_rechannel.set_weights_(weights);
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
    res += (this->_layers[i].get_kernel_size() - 1) * this->_layers[i].get_dilation();
  return res;
}


// Head =======================================================================

nam::wavenet::_Head::_Head(const int input_size, const int num_layers, const int channels, const std::string activation)
: _channels(channels)
, _head(num_layers > 0 ? channels : input_size, 1, true)
, _activation(activations::Activation::get_activation(activation))
{
  assert(num_layers > 0);
  int dx = input_size;
  for (int i = 0; i < num_layers; i++)
  {
    this->_layers.push_back(Conv1x1(dx, i == num_layers - 1 ? 1 : channels, true));
    dx = channels;
    if (i < num_layers - 1)
      this->_buffers.push_back(Eigen::MatrixXf());
  }
}

void nam::wavenet::_Head::Reset(const double sampleRate, const int maxBufferSize)
{
  set_num_frames_((long)maxBufferSize);
}

void nam::wavenet::_Head::set_weights_(std::vector<float>::iterator& weights)
{
  for (size_t i = 0; i < this->_layers.size(); i++)
    this->_layers[i].set_weights_(weights);
}

void nam::wavenet::_Head::process_(Eigen::MatrixXf& inputs, Eigen::MatrixXf& outputs)
{
  const size_t num_layers = this->_layers.size();
  this->_apply_activation_(inputs);
  if (num_layers == 1)
    outputs = this->_layers[0].process(inputs);
  else
  {
    this->_buffers[0] = this->_layers[0].process(inputs);
    for (size_t i = 1; i < num_layers; i++)
    { // Asserted > 0 layers
      this->_apply_activation_(this->_buffers[i - 1]);
      if (i < num_layers - 1)
        this->_buffers[i] = this->_layers[i].process(this->_buffers[i - 1]);
      else
        outputs = this->_layers[i].process(this->_buffers[i - 1]);
    }
  }
}

void nam::wavenet::_Head::set_num_frames_(const long num_frames)
{
  for (size_t i = 0; i < this->_buffers.size(); i++)
  {
    if (this->_buffers[i].rows() == this->_channels && this->_buffers[i].cols() == num_frames)
      continue; // Already has correct size
    this->_buffers[i].resize(this->_channels, num_frames);
    this->_buffers[i].setZero(); // Shouldn't be needed--these are written to before they're used.
  }
}

void nam::wavenet::_Head::_apply_activation_(Eigen::MatrixXf& x)
{
  this->_activation->apply(x);
}

// WaveNet ====================================================================

nam::wavenet::WaveNet::WaveNet(const std::vector<nam::wavenet::LayerArrayParams>& layer_array_params,
                               const float head_scale, const bool with_head, std::vector<float> weights,
                               const double expected_sample_rate)
: DSP(expected_sample_rate)
, _head_scale(head_scale)
{
  if (with_head)
    throw std::runtime_error("Head not implemented!");
  for (size_t i = 0; i < layer_array_params.size(); i++)
  {
    this->_layer_arrays.push_back(nam::wavenet::_LayerArray(
      layer_array_params[i].input_size, layer_array_params[i].condition_size, layer_array_params[i].head_size,
      layer_array_params[i].channels, layer_array_params[i].kernel_size, layer_array_params[i].dilations,
      layer_array_params[i].activation, layer_array_params[i].gated, layer_array_params[i].head_bias));
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
  this->_head_output.resize(1, 0); // Mono output!
  this->set_weights_(weights);

  mPrewarmSamples = 1;
  for (size_t i = 0; i < this->_layer_arrays.size(); i++)
    mPrewarmSamples += this->_layer_arrays[i].get_receptive_field();
}

void nam::wavenet::WaveNet::set_weights_(std::vector<float>& weights)
{
  std::vector<float>::iterator it = weights.begin();
  for (size_t i = 0; i < this->_layer_arrays.size(); i++)
    this->_layer_arrays[i].set_weights_(it);
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

void nam::wavenet::WaveNet::SetMaxBufferSize(const int maxBufferSize)
{
  DSP::SetMaxBufferSize(maxBufferSize);

  this->_condition.resize(this->_get_condition_dim(), maxBufferSize);
  for (size_t i = 0; i < this->_head_arrays.size(); i++)
    this->_head_arrays[i].resize(this->_head_arrays[i].rows(), maxBufferSize);
  this->_head_output.resize(this->_head_output.rows(), maxBufferSize);
  this->_head_output.setZero();

  // SetMaxBufferSize on layer arrays will propagate to Conv1D::Reset() via _Layer::SetMaxBufferSize()
  for (size_t i = 0; i < this->_layer_arrays.size(); i++)
    this->_layer_arrays[i].SetMaxBufferSize(maxBufferSize);
  // this->_head.SetMaxBufferSize(maxBufferSize);
}

void nam::wavenet::WaveNet::_advance_buffers_(const int num_frames)
{
  // No-op: Conv1D layers manage their own buffers now
}


void nam::wavenet::WaveNet::_set_condition_array(NAM_SAMPLE* input, const int num_frames)
{
  for (int j = 0; j < num_frames; j++)
  {
    this->_condition(0, j) = input[j];
  }
}

void nam::wavenet::WaveNet::process(NAM_SAMPLE* input, NAM_SAMPLE* output, const int num_frames)
{
  assert(num_frames <= mMaxBufferSize);
  this->_set_condition_array(input, num_frames);

  // Main layer arrays:
  // Layer-to-layer
  // Sum on head output
  this->_head_arrays[0].setZero();
  for (size_t i = 0; i < this->_layer_arrays.size(); i++)
  {
    // Get input for this layer array
    const Eigen::MatrixXf& layer_input =
      (i == 0) ? this->_condition : this->_layer_arrays[i - 1].GetLayerOutputs(num_frames);
    if (i == 0)
    {
      // First layer array - no head input
      this->_layer_arrays[i].Process(layer_input, this->_condition, num_frames);
    }
    else
    {
      // Subsequent layer arrays - use head output from previous layer array
      this->_layer_arrays[i].Process(
        layer_input, this->_condition, this->_layer_arrays[i - 1].GetHeadOutputs(num_frames), num_frames);
    }
  }

  // head not implemented

  auto final_head_outputs = this->_layer_arrays.back().GetHeadOutputs(num_frames);
  assert(final_head_outputs.rows() == 1);
  for (int s = 0; s < num_frames; s++)
  {
    const float out = this->_head_scale * final_head_outputs(0, s);
    output[s] = out;
  }

  // Finalize to prepare for the next call:
  this->_advance_buffers_(num_frames);
}

// Factory to instantiate from nlohmann json
std::unique_ptr<nam::DSP> nam::wavenet::Factory(const nlohmann::json& config, std::vector<float>& weights,
                                                const double expectedSampleRate)
{
  std::vector<nam::wavenet::LayerArrayParams> layer_array_params;
  for (size_t i = 0; i < config["layers"].size(); i++)
  {
    nlohmann::json layer_config = config["layers"][i];
    layer_array_params.push_back(nam::wavenet::LayerArrayParams(
      layer_config["input_size"], layer_config["condition_size"], layer_config["head_size"], layer_config["channels"],
      layer_config["kernel_size"], layer_config["dilations"], layer_config["activation"], layer_config["gated"],
      layer_config["head_bias"]));
  }
  const bool with_head = !config["head"].is_null();
  const float head_scale = config["head_scale"];
  return std::make_unique<nam::wavenet::WaveNet>(
    layer_array_params, head_scale, with_head, weights, expectedSampleRate);
}

// Register the factory
namespace
{
static nam::factory::Helper _register_WaveNet("WaveNet", nam::wavenet::Factory);
}
