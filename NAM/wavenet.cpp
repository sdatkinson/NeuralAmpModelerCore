#include <algorithm>
#include <iostream>
#include <math.h>

#include <Eigen/Dense>

#include "registry.h"
#include "wavenet.h"

nam::wavenet::_DilatedConv::_DilatedConv(const int in_channels, const int out_channels, const int kernel_size,
                                         const int bias, const int dilation)
{
  this->set_size_(in_channels, out_channels, kernel_size, bias, dilation);
}

// Layer ======================================================================

void nam::wavenet::_Layer::SetMaxBufferSize(const int maxBufferSize)
{
  // Reset Conv1D with new buffer size
  // Use -1.0 for sampleRate since it's unused
  _conv.Reset(-1.0, maxBufferSize);
  _input_mixin.SetMaxBufferSize(maxBufferSize);
  _z.resize(this->_conv.get_out_channels(), maxBufferSize);
  _1x1.SetMaxBufferSize(maxBufferSize);
}

void nam::wavenet::_Layer::set_weights_(std::vector<float>::iterator& weights)
{
  this->_conv.set_weights_(weights);
  this->_input_mixin.set_weights_(weights);
  this->_1x1.set_weights_(weights);
}

void nam::wavenet::_Layer::process_(const Eigen::MatrixXf& input, const Eigen::MatrixXf& condition,
                                    Eigen::MatrixXf& head_input, Eigen::MatrixXf& output, const long i_start,
                                    const long j_start, const int num_frames)
{
  const long ncols = (long)num_frames; // TODO clean this up
  const long channels = this->get_channels();
  
  // Extract input slice and process with Conv1D
  Eigen::MatrixXf input_slice = input.middleCols(i_start, num_frames);
  this->_conv.Process(input_slice, num_frames);
  
  // Get output from Conv1D
  auto conv_output = this->_conv.GetOutput(num_frames);
  
  // Still need _z buffer for intermediate processing (mixing condition, gating, activation)
  // Resize _z if needed
  if (this->_z.rows() != conv_output.rows() || this->_z.cols() < num_frames)
  {
    this->_z.resize(conv_output.rows(), num_frames);
  }
  this->_z.leftCols(num_frames) = conv_output;
  
  // Mix-in condition
  _input_mixin.process_(condition, num_frames);
  this->_z.leftCols(num_frames).noalias() += _input_mixin.GetOutput(num_frames);

  if (!this->_gated)
  {
    this->_activation->apply(this->_z.leftCols(num_frames));
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
  }

  head_input.leftCols(num_frames).noalias() += this->_z.block(0, 0, channels, num_frames);
  if (!_gated)
  {
    _1x1.process_(_z, num_frames);
  }
  else
  {
    // Probably not RT-safe yet
    _1x1.process_(_z.topRows(channels), num_frames);
  }
  output.middleCols(j_start, ncols).noalias() = input.middleCols(i_start, ncols) + _1x1.GetOutput(num_frames);
}

void nam::wavenet::_Layer::set_num_frames_(const long num_frames)
{
  // TODO deprecate for SetMaxBufferSize()
  if (this->_z.rows() == this->_conv.get_out_channels() && this->_z.cols() == num_frames)
    return; // Already has correct size

  this->_z.resize(this->_conv.get_out_channels(), num_frames);
  this->_z.setZero();
}

// LayerArray =================================================================

#define LAYER_ARRAY_BUFFER_SIZE 65536

nam::wavenet::_LayerArray::_LayerArray(const int input_size, const int condition_size, const int head_size,
                                       const int channels, const int kernel_size, const std::vector<int>& dilations,
                                       const std::string activation, const bool gated, const bool head_bias)
: _rechannel(input_size, channels, false)
, _head_rechannel(channels, head_size, head_bias)
{
  for (size_t i = 0; i < dilations.size(); i++)
    this->_layers.push_back(_Layer(condition_size, channels, kernel_size, dilations[i], activation, gated));
  const long receptive_field = this->_get_receptive_field();
  for (size_t i = 0; i < dilations.size(); i++)
  {
    this->_layer_buffers.push_back(Eigen::MatrixXf(channels, LAYER_ARRAY_BUFFER_SIZE + receptive_field - 1));
    this->_layer_buffers[i].setZero();
  }
  this->_buffer_start = this->_get_receptive_field() - 1;
}

void nam::wavenet::_LayerArray::SetMaxBufferSize(const int maxBufferSize)
{
  _rechannel.SetMaxBufferSize(maxBufferSize);
  _head_rechannel.SetMaxBufferSize(maxBufferSize);
  for (auto it = _layers.begin(); it != _layers.end(); ++it)
  {
    it->SetMaxBufferSize(maxBufferSize);
  }
}

void nam::wavenet::_LayerArray::advance_buffers_(const int num_frames)
{
  this->_buffer_start += num_frames;
}

long nam::wavenet::_LayerArray::get_receptive_field() const
{
  long result = 0;
  for (size_t i = 0; i < this->_layers.size(); i++)
    result += this->_layers[i].get_dilation() * (this->_layers[i].get_kernel_size() - 1);
  return result;
}

void nam::wavenet::_LayerArray::prepare_for_frames_(const long num_frames)
{
  // Example:
  // _buffer_start = 0
  // num_frames = 64
  // buffer_size = 64
  // -> this will write on indices 0 through 63, inclusive.
  // -> No illegal writes.
  // -> no rewind needed.
  if (this->_buffer_start + num_frames > this->_get_buffer_size())
    this->_rewind_buffers_();
}

void nam::wavenet::_LayerArray::process_(const Eigen::MatrixXf& layer_inputs, const Eigen::MatrixXf& condition,
                                         Eigen::MatrixXf& head_inputs, Eigen::MatrixXf& layer_outputs,
                                         Eigen::MatrixXf& head_outputs, const int num_frames)
{
  this->_rechannel.process_(layer_inputs, num_frames);
  // Still need _layer_buffers[0] to store rechannel output for compatibility
  // TODO: can simplify once Conv1D fully manages its own buffers
  this->_layer_buffers[0].middleCols(this->_buffer_start, num_frames) = _rechannel.GetOutput(num_frames);
  
  const size_t last_layer = this->_layers.size() - 1;
  for (size_t i = 0; i < this->_layers.size(); i++)
  {
    // For subsequent layers, we could use previous Conv1D's output directly
    // But for now, we still use _layer_buffers for compatibility with process_() interface
    if (i > 0)
    {
      // Get output from previous Conv1D and use it as input
      // But process_() still expects _layer_buffers, so we copy to it
      auto prev_output = this->_layers[i - 1].get_conv().GetOutput(num_frames);
      this->_layer_buffers[i].middleCols(this->_buffer_start, num_frames) = prev_output;
    }
    
    this->_layers[i].process_(this->_layer_buffers[i], condition, head_inputs,
                              i == last_layer ? layer_outputs : this->_layer_buffers[i + 1], this->_buffer_start,
                              i == last_layer ? 0 : this->_buffer_start, num_frames);
  }
  _head_rechannel.process_(head_inputs, num_frames);
  head_outputs.leftCols(num_frames) = _head_rechannel.GetOutput(num_frames);
}

void nam::wavenet::_LayerArray::set_num_frames_(const long num_frames)
{
  // Wavenet checks for unchanged num_frames; if we made it here, there's
  // something to do.
  if (LAYER_ARRAY_BUFFER_SIZE - num_frames < this->_get_receptive_field())
  {
    std::stringstream ss;
    ss << "Asked to accept a buffer of " << num_frames << " samples, but the buffer is too short ("
       << LAYER_ARRAY_BUFFER_SIZE << ") to get out of the recptive field (" << this->_get_receptive_field()
       << "); copy errors could occur!\n";
    throw std::runtime_error(ss.str().c_str());
  }
  for (size_t i = 0; i < this->_layers.size(); i++)
    this->_layers[i].set_num_frames_(num_frames);
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

void nam::wavenet::_LayerArray::_rewind_buffers_()
// Consider wrapping instead...
// Can make this smaller--largest dilation, not receptive field!
{
  const long start = this->_get_receptive_field() - 1;
  for (size_t i = 0; i < this->_layer_buffers.size(); i++)
  {
    const long d = (this->_layers[i].get_kernel_size() - 1) * this->_layers[i].get_dilation();
    this->_layer_buffers[i].middleCols(start - d, d) = this->_layer_buffers[i].middleCols(this->_buffer_start - d, d);
  }
  this->_buffer_start = start;
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
  for (size_t i = 0; i < this->_layer_array_outputs.size(); i++)
    this->_layer_array_outputs[i].resize(this->_layer_array_outputs[i].rows(), maxBufferSize);
  this->_head_output.resize(this->_head_output.rows(), maxBufferSize);
  this->_head_output.setZero();

  // SetMaxBufferSize on layer arrays will propagate to Conv1D::Reset() via _Layer::SetMaxBufferSize()
  for (size_t i = 0; i < this->_layer_arrays.size(); i++)
    this->_layer_arrays[i].SetMaxBufferSize(maxBufferSize);
  // this->_head.SetMaxBufferSize(maxBufferSize);
}

void nam::wavenet::WaveNet::_advance_buffers_(const int num_frames)
{
  for (size_t i = 0; i < this->_layer_arrays.size(); i++)
    this->_layer_arrays[i].advance_buffers_(num_frames);
}

void nam::wavenet::WaveNet::_prepare_for_frames_(const long num_frames)
{
  for (size_t i = 0; i < this->_layer_arrays.size(); i++)
    this->_layer_arrays[i].prepare_for_frames_(num_frames);
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
  this->_prepare_for_frames_(num_frames);
  this->_set_condition_array(input, num_frames);

  // Main layer arrays:
  // Layer-to-layer
  // Sum on head output
  this->_head_arrays[0].setZero();
  for (size_t i = 0; i < this->_layer_arrays.size(); i++)
    this->_layer_arrays[i].process_(i == 0 ? this->_condition : this->_layer_array_outputs[i - 1], this->_condition,
                                    this->_head_arrays[i], this->_layer_array_outputs[i], this->_head_arrays[i + 1],
                                    num_frames);
  // this->_head.process_(
  //   this->_head_input,
  //   this->_head_output
  //);
  //  Copy to required output array
  //  Hack: apply head scale here; revisit when/if I activate the head.
  //  assert(this->_head_output.rows() == 1);

  const long final_head_array = this->_head_arrays.size() - 1;
  assert(this->_head_arrays[final_head_array].rows() == 1);
  for (int s = 0; s < num_frames; s++)
  {
    const float out = this->_head_scale * this->_head_arrays[final_head_array](0, s);
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
