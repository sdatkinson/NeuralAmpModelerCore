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
  const long z_channels = this->_conv.get_out_channels(); // This is 2*bottleneck when gated, bottleneck when not
  _z.resize(z_channels, maxBufferSize);
  _1x1.SetMaxBufferSize(maxBufferSize);
  // Pre-allocate output buffers
  const long channels = this->get_channels();
  this->_output_next_layer.resize(channels, maxBufferSize);
  // _output_head stores the activated portion: bottleneck rows (the actual bottleneck value, not doubled)
  this->_output_head.resize(this->_bottleneck, maxBufferSize);
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
  const long bottleneck = this->_bottleneck; // Use the actual bottleneck value, not the doubled output channels

  // Step 1: input convolutions
  this->_conv.Process(input, num_frames);
  this->_input_mixin.process_(condition, num_frames);
  this->_z.leftCols(num_frames).noalias() =
    this->_conv.GetOutput().leftCols(num_frames) + _input_mixin.GetOutput().leftCols(num_frames);

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
      this->_activation->apply(this->_z.block(0, i, bottleneck, 1));
      // TODO Need to support other activation functions here instead of hardcoded sigmoid
      activations::Activation::get_activation("Sigmoid")->apply(this->_z.block(bottleneck, i, bottleneck, 1));
    }
    this->_z.block(0, 0, bottleneck, num_frames).array() *=
      this->_z.block(bottleneck, 0, bottleneck, num_frames).array();
    _1x1.process_(_z.topRows(bottleneck), num_frames); // Might not be RT safe
  }

  // Store output to head (skip connection: activated conv output)
  if (!this->_gated)
    this->_output_head.leftCols(num_frames).noalias() = this->_z.leftCols(num_frames);
  else
    this->_output_head.leftCols(num_frames).noalias() = this->_z.topRows(bottleneck).leftCols(num_frames);
  // Store output to next layer (residual connection: input + _1x1 output)
  this->_output_next_layer.leftCols(num_frames).noalias() =
    input.leftCols(num_frames) + _1x1.GetOutput().leftCols(num_frames);
}


// LayerArray =================================================================

nam::wavenet::_LayerArray::_LayerArray(const int input_size, const int condition_size, const int head_size,
                                       const int channels, const int bottleneck, const int kernel_size,
                                       const std::vector<int>& dilations, const std::string activation,
                                       const bool gated, const bool head_bias, const int groups_input,
                                       const int groups_1x1)
: _rechannel(input_size, channels, false)
, _head_rechannel(bottleneck, head_size, head_bias)
, _bottleneck(bottleneck)
{
  for (size_t i = 0; i < dilations.size(); i++)
    this->_layers.push_back(_Layer(
      condition_size, channels, bottleneck, kernel_size, dilations[i], activation, gated, groups_input, groups_1x1));
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
  this->_head_inputs.resize(this->_bottleneck, maxBufferSize);
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
  this->_head_inputs.leftCols(num_frames).noalias() = head_inputs.leftCols(num_frames);
  ProcessInner(layer_inputs, condition, num_frames);
}

void nam::wavenet::_LayerArray::ProcessInner(const Eigen::MatrixXf& layer_inputs, const Eigen::MatrixXf& condition,
                                             const int num_frames)
{
  // Process rechannel and get output
  this->_rechannel.process_(layer_inputs, num_frames);
  Eigen::MatrixXf& rechannel_output = _rechannel.GetOutput();

  // Process layers
  for (size_t i = 0; i < this->_layers.size(); i++)
  {
    // Process first layer with rechannel output, subsequent layers with previous layer output
    // Use separate branches to avoid ternary operator creating temporaries
    if (i == 0)
    {
      // First layer consumes the rechannel output buffer
      this->_layers[i].Process(rechannel_output, condition, num_frames);
    }
    else
    {
      // Subsequent layers consume the full output buffer of the previous layer
      Eigen::MatrixXf& prev_output = this->_layers[i - 1].GetOutputNextLayer();
      this->_layers[i].Process(prev_output, condition, num_frames);
    }

    // Accumulate head output from this layer
    this->_head_inputs.leftCols(num_frames).noalias() += this->_layers[i].GetOutputHead().leftCols(num_frames);
  }

  // Store output from last layer
  const size_t last_layer = this->_layers.size() - 1;
  this->_layer_outputs.leftCols(num_frames).noalias() =
    this->_layers[last_layer].GetOutputNextLayer().leftCols(num_frames);

  // Process head rechannel
  _head_rechannel.process_(this->_head_inputs, num_frames);
}


Eigen::MatrixXf& nam::wavenet::_LayerArray::GetHeadOutputs()
{
  return this->_head_rechannel.GetOutput();
}

const Eigen::MatrixXf& nam::wavenet::_LayerArray::GetHeadOutputs() const
{
  return this->_head_rechannel.GetOutput();
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
      layer_array_params[i].channels, layer_array_params[i].bottleneck, layer_array_params[i].kernel_size,
      layer_array_params[i].dilations, layer_array_params[i].activation, layer_array_params[i].gated,
      layer_array_params[i].head_bias, layer_array_params[i].groups_input, layer_array_params[i].groups_1x1));
    if (i > 0)
      if (layer_array_params[i].channels != layer_array_params[i - 1].head_size)
      {
        std::stringstream ss;
        ss << "channels of layer " << i << " (" << layer_array_params[i].channels
           << ") doesn't match head_size of preceding layer (" << layer_array_params[i - 1].head_size << "!\n";
        throw std::runtime_error(ss.str().c_str());
      }
  }
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
  for (size_t i = 0; i < this->_layer_arrays.size(); i++)
    this->_layer_arrays[i].SetMaxBufferSize(maxBufferSize);
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
  for (size_t i = 0; i < this->_layer_arrays.size(); i++)
  {
    if (i == 0)
    {
      // First layer array - no head input
      this->_layer_arrays[i].Process(this->_condition, this->_condition, num_frames);
    }
    else
    {
      // Subsequent layer arrays - use outputs from previous layer array.
      // Pass full buffers and slice inside the callee to avoid passing Blocks
      // across API boundaries (which can cause Eigen to allocate temporaries).
      Eigen::MatrixXf& prev_layer_outputs = this->_layer_arrays[i - 1].GetLayerOutputs();
      Eigen::MatrixXf& prev_head_outputs = this->_layer_arrays[i - 1].GetHeadOutputs();
      this->_layer_arrays[i].Process(prev_layer_outputs, this->_condition, prev_head_outputs, num_frames);
    }
  }

  // (Head not implemented)

  auto& final_head_outputs = this->_layer_arrays.back().GetHeadOutputs();
  assert(final_head_outputs.rows() == 1);
  for (int s = 0; s < num_frames; s++)
  {
    const float out = this->_head_scale * final_head_outputs(0, s);
    output[s] = out;
  }
}

// Factory to instantiate from nlohmann json
std::unique_ptr<nam::DSP> nam::wavenet::Factory(const nlohmann::json& config, std::vector<float>& weights,
                                                const double expectedSampleRate)
{
  std::vector<nam::wavenet::LayerArrayParams> layer_array_params;
  for (size_t i = 0; i < config["layers"].size(); i++)
  {
    nlohmann::json layer_config = config["layers"][i];
    const int groups = layer_config.value("groups", 1); // defaults to 1
    const int groups_1x1 = layer_config.value("groups_1x1", 1); // defaults to 1
    const int channels = layer_config["channels"];
    const int bottleneck = layer_config.value("bottleneck", channels); // defaults to channels if not present
    layer_array_params.push_back(nam::wavenet::LayerArrayParams(
      layer_config["input_size"], layer_config["condition_size"], layer_config["head_size"], channels, bottleneck,
      layer_config["kernel_size"], layer_config["dilations"], layer_config["activation"], layer_config["gated"],
      layer_config["head_bias"], groups, groups_1x1));
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
