#include <algorithm>
#include <iostream>
#include <math.h>
#include <sstream>

#include <Eigen/Dense>

#include "get_dsp.h"
#include "registry.h"
#include "wavenet.h"

// Layer ======================================================================

void nam::wavenet::_Layer::SetMaxBufferSize(const int maxBufferSize)
{
  _conv.SetMaxBufferSize(maxBufferSize);
  _input_mixin.SetMaxBufferSize(maxBufferSize);
  const long z_channels = this->_conv.get_out_channels(); // This is 2*bottleneck when gated, bottleneck when not
  _z.resize(z_channels, maxBufferSize);
  if (this->_layer1x1)
  {
    this->_layer1x1->SetMaxBufferSize(maxBufferSize);
  }
  // Pre-allocate output buffers
  const long channels = this->get_channels();
  this->_output_next_layer.resize(channels, maxBufferSize);
  // _output_head stores the activated portion: bottleneck rows when no head1x1, or head1x1 out_channels when head1x1 is
  // active
  if (_head1x1)
  {
    this->_output_head.resize(_head1x1->get_out_channels(), maxBufferSize);
    this->_output_head.setZero(); // Ensure consistent initialization across platforms
    _head1x1->SetMaxBufferSize(maxBufferSize);
  }
  else
  {
    this->_output_head.resize(this->_bottleneck, maxBufferSize);
    this->_output_head.setZero(); // Ensure consistent initialization across platforms
  }
  // Set max buffer size for FiLM objects
  if (this->_conv_pre_film)
    this->_conv_pre_film->SetMaxBufferSize(maxBufferSize);
  if (this->_conv_post_film)
    this->_conv_post_film->SetMaxBufferSize(maxBufferSize);
  if (this->_input_mixin_pre_film)
    this->_input_mixin_pre_film->SetMaxBufferSize(maxBufferSize);
  if (this->_input_mixin_post_film)
    this->_input_mixin_post_film->SetMaxBufferSize(maxBufferSize);
  if (this->_activation_pre_film)
    this->_activation_pre_film->SetMaxBufferSize(maxBufferSize);
  if (this->_activation_post_film)
    this->_activation_post_film->SetMaxBufferSize(maxBufferSize);
  if (this->_layer1x1_post_film)
    this->_layer1x1_post_film->SetMaxBufferSize(maxBufferSize);
  if (this->_head1x1_post_film)
    this->_head1x1_post_film->SetMaxBufferSize(maxBufferSize);
}

void nam::wavenet::_Layer::set_weights_(std::vector<float>::iterator& weights)
{
  this->_conv.set_weights_(weights);
  this->_input_mixin.set_weights_(weights);
  if (this->_layer1x1)
  {
    this->_layer1x1->set_weights_(weights);
  }
  if (this->_head1x1)
  {
    this->_head1x1->set_weights_(weights);
  }
  // Set weights for FiLM objects
  if (this->_conv_pre_film)
    this->_conv_pre_film->set_weights_(weights);
  if (this->_conv_post_film)
    this->_conv_post_film->set_weights_(weights);
  if (this->_input_mixin_pre_film)
    this->_input_mixin_pre_film->set_weights_(weights);
  if (this->_input_mixin_post_film)
    this->_input_mixin_post_film->set_weights_(weights);
  if (this->_activation_pre_film)
    this->_activation_pre_film->set_weights_(weights);
  if (this->_activation_post_film)
    this->_activation_post_film->set_weights_(weights);
  if (this->_layer1x1_post_film)
    this->_layer1x1_post_film->set_weights_(weights);
  if (this->_head1x1_post_film)
    this->_head1x1_post_film->set_weights_(weights);
}

void nam::wavenet::_Layer::Process(const Eigen::MatrixXf& input, const Eigen::MatrixXf& condition, const int num_frames)
{
  const long bottleneck = this->_bottleneck; // Use the actual bottleneck value, not the doubled output channels

  // Step 1: input convolutions
  if (this->_conv_pre_film)
  {
    // Use Process() instead of Process_() since input is const
    this->_conv_pre_film->Process(input, condition, num_frames);
    this->_conv.Process(this->_conv_pre_film->GetOutput(), num_frames);
  }
  else
  {
    this->_conv.Process(input, num_frames);
  }
  if (this->_conv_post_film)
  {
    Eigen::MatrixXf& conv_output = this->_conv.GetOutput();
    this->_conv_post_film->Process_(conv_output, condition, num_frames);
  }

  if (this->_input_mixin_pre_film)
  {
    // Use Process() instead of Process_() since condition is const
    this->_input_mixin_pre_film->Process(condition, condition, num_frames);
    this->_input_mixin.process_(this->_input_mixin_pre_film->GetOutput(), num_frames);
  }
  else
  {
    this->_input_mixin.process_(condition, num_frames);
  }
  if (this->_input_mixin_post_film)
  {
    Eigen::MatrixXf& input_mixin_output = this->_input_mixin.GetOutput();
    this->_input_mixin_post_film->Process_(input_mixin_output, condition, num_frames);
  }
  this->_z.leftCols(num_frames).noalias() =
    _conv.GetOutput().leftCols(num_frames) + _input_mixin.GetOutput().leftCols(num_frames);
  if (this->_activation_pre_film)
  {
    this->_activation_pre_film->Process_(this->_z, condition, num_frames);
  }

  // Step 2 & 3: activation and 1x1
  //
  // A note about the gating/blending activations:
  // They take 2x dimension as input.
  // The top channels are for the "primary" activation and will be in-place modified for the final result.
  // The bottom channels are for the "secondary" activation and should not be used post-activation.
  if (this->_gating_mode == GatingMode::NONE)
  {
    this->_activation->apply(this->_z.leftCols(num_frames));
    if (this->_activation_post_film)
    {
      this->_activation_post_film->Process_(this->_z, condition, num_frames);
    }
    if (this->_layer1x1)
    {
      this->_layer1x1->process_(this->_z, num_frames);
    }
  }
  else if (this->_gating_mode == GatingMode::GATED)
  {
    // Use the GatingActivation class
    // Extract the blocks first to avoid temporary reference issues
    auto input_block = this->_z.leftCols(num_frames);
    auto output_block = this->_z.topRows(bottleneck).leftCols(num_frames);
    this->_gating_activation->apply(input_block, output_block);
    if (this->_activation_post_film)
    {
      // Use Process() for blocks and copy result back
      this->_activation_post_film->Process(this->_z.topRows(bottleneck), condition, num_frames);
      this->_z.topRows(bottleneck).leftCols(num_frames).noalias() =
        this->_activation_post_film->GetOutput().leftCols(num_frames);
    }
    if (this->_layer1x1)
    {
      this->_layer1x1->process_(this->_z.topRows(bottleneck), num_frames);
    }
  }
  else if (this->_gating_mode == GatingMode::BLENDED)
  {
    // Use the BlendingActivation class
    // Extract the blocks first to avoid temporary reference issues
    auto input_block = this->_z.leftCols(num_frames);
    auto output_block = this->_z.topRows(bottleneck).leftCols(num_frames);
    this->_blending_activation->apply(input_block, output_block);
    if (this->_activation_post_film)
    {
      // Use Process() for blocks and copy result back
      this->_activation_post_film->Process(this->_z.topRows(bottleneck), condition, num_frames);
      this->_z.topRows(bottleneck).leftCols(num_frames).noalias() =
        this->_activation_post_film->GetOutput().leftCols(num_frames);
    }
    if (this->_layer1x1)
    {
      this->_layer1x1->process_(this->_z.topRows(bottleneck), num_frames);
      if (this->_layer1x1_post_film)
      {
        Eigen::MatrixXf& layer1x1_output = this->_layer1x1->GetOutput();
        this->_layer1x1_post_film->Process_(layer1x1_output, condition, num_frames);
      }
    }
  }

  if (this->_head1x1)
  {
    if (this->_gating_mode == GatingMode::NONE)
    {
      this->_head1x1->process_(this->_z.leftCols(num_frames), num_frames);
    }
    else
    {
      this->_head1x1->process_(this->_z.topRows(bottleneck).leftCols(num_frames), num_frames);
    }
    if (this->_head1x1_post_film)
    {
      Eigen::MatrixXf& head1x1_output = this->_head1x1->GetOutput();
      this->_head1x1_post_film->Process_(head1x1_output, condition, num_frames);
    }
    this->_output_head.leftCols(num_frames).noalias() = this->_head1x1->GetOutput().leftCols(num_frames);
  }
  else // No head 1x1
  {
    // (No FiLM)
    // Store output to head (skip connection: activated conv output)
    if (this->_gating_mode == GatingMode::NONE)
      this->_output_head.leftCols(num_frames).noalias() = this->_z.leftCols(num_frames);
    else
      this->_output_head.leftCols(num_frames).noalias() = this->_z.topRows(bottleneck).leftCols(num_frames);
  }

  // Store output to next layer (residual connection: input + layer1x1 output, or just input if layer1x1 inactive)
  if (this->_layer1x1)
  {
    this->_output_next_layer.leftCols(num_frames).noalias() =
      input.leftCols(num_frames) + this->_layer1x1->GetOutput().leftCols(num_frames);
  }
  else
  {
    // If layer1x1 is inactive, residual connection is just the input (identity)
    this->_output_next_layer.leftCols(num_frames).noalias() = input.leftCols(num_frames);
  }
}

// LayerArray =================================================================

nam::wavenet::_LayerArray::_LayerArray(const LayerArrayParams& params)
: _rechannel(params.input_size, params.channels, false)
, _head_rechannel(params.head1x1_params.active ? params.head1x1_params.out_channels : params.bottleneck,
                  params.head_size, params.head_bias)
, _head_output_size(params.head1x1_params.active ? params.head1x1_params.out_channels : params.bottleneck)
{
  const size_t num_layers = params.dilations.size();
  for (size_t i = 0; i < num_layers; i++)
  {
    LayerParams layer_params(
      params.condition_size, params.channels, params.bottleneck, params.kernel_size, params.dilations[i],
      params.activation_configs[i], params.gating_modes[i], params.groups_input, params.groups_input_mixin,
      params.layer1x1_params, params.head1x1_params, params.secondary_activation_configs[i],
      params.conv_pre_film_params, params.conv_post_film_params, params.input_mixin_pre_film_params,
      params.input_mixin_post_film_params, params.activation_pre_film_params, params.activation_post_film_params,
      params._layer1x1_post_film_params, params.head1x1_post_film_params);
    this->_layers.push_back(_Layer(layer_params));
  }
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
  // _head_inputs size matches actual head output: head1x1.out_channels if active, else bottleneck
  this->_head_inputs.resize(this->_head_output_size, maxBufferSize);
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

nam::wavenet::WaveNet::WaveNet(const int in_channels,
                               const std::vector<nam::wavenet::LayerArrayParams>& layer_array_params,
                               const float head_scale, const bool with_head, std::vector<float> weights,
                               std::unique_ptr<DSP> condition_dsp, const double expected_sample_rate)
: DSP(in_channels,
      layer_array_params.empty() ? throw std::runtime_error("WaveNet requires at least one layer array")
                                 : layer_array_params.back().head_size,
      expected_sample_rate)
, _condition_dsp(std::move(condition_dsp))
, _head_scale(head_scale)
{
  // Assert that if there's a condition DSP, its input is compatible with what it'll get from this WaveNet:
  if (this->_condition_dsp != nullptr)
  {
    if (this->_get_condition_dim() != this->_condition_dsp->NumInputChannels())
    {
      std::stringstream ss;
      ss << "input channels of WaveNet (" << in_channels << ") don't match input channels of condition DSP ("
         << this->_condition_dsp->NumInputChannels() << "!\n";
      throw std::runtime_error(ss.str().c_str());
    }
  }
  if (layer_array_params.empty())
    throw std::runtime_error("WaveNet requires at least one layer array");
  if (with_head)
    throw std::runtime_error("Head not implemented!");
  for (size_t i = 0; i < layer_array_params.size(); i++)
  {
    // Quick assert that the condition_dsp will output compatibly with this layer array
    if (this->_condition_dsp != nullptr)
    {
      if (layer_array_params[i].condition_size != this->_condition_dsp->NumOutputChannels())
      {
        std::stringstream ss;
        ss << "condition_size of layer " << i << " (" << layer_array_params[i].condition_size
           << ") doesn't match output channels of condition DSP (" << this->_condition_dsp->NumOutputChannels()
           << "!\n";
        throw std::runtime_error(ss.str().c_str());
      }
    }
    this->_layer_arrays.push_back(nam::wavenet::_LayerArray(layer_array_params[i]));
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

  // Finally, figure out how much pre-warming is needed for this model.
  mPrewarmSamples = this->_condition_dsp != nullptr ? this->_condition_dsp->PrewarmSamples() : 1;
  for (size_t i = 0; i < this->_layer_arrays.size(); i++)
    mPrewarmSamples += this->_layer_arrays[i].get_receptive_field();
}

void nam::wavenet::WaveNet::set_weights_(std::vector<float>& weights)
{
  std::vector<float>::iterator it = weights.begin();
  // Note: condition_dsp already has its own weights from construction,
  // so we don't need to set its weights here.
  for (size_t i = 0; i < this->_layer_arrays.size(); i++)
    this->_layer_arrays[i].set_weights_(it);
  this->_head_scale = *(it++); // TODO `LayerArray.absorb_head_scale()`
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
  this->_condition_input.resize(this->_get_condition_dim(), maxBufferSize);
  // Resize condition output
  if (this->_condition_dsp == nullptr)
  {
    this->_condition_output.resize(this->_get_condition_dim(), maxBufferSize);
  }
  else
  {
    this->_condition_dsp->SetMaxBufferSize(maxBufferSize);
    const int condition_output_channels = this->_condition_dsp->NumOutputChannels();
    this->_condition_output.resize(condition_output_channels, maxBufferSize);

    // Resize temporary buffers for condition DSP processing
    const int condition_dim = this->_get_condition_dim();
    this->_condition_dsp_input_buffers.resize(condition_dim);
    this->_condition_dsp_output_buffers.resize(condition_output_channels);
    this->_condition_dsp_input_ptrs.resize(condition_dim);
    this->_condition_dsp_output_ptrs.resize(condition_output_channels);

    for (int ch = 0; ch < condition_dim; ch++)
    {
      this->_condition_dsp_input_buffers[ch].resize(maxBufferSize);
      this->_condition_dsp_input_ptrs[ch] = this->_condition_dsp_input_buffers[ch].data();
    }

    for (int ch = 0; ch < condition_output_channels; ch++)
    {
      this->_condition_dsp_output_buffers[ch].resize(maxBufferSize);
      this->_condition_dsp_output_ptrs[ch] = this->_condition_dsp_output_buffers[ch].data();
    }
  }

  for (size_t i = 0; i < this->_layer_arrays.size(); i++)
    this->_layer_arrays[i].SetMaxBufferSize(maxBufferSize);
}

void nam::wavenet::WaveNet::_process_condition(const int num_frames)
{
  if (this->_condition_dsp == nullptr)
  {
    this->_condition_output.leftCols(num_frames) = this->_condition_input.leftCols(num_frames);
  }
  else
  {
    // Copy input data from Eigen matrix to pre-allocated contiguous buffers
    // Since Eigen matrices are column-major, rows are not contiguous
    // TODO maybe use row-major here?
    const int condition_dim = this->_get_condition_dim();
    for (int ch = 0; ch < condition_dim; ch++)
    {
      for (int j = 0; j < num_frames; j++)
        this->_condition_dsp_input_buffers[ch][j] = (NAM_SAMPLE)this->_condition_input(ch, j);
    }

    // Process through condition DSP using pre-allocated buffers
    this->_condition_dsp->process(
      this->_condition_dsp_input_ptrs.data(), this->_condition_dsp_output_ptrs.data(), num_frames);

    // Copy output data back to Eigen matrix
    const int condition_output_channels = this->_condition_dsp->NumOutputChannels();
    for (int ch = 0; ch < condition_output_channels; ch++)
    {
      for (int j = 0; j < num_frames; j++)
        this->_condition_output(ch, j) = (float)this->_condition_dsp_output_buffers[ch][j];
    }
  }
}

void nam::wavenet::WaveNet::_set_condition_array(NAM_SAMPLE** input, const int num_frames)
{
  const int in_channels = NumInputChannels();
  // Fill condition array with input channels
  for (int ch = 0; ch < in_channels; ch++)
  {
    for (int j = 0; j < num_frames; j++)
    {
      this->_condition_input(ch, j) = input[ch][j];
    }
  }
}

void nam::wavenet::WaveNet::process(NAM_SAMPLE** input, NAM_SAMPLE** output, const int num_frames)
{
  assert(num_frames <= mMaxBufferSize);
  const int out_channels = NumOutputChannels();

  this->_set_condition_array(input, num_frames);
  this->_process_condition(num_frames);

  // Main layer arrays:
  // Layer-to-layer
  for (size_t i = 0; i < this->_layer_arrays.size(); i++)
  {
    if (i == 0)
    {
      // First layer array - no head input
      // layer_inputs should be the original input (before condition_dsp processing),
      // condition should be the processed condition output (after condition_dsp)
      this->_layer_arrays[i].Process(this->_condition_input, this->_condition_output, num_frames);
    }
    else
    {
      // Subsequent layer arrays - use outputs from previous layer array.
      // Pass full buffers and slice inside the callee to avoid passing Blocks
      // across API boundaries (which can cause Eigen to allocate temporaries).
      Eigen::MatrixXf& prev_layer_outputs = this->_layer_arrays[i - 1].GetLayerOutputs();
      Eigen::MatrixXf& prev_head_outputs = this->_layer_arrays[i - 1].GetHeadOutputs();
      this->_layer_arrays[i].Process(prev_layer_outputs, this->_condition_output, prev_head_outputs, num_frames);
    }
  }

  // (Head not implemented)

  auto& final_head_outputs = this->_layer_arrays.back().GetHeadOutputs();
  assert(final_head_outputs.rows() == out_channels);

  for (int ch = 0; ch < out_channels; ch++)
  {
    for (int s = 0; s < num_frames; s++)
    {
      const float out = this->_head_scale * final_head_outputs(ch, s);
      output[ch][s] = out;
    }
  }
}

// Factory to instantiate from nlohmann json
std::unique_ptr<nam::DSP> nam::wavenet::Factory(const nlohmann::json& config, std::vector<float>& weights,
                                                const double expectedSampleRate)
{
  std::unique_ptr<nam::DSP> condition_dsp = nullptr;
  if (config.find("condition_dsp") != config.end() && !config["condition_dsp"].is_null())
  {
    const nlohmann::json& condition_dsp_json = config["condition_dsp"];
    condition_dsp = nam::get_dsp(condition_dsp_json);
    if (condition_dsp->GetExpectedSampleRate() != expectedSampleRate)
    {
      std::stringstream ss;
      ss << "Condition DSP expected sample rate (" << condition_dsp->GetExpectedSampleRate()
         << ") doesn't match WaveNet expected sample rate (" << expectedSampleRate << "!\n";
      throw std::runtime_error(ss.str().c_str());
    }
  }
  std::vector<nam::wavenet::LayerArrayParams> layer_array_params;
  for (size_t i = 0; i < config["layers"].size(); i++)
  {
    nlohmann::json layer_config = config["layers"][i];

    const int groups = layer_config.value("groups_input", 1); // defaults to 1
    const int groups_input_mixin = layer_config.value("groups_input_mixin", 1); // defaults to 1

    const int channels = layer_config["channels"];
    const int bottleneck = layer_config.value("bottleneck", channels); // defaults to channels if not present

    // Parse layer1x1 parameters
    bool layer1x1_active = true; // default to active if not present
    int layer1x1_groups = 1;
    if (layer_config.find("layer1x1") != layer_config.end())
    {
      const auto& layer1x1_config = layer_config["layer1x1"];
      layer1x1_active = layer1x1_config["active"]; // default to active
      layer1x1_groups = layer1x1_config["groups"];
    }
    nam::wavenet::Layer1x1Params layer1x1_params(layer1x1_active, layer1x1_groups);

    const int input_size = layer_config["input_size"];
    const int condition_size = layer_config["condition_size"];
    const int head_size = layer_config["head_size"];
    const int kernel_size = layer_config["kernel_size"];
    const auto dilations = layer_config["dilations"];
    const size_t num_layers = dilations.size();

    // Parse activation config(s) - support both single config and array
    std::vector<activations::ActivationConfig> activation_configs;
    if (layer_config["activation"].is_array())
    {
      // Array of activation configs
      for (const auto& activation_json : layer_config["activation"])
      {
        activation_configs.push_back(activations::ActivationConfig::from_json(activation_json));
      }
      if (activation_configs.size() != num_layers)
      {
        throw std::runtime_error("Layer array " + std::to_string(i) + ": activation array size ("
                                 + std::to_string(activation_configs.size()) + ") must match dilations size ("
                                 + std::to_string(num_layers) + ")");
      }
    }
    else
    {
      // Single activation config - duplicate it for all layers
      const activations::ActivationConfig activation_config =
        activations::ActivationConfig::from_json(layer_config["activation"]);
      activation_configs.resize(num_layers, activation_config);
    }
    // Parse gating mode(s) - support both single value and array, and old "gated" boolean
    std::vector<GatingMode> gating_modes;
    std::vector<activations::ActivationConfig> secondary_activation_configs;

    auto parse_gating_mode_str = [](const std::string& gating_mode_str) -> GatingMode {
      if (gating_mode_str == "gated")
        return GatingMode::GATED;
      else if (gating_mode_str == "blended")
        return GatingMode::BLENDED;
      else if (gating_mode_str == "none")
        return GatingMode::NONE;
      else
        throw std::runtime_error("Invalid gating_mode: " + gating_mode_str);
    };

    if (layer_config.find("gating_mode") != layer_config.end())
    {
      if (layer_config["gating_mode"].is_array())
      {
        // Array of gating modes
        for (const auto& gating_mode_json : layer_config["gating_mode"])
        {
          std::string gating_mode_str = gating_mode_json.get<std::string>();
          GatingMode mode = parse_gating_mode_str(gating_mode_str);
          gating_modes.push_back(mode);

          // Parse corresponding secondary activation if gating is enabled
          if (mode != GatingMode::NONE)
          {
            if (layer_config.find("secondary_activation") != layer_config.end())
            {
              if (layer_config["secondary_activation"].is_array())
              {
                // Array of secondary activations - use corresponding index
                if (gating_modes.size() > layer_config["secondary_activation"].size())
                {
                  throw std::runtime_error("Layer array " + std::to_string(i)
                                           + ": secondary_activation array size must be at least "
                                           + std::to_string(gating_modes.size()));
                }
                secondary_activation_configs.push_back(activations::ActivationConfig::from_json(
                  layer_config["secondary_activation"][gating_modes.size() - 1]));
              }
              else
              {
                // Single secondary activation - use for all gated layers
                secondary_activation_configs.push_back(
                  activations::ActivationConfig::from_json(layer_config["secondary_activation"]));
              }
            }
            else
            {
              // Default to Sigmoid for backward compatibility
              secondary_activation_configs.push_back(
                activations::ActivationConfig::simple(activations::ActivationType::Sigmoid));
            }
          }
          else
          {
            // NONE mode - use empty config
            secondary_activation_configs.push_back(activations::ActivationConfig{});
          }
        }
        if (gating_modes.size() != num_layers)
        {
          throw std::runtime_error("Layer array " + std::to_string(i) + ": gating_mode array size ("
                                   + std::to_string(gating_modes.size()) + ") must match dilations size ("
                                   + std::to_string(num_layers) + ")");
        }
        // Validate secondary_activation array size if it's an array
        if (layer_config.find("secondary_activation") != layer_config.end()
            && layer_config["secondary_activation"].is_array())
        {
          if (layer_config["secondary_activation"].size() != num_layers)
          {
            throw std::runtime_error("Layer array " + std::to_string(i) + ": secondary_activation array size ("
                                     + std::to_string(layer_config["secondary_activation"].size())
                                     + ") must match dilations size (" + std::to_string(num_layers) + ")");
          }
        }
      }
      else
      {
        // Single gating mode - duplicate for all layers
        std::string gating_mode_str = layer_config["gating_mode"].get<std::string>();
        GatingMode gating_mode = parse_gating_mode_str(gating_mode_str);
        gating_modes.resize(num_layers, gating_mode);

        // Parse secondary activation
        activations::ActivationConfig secondary_activation_config;
        if (gating_mode != GatingMode::NONE)
        {
          if (layer_config.find("secondary_activation") != layer_config.end())
          {
            secondary_activation_config =
              activations::ActivationConfig::from_json(layer_config["secondary_activation"]);
          }
          else
          {
            // Default to Sigmoid for backward compatibility
            secondary_activation_config = activations::ActivationConfig::simple(activations::ActivationType::Sigmoid);
          }
        }
        secondary_activation_configs.resize(num_layers, secondary_activation_config);
      }
    }
    else if (layer_config.find("gated") != layer_config.end())
    {
      // Backward compatibility: convert old "gated" boolean to new enum
      bool gated = layer_config["gated"];
      GatingMode gating_mode = gated ? GatingMode::GATED : GatingMode::NONE;
      gating_modes.resize(num_layers, gating_mode);

      if (gated)
      {
        activations::ActivationConfig secondary_config =
          activations::ActivationConfig::simple(activations::ActivationType::Sigmoid);
        secondary_activation_configs.resize(num_layers, secondary_config);
      }
      else
      {
        secondary_activation_configs.resize(num_layers, activations::ActivationConfig{});
      }
    }
    else
    {
      // Default to NONE for all layers
      gating_modes.resize(num_layers, GatingMode::NONE);
      secondary_activation_configs.resize(num_layers, activations::ActivationConfig{});
    }

    const bool head_bias = layer_config["head_bias"];

    // Parse head1x1 parameters
    bool head1x1_active = false;
    int head1x1_out_channels = channels;
    int head1x1_groups = 1;
    if (layer_config.find("head1x1") != layer_config.end())
    {
      const auto& head1x1_config = layer_config["head1x1"];
      head1x1_active = head1x1_config["active"];
      head1x1_out_channels = head1x1_config["out_channels"];
      head1x1_groups = head1x1_config["groups"];
    }
    nam::wavenet::Head1x1Params head1x1_params(head1x1_active, head1x1_out_channels, head1x1_groups);

    // Helper function to parse FiLM parameters
    auto parse_film_params = [&layer_config](const std::string& key) -> nam::wavenet::_FiLMParams {
      if (layer_config.find(key) == layer_config.end() || layer_config[key] == false)
      {
        return nam::wavenet::_FiLMParams(false, false);
      }
      const nlohmann::json& film_config = layer_config[key];
      bool active = film_config.value("active", true);
      bool shift = film_config.value("shift", true);
      int groups = film_config.value("groups", 1);
      return nam::wavenet::_FiLMParams(active, shift, groups);
    };

    // Parse FiLM parameters
    nam::wavenet::_FiLMParams conv_pre_film_params = parse_film_params("conv_pre_film");
    nam::wavenet::_FiLMParams conv_post_film_params = parse_film_params("conv_post_film");
    nam::wavenet::_FiLMParams input_mixin_pre_film_params = parse_film_params("input_mixin_pre_film");
    nam::wavenet::_FiLMParams input_mixin_post_film_params = parse_film_params("input_mixin_post_film");
    nam::wavenet::_FiLMParams activation_pre_film_params = parse_film_params("activation_pre_film");
    nam::wavenet::_FiLMParams activation_post_film_params = parse_film_params("activation_post_film");
    nam::wavenet::_FiLMParams _layer1x1_post_film_params = parse_film_params("layer1x1_post_film");
    nam::wavenet::_FiLMParams head1x1_post_film_params = parse_film_params("head1x1_post_film");

    // Validation: if layer1x1_post_film is active, layer1x1 must also be active
    if (_layer1x1_post_film_params.active && !layer1x1_active)
    {
      throw std::runtime_error("Layer array " + std::to_string(i)
                               + ": layer1x1_post_film cannot be active when layer1x1.active is false");
    }

    layer_array_params.push_back(nam::wavenet::LayerArrayParams(
      input_size, condition_size, head_size, channels, bottleneck, kernel_size, dilations,
      std::move(activation_configs), std::move(gating_modes), head_bias, groups, groups_input_mixin, layer1x1_params,
      head1x1_params, std::move(secondary_activation_configs), conv_pre_film_params, conv_post_film_params,
      input_mixin_pre_film_params, input_mixin_post_film_params, activation_pre_film_params,
      activation_post_film_params, _layer1x1_post_film_params, head1x1_post_film_params));
  }
  const bool with_head = config.find("head") != config.end() && !config["head"].is_null();
  const float head_scale = config["head_scale"];

  if (layer_array_params.empty())
    throw std::runtime_error("WaveNet config requires at least one layer array");

  // Backward compatibility: assume 1 input channel
  const int in_channels = config.value("in_channels", 1);

  // out_channels is determined from last layer array's head_size
  return std::make_unique<nam::wavenet::WaveNet>(
    in_channels, layer_array_params, head_scale, with_head, weights, std::move(condition_dsp), expectedSampleRate);
}

// Register the factory
namespace
{
static nam::factory::Helper _register_WaveNet("WaveNet", nam::wavenet::Factory);
}
