#include <algorithm>
#include <string>
#include <vector>
#include <memory>

#include "registry.h"
#include "lstm.h"

nam::lstm::LSTMCell::LSTMCell(const int input_size, const int hidden_size, std::vector<float>::iterator& weights)
{
  // Resize arrays
  this->_w.resize(4 * hidden_size, input_size + hidden_size);
  this->_b.resize(4 * hidden_size);
  this->_xh.resize(input_size + hidden_size);
  this->_ifgo.resize(4 * hidden_size);
  this->_c.resize(hidden_size);

  // Assign in row-major because that's how PyTorch goes.
  for (int i = 0; i < this->_w.rows(); i++)
    for (int j = 0; j < this->_w.cols(); j++)
      this->_w(i, j) = *(weights++);
  for (int i = 0; i < this->_b.size(); i++)
    this->_b[i] = *(weights++);
  const int h_offset = input_size;
  for (int i = 0; i < hidden_size; i++)
    this->_xh[i + h_offset] = *(weights++);
  for (int i = 0; i < hidden_size; i++)
    this->_c[i] = *(weights++);
}

void nam::lstm::LSTMCell::process_(const Eigen::VectorXf& x)
{
  const long hidden_size = this->_get_hidden_size();
  const long input_size = this->_get_input_size();
  // Assign inputs
  this->_xh(Eigen::seq(0, input_size - 1)) = x;
  // The matmul
  this->_ifgo = this->_w * this->_xh + this->_b;
  // Elementwise updates (apply nonlinearities here)
  const long i_offset = 0;
  const long f_offset = hidden_size;
  const long g_offset = 2 * hidden_size;
  const long o_offset = 3 * hidden_size;
  const long h_offset = input_size;

  if (activations::Activation::using_fast_tanh)
  {
    for (auto i = 0; i < hidden_size; i++)
      this->_c[i] =
        activations::fast_sigmoid(this->_ifgo[i + f_offset]) * this->_c[i]
        + activations::fast_sigmoid(this->_ifgo[i + i_offset]) * activations::fast_tanh(this->_ifgo[i + g_offset]);

    for (int i = 0; i < hidden_size; i++)
      this->_xh[i + h_offset] =
        activations::fast_sigmoid(this->_ifgo[i + o_offset]) * activations::fast_tanh(this->_c[i]);
  }
  else
  {
    for (auto i = 0; i < hidden_size; i++)
      this->_c[i] = activations::sigmoid(this->_ifgo[i + f_offset]) * this->_c[i]
                    + activations::sigmoid(this->_ifgo[i + i_offset]) * tanhf(this->_ifgo[i + g_offset]);

    for (int i = 0; i < hidden_size; i++)
      this->_xh[i + h_offset] = activations::sigmoid(this->_ifgo[i + o_offset]) * tanhf(this->_c[i]);
  }
}

nam::lstm::LSTM::LSTM(const int in_channels, const int out_channels, const int num_layers, const int input_size,
                      const int hidden_size, std::vector<float>& weights, const double expected_sample_rate)
: DSP(in_channels, out_channels, expected_sample_rate)
{
  // Allocate input and output vectors
  this->_input.resize(input_size);
  this->_output.resize(out_channels);

  // Store input_size for first layer
  this->_first_layer_input_size = input_size;

  std::vector<float>::iterator it = weights.begin();
  for (int i = 0; i < num_layers; i++)
    this->_layers.push_back(LSTMCell(i == 0 ? input_size : hidden_size, hidden_size, it));

  // Load head weight as matrix (out_channels x hidden_size)
  // Weights are stored row-major: first row (output 0), then row 1 (output 1), etc.
  this->_head_weight.resize(out_channels, hidden_size);
  for (int out_ch = 0; out_ch < out_channels; out_ch++)
  {
    for (int h = 0; h < hidden_size; h++)
    {
      this->_head_weight(out_ch, h) = *(it++);
    }
  }

  // Load head bias as vector (out_channels)
  this->_head_bias.resize(out_channels);
  for (int out_ch = 0; out_ch < out_channels; out_ch++)
  {
    this->_head_bias(out_ch) = *(it++);
  }

  assert(it == weights.end());
}

void nam::lstm::LSTM::process(NAM_SAMPLE** input, NAM_SAMPLE** output, const int num_frames)
{
  const int in_channels = NumInputChannels();
  const int out_channels = NumOutputChannels();

  for (int i = 0; i < num_frames; i++)
  {
    // Copy multi-channel input to _input vector
    for (int ch = 0; ch < in_channels; ch++)
    {
      this->_input(ch) = input[ch][i];
    }

    // Process sample (stores result in _output)
    this->_process_sample();

    // Copy multi-channel output from _output to output arrays
    for (int ch = 0; ch < out_channels; ch++)
    {
      output[ch][i] = this->_output(ch);
    }
  }
}

int nam::lstm::LSTM::PrewarmSamples()
{
  int result = (int)(0.5 * mExpectedSampleRate);
  // If the expected sample rate wasn't provided, it'll be -1.
  // Make sure something still happens.
  return result <= 0 ? 1 : result;
}

void nam::lstm::LSTM::_process_sample()
{
  const int in_channels = NumInputChannels();
  const int out_channels = NumOutputChannels();

  if (this->_layers.size() == 0)
  {
    // No layers - pass input through to output (using first in_channels of output)
    const int channels_to_copy = std::min(in_channels, out_channels);
    for (int ch = 0; ch < channels_to_copy; ch++)
      this->_output(ch) = this->_input(ch);
    // Zero-fill remaining output channels if in_channels < out_channels
    for (int ch = channels_to_copy; ch < out_channels; ch++)
      this->_output(ch) = 0.0f;
    return;
  }

  this->_layers[0].process_(this->_input);
  for (size_t i = 1; i < this->_layers.size(); i++)
    this->_layers[i].process_(this->_layers[i - 1].get_hidden_state());

  // Compute output using head weight matrix and bias vector
  // _output = _head_weight * hidden_state + _head_bias
  const Eigen::VectorXf& hidden_state = this->_layers[this->_layers.size() - 1].get_hidden_state();

  // Compute matrix-vector product: (out_channels x hidden_size) * (hidden_size) = (out_channels)
  // Store directly in _output (which is already sized correctly in constructor)
  this->_output.noalias() = this->_head_weight * hidden_state;

  // Add bias: (out_channels) += (out_channels)
  this->_output.noalias() += this->_head_bias;
}

// Factory to instantiate from nlohmann json
std::unique_ptr<nam::DSP> nam::lstm::Factory(const nlohmann::json& config, std::vector<float>& weights,
                                             const double expectedSampleRate)
{
  const int num_layers = config["num_layers"];
  const int input_size = config["input_size"];
  const int hidden_size = config["hidden_size"];
  // Default to 1 channel in/out for backward compatibility
  const int in_channels = config.value("in_channels", input_size);
  const int out_channels = config.value("out_channels", 1);
  return std::make_unique<nam::lstm::LSTM>(
    in_channels, out_channels, num_layers, input_size, hidden_size, weights, expectedSampleRate);
}

// Register the factory
namespace
{
static nam::factory::Helper _register_LSTM("LSTM", nam::lstm::Factory);
}
