// LSTM implementation

#include <algorithm>
#include <string>
#include <vector>

#include "lstm.h"

// Used by pre-conv
const size_t BUFFER_SIZE = 65536;

lstm::LSTMCell::LSTMCell(const int input_size, const int hidden_size, std::vector<float>::iterator& params)
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
      this->_w(i, j) = *(params++);
  for (int i = 0; i < this->_b.size(); i++)
    this->_b[i] = *(params++);
  const int h_offset = input_size;
  for (int i = 0; i < hidden_size; i++)
    this->_xh[i + h_offset] = *(params++);
  for (int i = 0; i < hidden_size; i++)
    this->_c[i] = *(params++);
}

void lstm::LSTMCell::process_(const Eigen::VectorXf& x)
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
  for (auto i = 0; i < hidden_size; i++)
    this->_c[i] = activations::sigmoid(this->_ifgo[i + f_offset]) * this->_c[i]
                  + activations::sigmoid(this->_ifgo[i + i_offset]) * tanhf(this->_ifgo[i + g_offset]);
  const long h_offset = input_size;
  for (int i = 0; i < hidden_size; i++)
    this->_xh[i + h_offset] = activations::sigmoid(this->_ifgo[i + o_offset]) * tanhf(this->_c[i]);
}

lstm::LSTM::LSTM(const int num_layers, const int input_size, const int hidden_size, std::vector<float>& params,
                 nlohmann::json& parametric)
: lstm::LSTM::LSTM(TARGET_DSP_LOUDNESS, num_layers, input_size, hidden_size, params, parametric, false, 0, 0, 0)
{
}

lstm::LSTM::LSTM(const int num_layers, const int input_size, const int hidden_size, std::vector<float>& params,
                 nlohmann::json& parametric, const bool have_pre_conv, const int pre_conv_in_channels,
                 const int pre_conv_out_channels, const int pre_conv_kernel_size)
: LSTM(TARGET_DSP_LOUDNESS, num_layers, input_size, hidden_size, params, parametric, have_pre_conv,
       pre_conv_in_channels, pre_conv_out_channels, pre_conv_kernel_size)
{
}

lstm::LSTM::LSTM(const double loudness, const int num_layers, const int input_size, const int hidden_size,
                 std::vector<float>& params, nlohmann::json& parametric)
: LSTM(loudness, num_layers, input_size, hidden_size, params, parametric, false, 0, 0, 0)
{
}

lstm::LSTM::LSTM(const double loudness, const int num_layers, const int input_size, const int hidden_size,
                 std::vector<float>& params, nlohmann::json& parametric, const bool have_pre_conv,
                 const int pre_conv_in_channels, const int pre_conv_out_channels, const int pre_conv_kernel_size)
: DSP(loudness)
, _pre_conv(pre_conv_in_channels, pre_conv_out_channels, pre_conv_kernel_size, true, 1)
, _have_pre_conv(have_pre_conv)
{
  this->_init_parametric(parametric);
  std::vector<float>::iterator it = params.begin();
  if (this->_have_pre_conv)
    this->_pre_conv.set_params_(it);
  this->_pre_conv_index = this->_pre_conv.get_kernel_size() - 1;
  for (int i = 0; i < num_layers; i++)
    this->_layers.push_back(LSTMCell(i == 0 ? input_size : hidden_size, hidden_size, it));
  this->_head_weight.resize(hidden_size);
  for (int i = 0; i < hidden_size; i++)
    this->_head_weight[i] = *(it++);
  this->_head_bias = *(it++);
  assert(it == params.end());
}

void lstm::LSTM::_init_parametric(nlohmann::json& parametric)
{
  std::vector<std::string> parametric_names;
  for (nlohmann::json::iterator it = parametric.begin(); it != parametric.end(); ++it)
  {
    parametric_names.push_back(it.key());
  }
  std::sort(parametric_names.begin(), parametric_names.end());
  {
    int i = 1;
    for (std::vector<std::string>::iterator it = parametric_names.begin(); it != parametric_names.end(); ++it, i++)
      this->_parametric_map[*it] = i;
  }

  this->_input_and_params.resize(1 + parametric.size()); // TODO amp parameters
  if (this->_have_pre_conv)
  {
    this->_pre_conv_input_buffer.resize(this->_input_and_params.size(), BUFFER_SIZE);
    this->_pre_conv_input_buffer.fill(0.0f);
  }
}

void lstm::LSTM::_prepare_pre_conv_for_frames(const size_t num_frames)
{
  if (this->_pre_conv_index + num_frames >= this->_pre_conv_input_buffer.cols())
    this->_rewind_buffer_();
  this->_pre_conv_output.resize(this->_pre_conv.get_out_channels(), num_frames);
}

void lstm::LSTM::_process_core_()
{
  // Get params into the input vector before starting
  const size_t num_frames = this->_input_post_gain.size();
  if (this->_stale_params)
  {
    for (std::unordered_map<std::string, double>::iterator it = this->_params.begin(); it != this->_params.end(); ++it)
      this->_input_and_params[this->_parametric_map[it->first]] = it->second;
    this->_stale_params = false;
  }

  if (!this->_have_pre_conv)
    for (int i = 0; i < num_frames; i++)
      this->_core_dsp_output[i] = this->_process_lstm_sample(this->_input_post_gain[i]);
  else
  {
    this->_process_pre_conv_();
    for (long i = 0; i < num_frames; i++)
      this->_core_dsp_output[i] = this->_process_lstm_vector(this->_pre_conv_output.col(i));
  }
}

void lstm::LSTM::_process_pre_conv_()
{
  const size_t num_frames = this->_input_post_gain.size();
  const size_t num_params = this->_get_num_params();
  // Check for rewind
  this->_prepare_pre_conv_for_frames(num_frames);
  // Copy signal
  for (long i = 0; i < num_frames; i++)
    this->_pre_conv_input_buffer(0, i + this->_pre_conv_index) = this->_input_post_gain[i];
  // Copy params
  this->_pre_conv_input_buffer.block(1, this->_pre_conv_index, num_params, num_frames).colwise() =
    this->_input_and_params.bottomRows(num_params);
  // And do the conv
  this->_pre_conv.process_(this->_pre_conv_input_buffer, this->_pre_conv_output, this->_pre_conv_index, num_frames, 0);
}

float lstm::LSTM::_process_lstm_sample(const float x)
{
  if (this->_layers.size() == 0)
    return x;
  this->_input_and_params(0) = x;
  return this->_process_lstm_vector(this->_input_and_params);
}

float lstm::LSTM::_process_lstm_vector(const Eigen::VectorXf& x)
{
  this->_layers[0].process_(x);
  for (int i = 1; i < this->_layers.size(); i++)
    this->_layers[i].process_(this->_layers[i - 1].get_hidden_state());
  return this->_head_weight.dot(this->_layers[this->_layers.size() - 1].get_hidden_state()) + this->_head_bias;
}

void lstm::LSTM::_rewind_buffer_()
{
  const long n = this->_pre_conv.get_kernel_size() - 1;
  this->_pre_conv_input_buffer.leftCols(n) = this->_pre_conv_input_buffer.middleCols(this->_pre_conv_index - n, n);
  this->_pre_conv_index = n;
}
