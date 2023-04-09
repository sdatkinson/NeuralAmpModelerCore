#pragma once
// LSTM interface

#include <map>
#include <vector>

#include <Eigen/Dense>

#include "dsp.h"
#include "json.hpp"
#include "wavenet.h"


namespace lstm
{
// A Single LSTM cell
// i input
// f forget
// g cell
// o output
// c cell state
// h hidden state
class LSTMCell
{
public:
  LSTMCell(const int input_size, const int hidden_size, std::vector<float>::iterator& params);
  Eigen::VectorXf get_hidden_state() const { return this->_xh(Eigen::placeholders::lastN(this->_get_hidden_size())); };
  void process_(const Eigen::VectorXf& x);

private:
  // Parameters
  // xh -> ifgo
  // (dx+dh) -> (4*dh)
  Eigen::MatrixXf _w;
  Eigen::VectorXf _b;

  // State
  // Concatenated input and hidden state
  Eigen::VectorXf _xh;
  // Input, Forget, Cell, Output gates
  Eigen::VectorXf _ifgo;

  // Cell state
  Eigen::VectorXf _c;

  long _get_hidden_size() const { return this->_b.size() / 4; };
  long _get_input_size() const { return this->_xh.size() - this->_get_hidden_size(); };
};

// The multi-layer LSTM model
class LSTM : public DSP
{
public:
  // Loudness, pre-conv
  // No, no
  LSTM(const int num_layers, const int input_size, const int hidden_size, std::vector<float>& params,
       nlohmann::json& parametric);
  // No, yes
  LSTM(const int num_layers, const int input_size, const int hidden_size, std::vector<float>& params,
       nlohmann::json& parametric, const bool have_pre_conv, const int pre_conv_in_channels,
       const int pre_conv_out_channels, const int pre_conv_kernel_size);
  // Yes, no
  LSTM(const double loudness, const int num_layers, const int input_size, const int hidden_size,
       std::vector<float>& params, nlohmann::json& parametric);
  // Yes, yes
  LSTM(const double loudness, const int num_layers, const int input_size, const int hidden_size,
       std::vector<float>& params, nlohmann::json& parametric, const bool have_pre_conv, const int pre_conv_in_channels,
       const int pre_conv_out_channels, const int pre_conv_kernel_size);


  void finalize_(const int num_frames) override
  {
    DSP::finalize_(num_frames);
    if (this->_have_pre_conv)
      this->_pre_conv_index += num_frames;
  }

protected:
  long _get_num_params() const { return this->_input_and_params.size() - 1; };
  long _get_receptive_field() const { return this->_pre_conv.get_kernel_size(); };
  // Allocates arrays
  void _prepare_pre_conv_for_frames(const size_t num_frames);

  // Checks if a pre-conv exists, and uses it if so.
  // Output is always available in this->_pre_conv_output
  void _process_pre_conv_();
  void _process_core_() override;
  void _rewind_buffer_();

  bool _have_pre_conv;
  Eigen::MatrixXf _pre_conv_input_buffer;
  long _pre_conv_index;
  Eigen::MatrixXf _pre_conv_output;
  wavenet::DilatedConv _pre_conv;

  std::vector<LSTMCell> _layers;

  Eigen::VectorXf _head_weight;
  float _head_bias;

  // Parameter interpolation
  Eigen::VectorXf _current_buffer_params;
  Eigen::VectorXf _param_interp_from;
  Eigen::VectorXf _param_interp_to;
  const long _param_interp_n = 4096;
  long _param_interp_i;

  // LSTM processing with a single sample.
  // Used when no pre-conv
  float _process_lstm_sample(const float x);
  // LSTM processing on vectors coming out of the pre-conv
  float _process_lstm_vector(const Eigen::VectorXf& x);

  // Initialize the parametric map
  void _init_parametric(nlohmann::json& parametric);

  // Mapping from param name to index in _input_and_params:
  std::map<std::string, int> _parametric_map;
  // Input sample first, params second
  Eigen::VectorXf _input_and_params;
};
}; // namespace lstm
