#include <algorithm>
#include <string>
#include <vector>

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

nam::lstm::LSTM::LSTM(const int num_layers, const int input_size, const int hidden_size, std::vector<float>& weights,
                      const double expected_sample_rate)
: DSP(expected_sample_rate)
{
  this->_input.resize(1);
  std::vector<float>::iterator it = weights.begin();
  for (int i = 0; i < num_layers; i++)
    this->_layers.push_back(LSTMCell(i == 0 ? input_size : hidden_size, hidden_size, it));
  this->_head_weight.resize(hidden_size);
  for (int i = 0; i < hidden_size; i++)
    this->_head_weight[i] = *(it++);
  this->_head_bias = *(it++);
  assert(it == weights.end());
}

void nam::lstm::LSTM::process(NAM_SAMPLE* input, NAM_SAMPLE* output, const int num_frames)
{
  for (int i = 0; i < num_frames; i++)
    output[i] = this->_process_sample(input[i]);
}

int nam::lstm::LSTM::PrewarmSamples()
{
  int result = (int)(0.5 * mExpectedSampleRate);
  // If the expected sample rate wasn't provided, it'll be -1.
  // Make sure something still happens.
  return result <= 0 ? 1 : result;
}

float nam::lstm::LSTM::_process_sample(const float x)
{
  if (this->_layers.size() == 0)
    return x;
  this->_input(0) = x;
  this->_layers[0].process_(this->_input);
  for (size_t i = 1; i < this->_layers.size(); i++)
    this->_layers[i].process_(this->_layers[i - 1].get_hidden_state());
  return this->_head_weight.dot(this->_layers[this->_layers.size() - 1].get_hidden_state()) + this->_head_bias;
}
