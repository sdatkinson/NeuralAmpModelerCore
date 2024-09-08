#pragma once
// LSTM implementation

#include <map>
#include <vector>

#include <Eigen/Dense>

#include "dsp.h"

namespace nam
{
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
  LSTMCell(const int input_size, const int hidden_size, std::vector<float>::iterator& weights);
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
  LSTM(const int num_layers, const int input_size, const int hidden_size, std::vector<float>& weights,
       const double expected_sample_rate = -1.0);
  ~LSTM() = default;

protected:
  // Hacky, but a half-second seems to work for most models.
  int PrewarmSamples() override;

  Eigen::VectorXf _head_weight;
  float _head_bias;
  void process(NAM_SAMPLE* input, NAM_SAMPLE* output, const int num_frames) override;
  std::vector<LSTMCell> _layers;

  float _process_sample(const float x);

  // Input to the LSTM.
  // Since this is assumed to not be a parametric model, its shape should be (1,)
  Eigen::VectorXf _input;
};
}; // namespace lstm
}; // namespace nam
