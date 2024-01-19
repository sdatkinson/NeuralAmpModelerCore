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
  LSTMCell(const int inputSize, const int hiddenSize, weights_it& weights);
  Eigen::VectorXf GetHiddenState() const { return this->_xh(Eigen::placeholders::lastN(this->GetHiddenSize())); };
  void Process(const Eigen::VectorXf& x);

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

  long GetHiddenSize() const { return this->_b.size() / 4; };
  long GetInputSize() const { return this->_xh.size() - this->GetHiddenSize(); };
};

// The multi-layer LSTM model
class LSTM : public DSP
{
public:
  LSTM(const int numLayers, const int inputSize, const int hiddenSize, const std::vector<float>& weights,
       const double expectedSampleRate = -1.0);
  ~LSTM() = default;

protected:
  Eigen::VectorXf mHeadWeight;
  float mHeadBias;
  void Process(float* input, float* output, const int numFrames) override;
  std::vector<LSTMCell> mLayers;

  float ProcessSample(const float x);

  // Input to the LSTM.
  // Since this is assumed to not be a parametric model, its shape should be (1,)
  Eigen::VectorXf mInput;
};
}; // namespace lstm
}; // namespace nam
