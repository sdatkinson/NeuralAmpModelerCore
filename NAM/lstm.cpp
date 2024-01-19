#include <algorithm>
#include <string>
#include <vector>

#include "dsp.h"
#include "lstm.h"

nam::lstm::LSTMCell::LSTMCell(const int inputSize, const int hiddenSize, weights_it& weights)
{
  // Resize arrays
  this->_w.resize(4 * hiddenSize, inputSize + hiddenSize);
  this->_b.resize(4 * hiddenSize);
  this->_xh.resize(inputSize + hiddenSize);
  this->_ifgo.resize(4 * hiddenSize);
  this->_c.resize(hiddenSize);

  // Assign in row-major because that's how PyTorch goes.
  for (int i = 0; i < this->_w.rows(); i++)
    for (int j = 0; j < this->_w.cols(); j++)
      this->_w(i, j) = *(weights++);
  for (int i = 0; i < this->_b.size(); i++)
    this->_b[i] = *(weights++);
  const int h_offset = inputSize;
  for (int i = 0; i < hiddenSize; i++)
    this->_xh[i + h_offset] = *(weights++);
  for (int i = 0; i < hiddenSize; i++)
    this->_c[i] = *(weights++);
}

void nam::lstm::LSTMCell::Process(const Eigen::VectorXf& x)
{
  const long hiddenSize = this->_get_hidden_size();
  const long inputSize = this->_get_input_size();
  // Assign inputs
  this->_xh(Eigen::seq(0, inputSize - 1)) = x;
  // The matmul
  this->_ifgo = this->_w * this->_xh + this->_b;
  // Elementwise updates (apply nonlinearities here)
  const long i_offset = 0;
  const long f_offset = hiddenSize;
  const long g_offset = 2 * hiddenSize;
  const long o_offset = 3 * hiddenSize;
  const long h_offset = inputSize;

  if (activations::Activation::sUsingFastTanh)
  {
    for (auto i = 0; i < hiddenSize; i++)
      this->_c[i] =
        activations::fast_sigmoid(this->_ifgo[i + f_offset]) * this->_c[i]
        + activations::fast_sigmoid(this->_ifgo[i + i_offset]) * activations::fast_tanh(this->_ifgo[i + g_offset]);

    for (int i = 0; i < hiddenSize; i++)
      this->_xh[i + h_offset] =
        activations::fast_sigmoid(this->_ifgo[i + o_offset]) * activations::fast_tanh(this->_c[i]);
  }
  else
  {
    for (auto i = 0; i < hiddenSize; i++)
      this->_c[i] = activations::sigmoid(this->_ifgo[i + f_offset]) * this->_c[i]
                    + activations::sigmoid(this->_ifgo[i + i_offset]) * tanhf(this->_ifgo[i + g_offset]);

    for (int i = 0; i < hiddenSize; i++)
      this->_xh[i + h_offset] = activations::sigmoid(this->_ifgo[i + o_offset]) * tanhf(this->_c[i]);
  }
}

nam::lstm::LSTM::LSTM(const int numLayers, const int inputSize, const int hiddenSize, const std::vector<float>& weights,
                      const double expectedSampleRate)
: DSP(expectedSampleRate)
{
  this->mInput.resize(1);
  auto it = weights.begin();
  for (int i = 0; i < numLayers; i++)
    this->mLayers.push_back(LSTMCell(i == 0 ? inputSize : hiddenSize, hiddenSize, it));
  this->mHeadWeight.resize(hiddenSize);
  for (int i = 0; i < hiddenSize; i++)
    this->mHeadWeight[i] = *(it++);
  this->mHeadBias = *(it++);
  assert(it == weights.end());
}

void nam::lstm::LSTM::Process(float* input, float* output, const int numFrames)
{
  for (auto i = 0; i < numFrames; i++)
    output[i] = this->ProcessSample(input[i]);
}

float nam::lstm::LSTM::ProcessSample(const float x)
{
  if (this->mLayers.size() == 0)
    return x;
  this->mInput(0) = x;
  this->mLayers[0].Process(this->mInput);
  for (size_t i = 1; i < this->mLayers.size(); i++)
    this->mLayers[i].Process(this->mLayers[i - 1].get_hidden_state());
  return this->mHeadWeight.dot(this->mLayers[this->mLayers.size() - 1].get_hidden_state()) + this->mHeadBias;
}
