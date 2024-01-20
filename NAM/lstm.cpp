#include <algorithm>
#include <string>
#include <vector>

#include "dsp.h"
#include "lstm.h"

nam::lstm::LSTMCell::LSTMCell(const int inputSize, const int hiddenSize, weights_it& weights)
{
  // Resize arrays
  _w.resize(4 * hiddenSize, inputSize + hiddenSize);
  _b.resize(4 * hiddenSize);
  _xh.resize(inputSize + hiddenSize);
  _ifgo.resize(4 * hiddenSize);
  _c.resize(hiddenSize);

  // Assign in row-major because that's how PyTorch goes.
  for (int i = 0; i < _w.rows(); i++)
    for (int j = 0; j < _w.cols(); j++)
      _w(i, j) = *(weights++);
  for (int i = 0; i < _b.size(); i++)
    _b[i] = *(weights++);
  const int h_offset = inputSize;
  for (int i = 0; i < hiddenSize; i++)
    _xh[i + h_offset] = *(weights++);
  for (int i = 0; i < hiddenSize; i++)
    _c[i] = *(weights++);
}

void nam::lstm::LSTMCell::Process(const Eigen::VectorXf& x)
{
  const long hiddenSize = GetHiddenSize();
  const long inputSize = GetInputSize();
  // Assign inputs
  _xh(Eigen::seq(0, inputSize - 1)) = x;
  // The matmul
  _ifgo = _w * _xh + _b;
  // Elementwise updates (apply nonlinearities here)
  const long i_offset = 0;
  const long f_offset = hiddenSize;
  const long g_offset = 2 * hiddenSize;
  const long o_offset = 3 * hiddenSize;
  const long h_offset = inputSize;

  if (activations::Activation::sUsingFastTanh)
  {
    for (auto i = 0; i < hiddenSize; i++)
      _c[i] =
        activations::fast_sigmoid(_ifgo[i + f_offset]) * _c[i]
        + activations::fast_sigmoid(_ifgo[i + i_offset]) * activations::fast_tanh(_ifgo[i + g_offset]);

    for (int i = 0; i < hiddenSize; i++)
      _xh[i + h_offset] =
        activations::fast_sigmoid(_ifgo[i + o_offset]) * activations::fast_tanh(_c[i]);
  }
  else
  {
    for (auto i = 0; i < hiddenSize; i++)
      _c[i] = activations::sigmoid(_ifgo[i + f_offset]) * _c[i]
                    + activations::sigmoid(_ifgo[i + i_offset]) * tanhf(_ifgo[i + g_offset]);

    for (int i = 0; i < hiddenSize; i++)
      _xh[i + h_offset] = activations::sigmoid(_ifgo[i + o_offset]) * tanhf(_c[i]);
  }
}

nam::lstm::LSTM::LSTM(const int numLayers, const int inputSize, const int hiddenSize, const std::vector<float>& weights,
                      const double expectedSampleRate)
: DSP(expectedSampleRate)
{
  mInput.resize(1);
  auto it = weights.begin();
  for (int i = 0; i < numLayers; i++)
    mLayers.push_back(LSTMCell(i == 0 ? inputSize : hiddenSize, hiddenSize, it));
  mHeadWeight.resize(hiddenSize);
  for (int i = 0; i < hiddenSize; i++)
    mHeadWeight[i] = *(it++);
  mHeadBias = *(it++);
  assert(it == weights.end());
}

void nam::lstm::LSTM::Process(float* input, float* output, const int numFrames)
{
  for (auto i = 0; i < numFrames; i++)
    output[i] = ProcessSample(input[i]);
}

float nam::lstm::LSTM::ProcessSample(const float x)
{
  if (mLayers.size() == 0)
    return x;
  mInput(0) = x;
  mLayers[0].Process(mInput);
  for (size_t i = 1; i < mLayers.size(); i++)
    mLayers[i].Process(mLayers[i - 1].GetHiddenState());
  return mHeadWeight.dot(mLayers[mLayers.size() - 1].GetHiddenState()) + mHeadBias;
}
