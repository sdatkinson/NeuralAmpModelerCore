#include <algorithm> // std::max_element
#include <algorithm>
#include <cmath> // pow, tanh, expf
#include <filesystem>
#include <fstream>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "dsp.h"

#define tanh_impl_ std::tanh
// #define tanh_impl_ fast_tanh_

constexpr const long _INPUT_BUFFER_SAFETY_FACTOR = 32;

nam::DSP::DSP(const double expectedSampleRate)
: mExpectedSampleRate(expectedSampleRate)
{
}

void nam::DSP::Prewarm()
{
  if (mPrewarmSamples == 0)
    return;

  float sample = 0;
  float* sample_ptr = &sample;

  // pre-warm the model for a model-specific number of samples
  for (long i = 0; i < mPrewarmSamples; i++)
  {
    Process(sample_ptr, sample_ptr, 1);
    Finalize(1);
    sample = 0;
  }
}

void nam::DSP::Process(float* input, float* output, const int numFrames)
{
  // Default implementation is the null operation
  for (auto i = 0; i < numFrames; i++)
    output[i] = input[i];
}

double nam::DSP::GetLoudness() const
{
  if (!HasLoudness())
  {
    throw std::runtime_error("Asked for loudness of a model that doesn't know how loud it is!");
  }
  return mLoudness;
}

void nam::DSP::SetLoudness(const double loudness)
{
  mLoudness = loudness;
  mHasLoudness = true;
}

void nam::DSP::Finalize(const int numFrames) {}

// Buffer =====================================================================

nam::Buffer::Buffer(const int receptiveField, const double expectedSampleRate)
: nam::DSP(expectedSampleRate)
{
  SetReceptiveField(receptiveField);
}

void nam::Buffer::SetReceptiveField(const int newReceptiveField)
{
  SetReceptiveField(newReceptiveField, _INPUT_BUFFER_SAFETY_FACTOR * newReceptiveField);
};

void nam::Buffer::SetReceptiveField(const int newReceptiveField, const int inputBufferSize)
{
  mReceptiveField = newReceptiveField;
  mInputBuffer.resize(inputBufferSize);
  std::fill(mInputBuffer.begin(), mInputBuffer.end(), 0.0f);
  ResetInputBuffer();
}

void nam::Buffer::UpdateBuffers(float* input, const int numFrames)
{
  // Make sure that the buffer is big enough for the receptive field and the
  // frames needed!
  {
    const long minimum_input_buffer_size = (long)mReceptiveField + _INPUT_BUFFER_SAFETY_FACTOR * numFrames;
    if ((long)mInputBuffer.size() < minimum_input_buffer_size)
    {
      long new_buffer_size = 2;
      while (new_buffer_size < minimum_input_buffer_size)
        new_buffer_size *= 2;
      mInputBuffer.resize(new_buffer_size);
      std::fill(mInputBuffer.begin(), mInputBuffer.end(), 0.0f);
    }
  }

  // If we'd run off the end of the input buffer, then we need to move the data
  // back to the start of the buffer and start again.
  if (mInputBufferOffset + numFrames > (long)mInputBuffer.size())
    RewindBuffers();
  // Put the new samples into the input buffer
  for (long i = mInputBufferOffset, j = 0; j < numFrames; i++, j++)
    mInputBuffer[i] = input[j];
  // And resize the output buffer:
  mOutputBuffer.resize(numFrames);
  std::fill(mOutputBuffer.begin(), mOutputBuffer.end(), 0.0f);
}

void nam::Buffer::RewindBuffers()
{
  // Copy the input buffer back
  // RF-1 samples because we've got at least one new one inbound.
  for (long i = 0, j = mInputBufferOffset - mReceptiveField; i < mReceptiveField; i++, j++)
    mInputBuffer[i] = mInputBuffer[j];
  // And reset the offset.
  // Even though we could be stingy about that one sample that we won't be using
  // (because a new set is incoming) it's probably not worth the
  // hyper-optimization and liable for bugs. And the code looks way tidier this
  // way.
  mInputBufferOffset = mReceptiveField;
}

void nam::Buffer::ResetInputBuffer()
{
  mInputBufferOffset = mReceptiveField;
}

void nam::Buffer::Finalize(const int numFrames)
{
  nam::DSP::Finalize(numFrames);
  mInputBufferOffset += numFrames;
}

// Linear =====================================================================

nam::Linear::Linear(const int receptiveField, const bool bias, const std::vector<float>& weights,
                    const double expectedSampleRate)
: nam::Buffer(receptiveField, expectedSampleRate)
{
  if ((int)weights.size() != (receptiveField + (bias ? 1 : 0)))
    throw std::runtime_error(
      "Params vector does not match expected size based "
      "on architecture parameters");

  mWeight.resize(mReceptiveField);
  // Pass in in reverse order so that dot products work out of the box.
  for (int i = 0; i < mReceptiveField; i++)
    mWeight(i) = weights[receptiveField - 1 - i];
  mBias = bias ? weights[receptiveField] : (float)0.0;
}

void nam::Linear::Process(float* input, float* output, const int numFrames)
{
  nam::Buffer::UpdateBuffers(input, numFrames);

  // Main computation!
  for (auto i = 0; i < numFrames; i++)
  {
    const size_t offset = mInputBufferOffset - mWeight.size() + i + 1;
    auto input = Eigen::Map<const Eigen::VectorXf>(&mInputBuffer[offset], mReceptiveField);
    output[i] = mBias + mWeight.dot(input);
  }
}

// NN modules =================================================================

void nam::Conv1D::SetWeights(weightsIterator& weights)
{
  if (mWeight.size() > 0)
  {
    const long outChannels = mWeight[0].rows();
    const long inChannels = mWeight[0].cols();
    // Crazy ordering because that's how it gets flattened.
    for (auto i = 0; i < outChannels; i++)
      for (auto j = 0; j < inChannels; j++)
        for (size_t k = 0; k < mWeight.size(); k++)
          mWeight[k](i, j) = *(weights++);
  }
  for (long i = 0; i < mBias.size(); i++)
    mBias(i) = *(weights++);
}

void nam::Conv1D::SetSize(const int inChannels, const int outChannels, const int kernelSize, const bool doBias,
                            const int dilation)
{
  mWeight.resize(kernelSize);
  for (size_t i = 0; i < mWeight.size(); i++)
    mWeight[i].resize(outChannels,
                            inChannels); // y = Ax, input array (C,L)
  if (doBias)
    mBias.resize(outChannels);
  else
    mBias.resize(0);
  mDilation = dilation;
}

void nam::Conv1D::SetSizeAndWeights(const int inChannels, const int outChannels, const int kernelSize,
                                        const int dilation, const bool doBias, weightsIterator& weights)
{
  SetSize(inChannels, outChannels, kernelSize, doBias, dilation);
  SetWeights(weights);
}

void nam::Conv1D::Process(const Eigen::Ref<const Eigen::MatrixXf> input, Eigen::Ref<Eigen::MatrixXf> output, const long i_start, const long ncols,
                           const long j_start) const
{
  // This is the clever part ;)
  for (size_t k = 0; k < mWeight.size(); k++)
  {
    const long offset = mDilation * (k + 1 - mWeight.size());
    if (k == 0)
      output.middleCols(j_start, ncols) = mWeight[k] * input.middleCols(i_start + offset, ncols);
    else
      output.middleCols(j_start, ncols) += mWeight[k] * input.middleCols(i_start + offset, ncols);
  }
  if (mBias.size() > 0)
    output.middleCols(j_start, ncols).colwise() += mBias;
}

long nam::Conv1D::GetNumWeights() const
{
  long num_weights = mBias.size();
  for (size_t i = 0; i < mWeight.size(); i++)
    num_weights += mWeight[i].size();
  return num_weights;
}

nam::Conv1x1::Conv1x1(const int inChannels, const int outChannels, const bool bias)
{
  mWeight.resize(outChannels, inChannels);
  mDoBias = bias;
  if (bias)
    mBias.resize(outChannels);
}

void nam::Conv1x1::SetWeights(weightsIterator& weights)
{
  for (int i = 0; i < mWeight.rows(); i++)
    for (int j = 0; j < mWeight.cols(); j++)
      mWeight(i, j) = *(weights++);
  if (mDoBias)
    for (int i = 0; i < mBias.size(); i++)
      mBias(i) = *(weights++);
}

Eigen::MatrixXf nam::Conv1x1::Process(const Eigen::Ref<const Eigen::MatrixXf> input) const
{
  if (mDoBias)
    return (mWeight * input).colwise() + mBias;
  else
    return mWeight * input;
}
