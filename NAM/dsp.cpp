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
    this->Process(sample_ptr, sample_ptr, 1);
    this->Finalize(1);
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
  this->SetReceptiveField(receptiveField);
}

void nam::Buffer::SetReceptiveField(const int newReceptiveField)
{
  this->SetReceptiveField(newReceptiveField, _INPUT_BUFFER_SAFETY_FACTOR * newReceptiveField);
};

void nam::Buffer::SetReceptiveField(const int newReceptiveField, const int inputBufferSize)
{
  this->mReceptiveField = newReceptiveField;
  this->mInputBuffer.resize(inputBufferSize);
  std::fill(this->mInputBuffer.begin(), this->mInputBuffer.end(), 0.0f);
  this->ResetInputBuffer();
}

void nam::Buffer::UpdateBuffers(float* input, const int numFrames)
{
  // Make sure that the buffer is big enough for the receptive field and the
  // frames needed!
  {
    const long minimum_input_buffer_size = (long)this->mReceptiveField + _INPUT_BUFFER_SAFETY_FACTOR * numFrames;
    if ((long)this->mInputBuffer.size() < minimum_input_buffer_size)
    {
      long new_buffer_size = 2;
      while (new_buffer_size < minimum_input_buffer_size)
        new_buffer_size *= 2;
      this->mInputBuffer.resize(new_buffer_size);
      std::fill(this->mInputBuffer.begin(), this->mInputBuffer.end(), 0.0f);
    }
  }

  // If we'd run off the end of the input buffer, then we need to move the data
  // back to the start of the buffer and start again.
  if (this->mInputBufferOffset + numFrames > (long)this->mInputBuffer.size())
    this->RewindBuffers();
  // Put the new samples into the input buffer
  for (long i = this->mInputBufferOffset, j = 0; j < numFrames; i++, j++)
    this->mInputBuffer[i] = input[j];
  // And resize the output buffer:
  this->mOutputBuffer.resize(numFrames);
  std::fill(this->mOutputBuffer.begin(), this->mOutputBuffer.end(), 0.0f);
}

void nam::Buffer::RewindBuffers()
{
  // Copy the input buffer back
  // RF-1 samples because we've got at least one new one inbound.
  for (long i = 0, j = this->mInputBufferOffset - this->mReceptiveField; i < this->mReceptiveField; i++, j++)
    this->mInputBuffer[i] = this->mInputBuffer[j];
  // And reset the offset.
  // Even though we could be stingy about that one sample that we won't be using
  // (because a new set is incoming) it's probably not worth the
  // hyper-optimization and liable for bugs. And the code looks way tidier this
  // way.
  this->mInputBufferOffset = this->mReceptiveField;
}

void nam::Buffer::ResetInputBuffer()
{
  this->mInputBufferOffset = this->mReceptiveField;
}

void nam::Buffer::Finalize(const int numFrames)
{
  this->nam::DSP::Finalize(numFrames);
  this->mInputBufferOffset += numFrames;
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

  this->mWeight.resize(this->mReceptiveField);
  // Pass in in reverse order so that dot products work out of the box.
  for (int i = 0; i < this->mReceptiveField; i++)
    this->mWeight(i) = weights[receptiveField - 1 - i];
  this->mBias = bias ? weights[receptiveField] : (float)0.0;
}

void nam::Linear::Process(float* input, float* output, const int numFrames)
{
  this->nam::Buffer::UpdateBuffers(input, numFrames);

  // Main computation!
  for (auto i = 0; i < numFrames; i++)
  {
    const size_t offset = this->mInputBufferOffset - this->mWeight.size() + i + 1;
    auto input = Eigen::Map<const Eigen::VectorXf>(&this->mInputBuffer[offset], this->mReceptiveField);
    output[i] = this->mBias + this->mWeight.dot(input);
  }
}

// NN modules =================================================================

void nam::Conv1D::SetWeights(weights_it& weights)
{
  if (this->mWeight.size() > 0)
  {
    const long outChannels = this->mWeight[0].rows();
    const long inChannels = this->mWeight[0].cols();
    // Crazy ordering because that's how it gets flattened.
    for (auto i = 0; i < outChannels; i++)
      for (auto j = 0; j < inChannels; j++)
        for (size_t k = 0; k < this->mWeight.size(); k++)
          this->mWeight[k](i, j) = *(weights++);
  }
  for (long i = 0; i < this->mBias.size(); i++)
    this->mBias(i) = *(weights++);
}

void nam::Conv1D::SetSize(const int inChannels, const int outChannels, const int kernel_size, const bool doBias,
                            const int dilation)
{
  this->mWeight.resize(kernel_size);
  for (size_t i = 0; i < this->mWeight.size(); i++)
    this->mWeight[i].resize(outChannels,
                            inChannels); // y = Ax, input array (C,L)
  if (doBias)
    this->mBias.resize(outChannels);
  else
    this->mBias.resize(0);
  this->mDilation = dilation;
}

void nam::Conv1D::SetSizeAndWeights(const int inChannels, const int outChannels, const int kernel_size,
                                        const int dilation, const bool doBias, weights_it& weights)
{
  this->SetSize(inChannels, outChannels, kernel_size, doBias, dilation);
  this->SetWeights(weights);
}

void nam::Conv1D::Process(const Eigen::Ref<const Eigen::MatrixXf> input, Eigen::Ref<Eigen::MatrixXf> output, const long i_start, const long ncols,
                           const long j_start) const
{
  // This is the clever part ;)
  for (size_t k = 0; k < this->mWeight.size(); k++)
  {
    const long offset = this->mDilation * (k + 1 - this->mWeight.size());
    if (k == 0)
      output.middleCols(j_start, ncols) = this->mWeight[k] * input.middleCols(i_start + offset, ncols);
    else
      output.middleCols(j_start, ncols) += this->mWeight[k] * input.middleCols(i_start + offset, ncols);
  }
  if (this->mBias.size() > 0)
    output.middleCols(j_start, ncols).colwise() += this->mBias;
}

long nam::Conv1D::get_num_weights() const
{
  long num_weights = this->mBias.size();
  for (size_t i = 0; i < this->mWeight.size(); i++)
    num_weights += this->mWeight[i].size();
  return num_weights;
}

nam::Conv1x1::Conv1x1(const int inChannels, const int outChannels, const bool bias)
{
  this->mWeight.resize(outChannels, inChannels);
  this->mDoBias = bias;
  if (bias)
    this->mBias.resize(outChannels);
}

void nam::Conv1x1::SetWeights(weights_it& weights)
{
  for (int i = 0; i < this->mWeight.rows(); i++)
    for (int j = 0; j < this->mWeight.cols(); j++)
      this->mWeight(i, j) = *(weights++);
  if (this->mDoBias)
    for (int i = 0; i < this->mBias.size(); i++)
      this->mBias(i) = *(weights++);
}

Eigen::MatrixXf nam::Conv1x1::Process(const Eigen::Ref<const Eigen::MatrixXf> input) const
{
  if (this->mDoBias)
    return (this->mWeight * input).colwise() + this->mBias;
  else
    return this->mWeight * input;
}
