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

nam::DSP::DSP(const double expected_sample_rate)
: mExpectedSampleRate(expected_sample_rate)
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

nam::Buffer::Buffer(const int receptive_field, const double expected_sample_rate)
: nam::DSP(expected_sample_rate)
{
  this->SetReceptiveField(receptive_field);
}

void nam::Buffer::SetReceptiveField(const int newReceptiveField)
{
  this->SetReceptiveField(newReceptiveField, _INPUT_BUFFER_SAFETY_FACTOR * newReceptiveField);
};

void nam::Buffer::SetReceptiveField(const int newReceptiveField, const int input_buffer_size)
{
  this->mReceptiveField = newReceptiveField;
  this->mInputBuffer.resize(input_buffer_size);
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

nam::Linear::Linear(const int receptive_field, const bool _bias, const std::vector<float>& weights,
                    const double expected_sample_rate)
: nam::Buffer(receptive_field, expected_sample_rate)
{
  if ((int)weights.size() != (receptive_field + (_bias ? 1 : 0)))
    throw std::runtime_error(
      "Params vector does not match expected size based "
      "on architecture parameters");

  this->_weight.resize(this->mReceptiveField);
  // Pass in in reverse order so that dot products work out of the box.
  for (int i = 0; i < this->mReceptiveField; i++)
    this->_weight(i) = weights[receptive_field - 1 - i];
  this->_bias = _bias ? weights[receptive_field] : (float)0.0;
}

void nam::Linear::Process(float* input, float* output, const int numFrames)
{
  this->nam::Buffer::UpdateBuffers(input, numFrames);

  // Main computation!
  for (auto i = 0; i < numFrames; i++)
  {
    const size_t offset = this->mInputBufferOffset - this->_weight.size() + i + 1;
    auto input = Eigen::Map<const Eigen::VectorXf>(&this->mInputBuffer[offset], this->mReceptiveField);
    output[i] = this->_bias + this->_weight.dot(input);
  }
}

// NN modules =================================================================

void nam::Conv1D::set_weights_(weights_it& weights)
{
  if (this->_weight.size() > 0)
  {
    const long out_channels = this->_weight[0].rows();
    const long in_channels = this->_weight[0].cols();
    // Crazy ordering because that's how it gets flattened.
    for (auto i = 0; i < out_channels; i++)
      for (auto j = 0; j < in_channels; j++)
        for (size_t k = 0; k < this->_weight.size(); k++)
          this->_weight[k](i, j) = *(weights++);
  }
  for (long i = 0; i < this->_bias.size(); i++)
    this->_bias(i) = *(weights++);
}

void nam::Conv1D::set_size_(const int in_channels, const int out_channels, const int kernel_size, const bool do_bias,
                            const int _dilation)
{
  this->_weight.resize(kernel_size);
  for (size_t i = 0; i < this->_weight.size(); i++)
    this->_weight[i].resize(out_channels,
                            in_channels); // y = Ax, input array (C,L)
  if (do_bias)
    this->_bias.resize(out_channels);
  else
    this->_bias.resize(0);
  this->_dilation = _dilation;
}

void nam::Conv1D::set_size_and_weights_(const int in_channels, const int out_channels, const int kernel_size,
                                        const int _dilation, const bool do_bias, weights_it& weights)
{
  this->set_size_(in_channels, out_channels, kernel_size, do_bias, _dilation);
  this->set_weights_(weights);
}

void nam::Conv1D::process_(const Eigen::Ref<const Eigen::MatrixXf> input, Eigen::Ref<Eigen::MatrixXf> output, const long i_start, const long ncols,
                           const long j_start) const
{
  // This is the clever part ;)
  for (size_t k = 0; k < this->_weight.size(); k++)
  {
    const long offset = this->_dilation * (k + 1 - this->_weight.size());
    if (k == 0)
      output.middleCols(j_start, ncols) = this->_weight[k] * input.middleCols(i_start + offset, ncols);
    else
      output.middleCols(j_start, ncols) += this->_weight[k] * input.middleCols(i_start + offset, ncols);
  }
  if (this->_bias.size() > 0)
    output.middleCols(j_start, ncols).colwise() += this->_bias;
}

long nam::Conv1D::get_num_weights() const
{
  long num_weights = this->_bias.size();
  for (size_t i = 0; i < this->_weight.size(); i++)
    num_weights += this->_weight[i].size();
  return num_weights;
}

nam::Conv1x1::Conv1x1(const int in_channels, const int out_channels, const bool _bias)
{
  this->_weight.resize(out_channels, in_channels);
  this->_do_bias = _bias;
  if (_bias)
    this->_bias.resize(out_channels);
}

void nam::Conv1x1::set_weights_(weights_it& weights)
{
  for (int i = 0; i < this->_weight.rows(); i++)
    for (int j = 0; j < this->_weight.cols(); j++)
      this->_weight(i, j) = *(weights++);
  if (this->_do_bias)
    for (int i = 0; i < this->_bias.size(); i++)
      this->_bias(i) = *(weights++);
}

Eigen::MatrixXf nam::Conv1x1::Process(const Eigen::Ref<const Eigen::MatrixXf> input) const
{
  if (this->_do_bias)
    return (this->_weight * input).colwise() + this->_bias;
  else
    return this->_weight * input;
}
