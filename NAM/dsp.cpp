#include <algorithm> // std::max_element
#include <cmath> // pow, tanh, expf
#include <filesystem>
#include <fstream>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "dsp.h"
#include "registry.h"

#define tanh_impl_ std::tanh
// #define tanh_impl_ fast_tanh_

constexpr const long _INPUT_BUFFER_SAFETY_FACTOR = 32;

nam::DSP::DSP(const double expected_sample_rate)
: mExpectedSampleRate(expected_sample_rate)
{
}

void nam::DSP::prewarm()
{
  if (mMaxBufferSize == 0)
  {
    SetMaxBufferSize(4096);
  }
  const int prewarmSamples = PrewarmSamples();
  if (prewarmSamples == 0)
    return;

  const size_t bufferSize = std::max(mMaxBufferSize, 1);
  std::vector<NAM_SAMPLE> inputBuffer, outputBuffer;
  inputBuffer.resize(bufferSize);
  outputBuffer.resize(bufferSize);
  for (auto it = inputBuffer.begin(); it != inputBuffer.end(); ++it)
  {
    (*it) = (NAM_SAMPLE)0.0;
  }

  NAM_SAMPLE* inputPtr = inputBuffer.data();
  NAM_SAMPLE* outputPtr = outputBuffer.data();
  int samplesProcessed = 0;
  while (samplesProcessed < prewarmSamples)
  {
    this->process(inputPtr, outputPtr, bufferSize);
    samplesProcessed += bufferSize;
  }
}

void nam::DSP::process(NAM_SAMPLE* input, NAM_SAMPLE* output, const int num_frames)
{
  // Default implementation is the null operation
  for (int i = 0; i < num_frames; i++)
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

void nam::DSP::Reset(const double sampleRate, const int maxBufferSize)
{
  // Some subclasses might want to throw an exception if the sample rate is "wrong".
  // This could be under a debugging flag potentially.
  mExternalSampleRate = sampleRate;
  mHaveExternalSampleRate = true;
  SetMaxBufferSize(maxBufferSize);

  prewarm();
}

void nam::DSP::SetLoudness(const double loudness)
{
  mLoudness = loudness;
  mHasLoudness = true;
}

void nam::DSP::SetMaxBufferSize(const int maxBufferSize)
{
  mMaxBufferSize = maxBufferSize;
}

// Buffer =====================================================================

nam::Buffer::Buffer(const int receptive_field, const double expected_sample_rate)
: nam::DSP(expected_sample_rate)
{
  this->_set_receptive_field(receptive_field);
}

void nam::Buffer::_set_receptive_field(const int new_receptive_field)
{
  this->_set_receptive_field(new_receptive_field, _INPUT_BUFFER_SAFETY_FACTOR * new_receptive_field);
};

void nam::Buffer::_set_receptive_field(const int new_receptive_field, const int input_buffer_size)
{
  this->_receptive_field = new_receptive_field;
  this->_input_buffer.resize(input_buffer_size);
  std::fill(this->_input_buffer.begin(), this->_input_buffer.end(), 0.0f);
  this->_reset_input_buffer();
}

void nam::Buffer::_update_buffers_(NAM_SAMPLE* input, const int num_frames)
{
  // Make sure that the buffer is big enough for the receptive field and the
  // frames needed!
  {
    const long minimum_input_buffer_size = (long)this->_receptive_field + _INPUT_BUFFER_SAFETY_FACTOR * num_frames;
    if ((long)this->_input_buffer.size() < minimum_input_buffer_size)
    {
      long new_buffer_size = 2;
      while (new_buffer_size < minimum_input_buffer_size)
        new_buffer_size *= 2;
      this->_input_buffer.resize(new_buffer_size);
      std::fill(this->_input_buffer.begin(), this->_input_buffer.end(), 0.0f);
    }
  }

  // If we'd run off the end of the input buffer, then we need to move the data
  // back to the start of the buffer and start again.
  if (this->_input_buffer_offset + num_frames > (long)this->_input_buffer.size())
    this->_rewind_buffers_();
  // Put the new samples into the input buffer
  for (long i = this->_input_buffer_offset, j = 0; j < num_frames; i++, j++)
    this->_input_buffer[i] = input[j];
  // And resize the output buffer:
  this->_output_buffer.resize(num_frames);
  std::fill(this->_output_buffer.begin(), this->_output_buffer.end(), 0.0f);
}

void nam::Buffer::_rewind_buffers_()
{
  // Copy the input buffer back
  // RF-1 samples because we've got at least one new one inbound.
  for (long i = 0, j = this->_input_buffer_offset - this->_receptive_field; i < this->_receptive_field; i++, j++)
    this->_input_buffer[i] = this->_input_buffer[j];
  // And reset the offset.
  // Even though we could be stingy about that one sample that we won't be using
  // (because a new set is incoming) it's probably not worth the
  // hyper-optimization and liable for bugs. And the code looks way tidier this
  // way.
  this->_input_buffer_offset = this->_receptive_field;
}

void nam::Buffer::_reset_input_buffer()
{
  this->_input_buffer_offset = this->_receptive_field;
}

void nam::Buffer::_advance_input_buffer_(const int num_frames)
{
  this->_input_buffer_offset += num_frames;
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

  this->_weight.resize(this->_receptive_field);
  // Pass in in reverse order so that dot products work out of the box.
  for (int i = 0; i < this->_receptive_field; i++)
    this->_weight(i) = weights[receptive_field - 1 - i];
  this->_bias = _bias ? weights[receptive_field] : (float)0.0;
}

void nam::Linear::process(NAM_SAMPLE* input, NAM_SAMPLE* output, const int num_frames)
{
  this->nam::Buffer::_update_buffers_(input, num_frames);

  // Main computation!
  for (int i = 0; i < num_frames; i++)
  {
    const long offset = this->_input_buffer_offset - this->_weight.size() + i + 1;
    auto input = Eigen::Map<const Eigen::VectorXf>(&this->_input_buffer[offset], this->_receptive_field);
    output[i] = this->_bias + this->_weight.dot(input);
  }

  // Prepare for next call:
  nam::Buffer::_advance_input_buffer_(num_frames);
}

// Factory
std::unique_ptr<nam::DSP> nam::linear::Factory(const nlohmann::json& config, std::vector<float>& weights,
                                               const double expectedSampleRate)
{
  const int receptive_field = config["receptive_field"];
  const bool bias = config["bias"];
  return std::make_unique<nam::Linear>(receptive_field, bias, weights, expectedSampleRate);
}

// NN modules =================================================================

// Conv1x1 ====================================================================

nam::Conv1x1::Conv1x1(const int in_channels, const int out_channels, const bool _bias)
{
  this->_weight.resize(out_channels, in_channels);
  this->_do_bias = _bias;
  if (_bias)
    this->_bias.resize(out_channels);
}


void nam::Conv1x1::SetMaxBufferSize(const int maxBufferSize)
{
  _output.resize(get_out_channels(), maxBufferSize);
}

void nam::Conv1x1::set_weights_(std::vector<float>::iterator& weights)
{
  for (int i = 0; i < this->_weight.rows(); i++)
    for (int j = 0; j < this->_weight.cols(); j++)
      this->_weight(i, j) = *(weights++);
  if (this->_do_bias)
    for (int i = 0; i < this->_bias.size(); i++)
      this->_bias(i) = *(weights++);
}

Eigen::MatrixXf nam::Conv1x1::process(const Eigen::MatrixXf& input, const int num_frames) const
{
  if (this->_do_bias)
    return (this->_weight * input.leftCols(num_frames)).colwise() + this->_bias;
  else
    return this->_weight * input.leftCols(num_frames);
}

void nam::Conv1x1::process_(const Eigen::MatrixXf& input, const int num_frames)
{
  assert(num_frames <= _output.cols());
  _output.leftCols(num_frames).noalias() = this->_weight * input.leftCols(num_frames);
  if (this->_do_bias)
  {
    _output.leftCols(num_frames).colwise() += this->_bias;
  }
}
