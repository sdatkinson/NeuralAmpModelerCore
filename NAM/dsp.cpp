#include <algorithm> // std::max_element
#include <cmath> // pow, tanh, expf
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "dsp.h"
#include "registry.h"

#define tanh_impl_ std::tanh
// #define tanh_impl_ fast_tanh_

constexpr const long _INPUT_BUFFER_SAFETY_FACTOR = 32;

nam::DSP::DSP(const int in_channels, const int out_channels, const double expected_sample_rate)
: mExpectedSampleRate(expected_sample_rate)
, mInChannels(in_channels)
, mOutChannels(out_channels)
{
  if (in_channels <= 0 || out_channels <= 0)
  {
    throw std::runtime_error("Channel counts must be positive");
  }
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
  // Allocate buffers for all channels
  std::vector<std::vector<NAM_SAMPLE>> inputBuffers(mInChannels);
  std::vector<std::vector<NAM_SAMPLE>> outputBuffers(mOutChannels);
  std::vector<NAM_SAMPLE*> inputPtrs(mInChannels);
  std::vector<NAM_SAMPLE*> outputPtrs(mOutChannels);

  for (int ch = 0; ch < mInChannels; ch++)
  {
    inputBuffers[ch].resize(bufferSize, (NAM_SAMPLE)0.0);
    inputPtrs[ch] = inputBuffers[ch].data();
  }
  for (int ch = 0; ch < mOutChannels; ch++)
  {
    outputBuffers[ch].resize(bufferSize, (NAM_SAMPLE)0.0);
    outputPtrs[ch] = outputBuffers[ch].data();
  }

  int samplesProcessed = 0;
  while (samplesProcessed < prewarmSamples)
  {
    this->process(inputPtrs.data(), outputPtrs.data(), bufferSize);
    samplesProcessed += bufferSize;
  }
}

void nam::DSP::process(NAM_SAMPLE** input, NAM_SAMPLE** output, const int num_frames)
{
  // Default implementation is the null operation: copy input to output
  // For now, assume 1:1 channel mapping (first min(in_channels, out_channels) channels)
  const int channelsToProcess = std::min(mInChannels, mOutChannels);
  for (int ch = 0; ch < channelsToProcess; ch++)
  {
    for (int i = 0; i < num_frames; i++)
      output[ch][i] = input[ch][i];
  }
  // Zero out any extra output channels
  for (int ch = channelsToProcess; ch < mOutChannels; ch++)
  {
    for (int i = 0; i < num_frames; i++)
      output[ch][i] = (NAM_SAMPLE)0.0;
  }
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

double nam::DSP::GetInputLevel()
{
  return mInputLevel.level;
}

double nam::DSP::GetOutputLevel()
{
  return mOutputLevel.level;
}

bool nam::DSP::HasInputLevel()
{
  return mInputLevel.haveLevel;
}

bool nam::DSP::HasOutputLevel()
{
  return mOutputLevel.haveLevel;
}

void nam::DSP::SetInputLevel(const double inputLevel)
{
  mInputLevel.haveLevel = true;
  mInputLevel.level = inputLevel;
}

void nam::DSP::SetOutputLevel(const double outputLevel)
{
  mOutputLevel.haveLevel = true;
  mOutputLevel.level = outputLevel;
}

// Buffer =====================================================================

nam::Buffer::Buffer(const int in_channels, const int out_channels, const int receptive_field,
                    const double expected_sample_rate)
: nam::DSP(in_channels, out_channels, expected_sample_rate)
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
  const int in_channels = NumInputChannels();
  const int out_channels = NumOutputChannels();

  // Resize buffers for all input channels
  _input_buffers.resize(in_channels);
  for (int ch = 0; ch < in_channels; ch++)
  {
    _input_buffers[ch].resize(input_buffer_size);
    std::fill(_input_buffers[ch].begin(), _input_buffers[ch].end(), 0.0f);
  }

  // Resize output buffers (though they'll be resized per call in _update_buffers_)
  _output_buffers.resize(out_channels);

  this->_reset_input_buffer();
}

void nam::Buffer::_update_buffers_(NAM_SAMPLE** input, const int num_frames)
{
  const int in_channels = NumInputChannels();
  const int out_channels = NumOutputChannels();

  // Make sure that the buffers are big enough for the receptive field and the
  // frames needed. All channels use the same buffer size.
  const long minimum_input_buffer_size = (long)this->_receptive_field + _INPUT_BUFFER_SAFETY_FACTOR * num_frames;

  for (int ch = 0; ch < in_channels; ch++)
  {
    if ((long)this->_input_buffers[ch].size() < minimum_input_buffer_size)
    {
      long new_buffer_size = 2;
      while (new_buffer_size < minimum_input_buffer_size)
        new_buffer_size *= 2;
      this->_input_buffers[ch].resize(new_buffer_size);
      std::fill(this->_input_buffers[ch].begin(), this->_input_buffers[ch].end(), 0.0f);
    }
  }

  // If we'd run off the end of the input buffer, then we need to move the data
  // back to the start of the buffer and start again. All channels move together.
  const long buffer_size = (long)this->_input_buffers[0].size();
  if (this->_input_buffer_offset + num_frames > buffer_size)
    this->_rewind_buffers_();

  // Put the new samples into the input buffer for each channel
  for (int ch = 0; ch < in_channels; ch++)
  {
    for (long i = this->_input_buffer_offset, j = 0; j < num_frames; i++, j++)
      this->_input_buffers[ch][i] = (float)input[ch][j];
  }

  // Resize output buffers for all output channels
  for (int ch = 0; ch < out_channels; ch++)
  {
    this->_output_buffers[ch].resize(num_frames);
    std::fill(this->_output_buffers[ch].begin(), this->_output_buffers[ch].end(), 0.0f);
  }
}

void nam::Buffer::_rewind_buffers_()
{
  const int in_channels = NumInputChannels();

  // Rewind buffers for all input channels (they all move together)
  for (int ch = 0; ch < in_channels; ch++)
  {
    // Copy the input buffer back
    // RF-1 samples because we've got at least one new one inbound.
    for (long i = 0, j = this->_input_buffer_offset - this->_receptive_field; i < this->_receptive_field; i++, j++)
      this->_input_buffers[ch][i] = this->_input_buffers[ch][j];
  }
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

nam::Linear::Linear(const int in_channels, const int out_channels, const int receptive_field, const bool _bias,
                    const std::vector<float>& weights, const double expected_sample_rate)
: nam::Buffer(in_channels, out_channels, receptive_field, expected_sample_rate)
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

void nam::Linear::process(NAM_SAMPLE** input, NAM_SAMPLE** output, const int num_frames)
{
  this->nam::Buffer::_update_buffers_(input, num_frames);

  const int in_channels = NumInputChannels();
  const int out_channels = NumOutputChannels();

  // For now, Linear processes each input channel independently to corresponding output channel
  // This is a simple implementation - can be extended later for cross-channel mixing
  const int channelsToProcess = std::min(in_channels, out_channels);

  // Main computation!
  for (int ch = 0; ch < channelsToProcess; ch++)
  {
    for (int i = 0; i < num_frames; i++)
    {
      const long offset = this->_input_buffer_offset - this->_weight.size() + i + 1;
      auto input_vec = Eigen::Map<const Eigen::VectorXf>(&this->_input_buffers[ch][offset], this->_receptive_field);
      output[ch][i] = this->_bias + this->_weight.dot(input_vec);
    }
  }

  // Zero out any extra output channels
  for (int ch = channelsToProcess; ch < out_channels; ch++)
  {
    for (int i = 0; i < num_frames; i++)
      output[ch][i] = (NAM_SAMPLE)0.0;
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
  // Default to 1 channel in/out for backward compatibility
  const int in_channels = config.value("in_channels", 1);
  const int out_channels = config.value("out_channels", 1);
  return std::make_unique<nam::Linear>(in_channels, out_channels, receptive_field, bias, weights, expectedSampleRate);
}

// NN modules =================================================================

// Conv1x1 ====================================================================

nam::Conv1x1::Conv1x1(const int in_channels, const int out_channels, const bool _bias, const int groups)
{
  // Validate that channels divide evenly by groups
  if (in_channels % groups != 0)
  {
    throw std::runtime_error("in_channels (" + std::to_string(in_channels) + ") must be divisible by numGroups ("
                             + std::to_string(groups) + ")");
  }
  if (out_channels % groups != 0)
  {
    throw std::runtime_error("out_channels (" + std::to_string(out_channels) + ") must be divisible by numGroups ("
                             + std::to_string(groups) + ")");
  }

  this->_num_groups = groups;
  this->_do_bias = _bias;

  // Check for depthwise convolution: groups == in_channels == out_channels
  // In this case, each channel is processed independently with a single weight,
  // so we can use efficient element-wise multiplication instead of matrix multiplication.
  this->_is_depthwise = (groups == in_channels && in_channels == out_channels);

  if (this->_is_depthwise)
  {
    // Depthwise: store one weight per channel
    this->_channels = in_channels;
    this->_depthwise_weight.resize(in_channels);
    this->_depthwise_weight.setZero();
    // Clear the matrix weight (not used)
    this->_weight.resize(0, 0);
  }
  else
  {
    // Non-depthwise: store full weight matrix (block-diagonal for grouped convolutions)
    this->_weight.resize(out_channels, in_channels);
    this->_weight.setZero();
    this->_channels = 0;
  }

  if (_bias)
  {
    this->_bias.resize(out_channels);
    this->_bias.setZero();
  }
}


void nam::Conv1x1::SetMaxBufferSize(const int maxBufferSize)
{
  _output.resize(get_out_channels(), maxBufferSize);
}

void nam::Conv1x1::set_weights_(std::vector<float>::iterator& weights)
{
  if (this->_is_depthwise)
  {
    // Depthwise convolution: one weight per channel
    for (int c = 0; c < this->_channels; c++)
    {
      this->_depthwise_weight(c) = *(weights++);
    }
  }
  else if (this->_weight.size() > 0)
  {
    const long out_channels = this->_weight.rows();
    const long in_channels = this->_weight.cols();
    const int numGroups = this->_num_groups;
    const long out_per_group = out_channels / numGroups;
    const long in_per_group = in_channels / numGroups;

    // For grouped convolutions, weights are organized per group
    // Weight layout: weights are [group0, group1, ..., groupN-1]
    // Each group's weight matrix is (out_channels/numGroups, in_channels/numGroups)
    for (int g = 0; g < numGroups; g++)
    {
      for (auto i = 0; i < out_per_group; i++)
      {
        for (auto j = 0; j < in_per_group; j++)
        {
          this->_weight(g * out_per_group + i, g * in_per_group + j) = *(weights++);
        }
      }
    }
  }
  if (this->_do_bias)
    for (int i = 0; i < this->_bias.size(); i++)
      this->_bias(i) = *(weights++);
}

long nam::Conv1x1::get_out_channels() const
{
  if (this->_is_depthwise)
    return this->_channels;
  return this->_weight.rows();
}

long nam::Conv1x1::get_in_channels() const
{
  if (this->_is_depthwise)
    return this->_channels;
  return this->_weight.cols();
}

Eigen::MatrixXf nam::Conv1x1::process(const Eigen::MatrixXf& input, const int num_frames) const
{
  Eigen::MatrixXf result(get_out_channels(), num_frames);

  if (this->_is_depthwise)
  {
    // Depthwise convolution: efficient element-wise multiplication
    // Each channel is scaled by its corresponding weight
    result.noalias() = this->_depthwise_weight.asDiagonal() * input.leftCols(num_frames);
  }
  else
  {
    // Single GEMM for all cases - block-diagonal zero structure handles grouping
    result.noalias() = this->_weight * input.leftCols(num_frames);
  }

  if (this->_do_bias)
    result.colwise() += this->_bias;

  return result;
}

void nam::Conv1x1::process_(const Eigen::Ref<const Eigen::MatrixXf>& input, const int num_frames)
{
  assert(num_frames <= _output.cols());

  if (this->_is_depthwise)
  {
    // Depthwise convolution: efficient element-wise multiplication
    // Each channel is scaled by its corresponding weight
    _output.leftCols(num_frames).noalias() = this->_depthwise_weight.asDiagonal() * input.leftCols(num_frames);
  }
  else
  {
#ifdef NAM_USE_INLINE_GEMM
    // Hand-optimized GEMM for small matrices (1x1 convolution)
    // output(out_ch, frames) = weight(out_ch, in_ch) * input(in_ch, frames)
    const int out_ch = (int)get_out_channels();
    const int in_ch = (int)get_in_channels();
    const float* __restrict__ input_ptr = input.data();
    const float* __restrict__ weight_ptr = this->_weight.data();
    float* __restrict__ output_ptr = _output.data();
    // Use outerStride() instead of in_ch to correctly handle non-contiguous
    // block expressions (e.g. topRows()) where outerStride > rows
    const int in_stride = (int)input.outerStride();

    // Specialized paths for common small sizes
    if (out_ch == 2 && in_ch == 1)
    {
      const float w0 = weight_ptr[0], w1 = weight_ptr[1];
      for (int f = 0; f < num_frames; f++)
      {
        const float in_val = input_ptr[f * in_stride];
        output_ptr[f * 2]     = w0 * in_val;
        output_ptr[f * 2 + 1] = w1 * in_val;
      }
    }
    else if (out_ch == 4 && in_ch == 1)
    {
      const float w0 = weight_ptr[0], w1 = weight_ptr[1];
      const float w2 = weight_ptr[2], w3 = weight_ptr[3];
      for (int f = 0; f < num_frames; f++)
      {
        const float in_val = input_ptr[f * in_stride];
        output_ptr[f * 4]     = w0 * in_val;
        output_ptr[f * 4 + 1] = w1 * in_val;
        output_ptr[f * 4 + 2] = w2 * in_val;
        output_ptr[f * 4 + 3] = w3 * in_val;
      }
    }
    else if (out_ch == 1 && in_ch == 2)
    {
      const float w0 = weight_ptr[0], w1 = weight_ptr[1];
      for (int f = 0; f < num_frames; f++)
      {
        const float* __restrict__ in_col = input_ptr + f * in_stride;
        output_ptr[f] = w0 * in_col[0] + w1 * in_col[1];
      }
    }
    else if (out_ch == 2 && in_ch == 2)
    {
      // 2x2 fully unrolled
      const float w00 = weight_ptr[0], w10 = weight_ptr[1];
      const float w01 = weight_ptr[2], w11 = weight_ptr[3];
      for (int f = 0; f < num_frames; f++)
      {
        const float* __restrict__ in_col = input_ptr + f * in_stride;
        const float i0 = in_col[0];
        const float i1 = in_col[1];
        output_ptr[f * 2]     = w00 * i0 + w01 * i1;
        output_ptr[f * 2 + 1] = w10 * i0 + w11 * i1;
      }
    }
    else if (out_ch == 2 && in_ch == 4)
    {
      const float w00 = weight_ptr[0], w10 = weight_ptr[1];
      const float w01 = weight_ptr[2], w11 = weight_ptr[3];
      const float w02 = weight_ptr[4], w12 = weight_ptr[5];
      const float w03 = weight_ptr[6], w13 = weight_ptr[7];
      for (int f = 0; f < num_frames; f++)
      {
        const float* __restrict__ in_col = input_ptr + f * in_stride;
        const float i0 = in_col[0];
        const float i1 = in_col[1];
        const float i2 = in_col[2];
        const float i3 = in_col[3];
        output_ptr[f * 2]     = w00 * i0 + w01 * i1 + w02 * i2 + w03 * i3;
        output_ptr[f * 2 + 1] = w10 * i0 + w11 * i1 + w12 * i2 + w13 * i3;
      }
    }
    else if (out_ch == 1 && in_ch == 4)
    {
      const float w0 = weight_ptr[0], w1 = weight_ptr[1];
      const float w2 = weight_ptr[2], w3 = weight_ptr[3];
      for (int f = 0; f < num_frames; f++)
      {
        const float* __restrict__ in_col = input_ptr + f * in_stride;
        output_ptr[f] = w0 * in_col[0] + w1 * in_col[1]
                      + w2 * in_col[2] + w3 * in_col[3];
      }
    }
    else if (out_ch == 4 && in_ch == 2)
    {
      const float w00 = weight_ptr[0], w10 = weight_ptr[1], w20 = weight_ptr[2], w30 = weight_ptr[3];
      const float w01 = weight_ptr[4], w11 = weight_ptr[5], w21 = weight_ptr[6], w31 = weight_ptr[7];
      for (int f = 0; f < num_frames; f++)
      {
        const float* __restrict__ in_col = input_ptr + f * in_stride;
        const float i0 = in_col[0];
        const float i1 = in_col[1];
        output_ptr[f * 4]     = w00 * i0 + w01 * i1;
        output_ptr[f * 4 + 1] = w10 * i0 + w11 * i1;
        output_ptr[f * 4 + 2] = w20 * i0 + w21 * i1;
        output_ptr[f * 4 + 3] = w30 * i0 + w31 * i1;
      }
    }
    else if (out_ch == 3 && in_ch == 3)
    {
      const float w00 = weight_ptr[0], w10 = weight_ptr[1], w20 = weight_ptr[2];
      const float w01 = weight_ptr[3], w11 = weight_ptr[4], w21 = weight_ptr[5];
      const float w02 = weight_ptr[6], w12 = weight_ptr[7], w22 = weight_ptr[8];
      for (int f = 0; f < num_frames; f++)
      {
        const float* __restrict__ in_col = input_ptr + f * in_stride;
        const float i0 = in_col[0];
        const float i1 = in_col[1];
        const float i2 = in_col[2];
        output_ptr[f * 3]     = w00 * i0 + w01 * i1 + w02 * i2;
        output_ptr[f * 3 + 1] = w10 * i0 + w11 * i1 + w12 * i2;
        output_ptr[f * 3 + 2] = w20 * i0 + w21 * i1 + w22 * i2;
      }
    }
    else if (out_ch == 4 && in_ch == 4)
    {
      const float w00 = weight_ptr[0],  w10 = weight_ptr[1],  w20 = weight_ptr[2],  w30 = weight_ptr[3];
      const float w01 = weight_ptr[4],  w11 = weight_ptr[5],  w21 = weight_ptr[6],  w31 = weight_ptr[7];
      const float w02 = weight_ptr[8],  w12 = weight_ptr[9],  w22 = weight_ptr[10], w32 = weight_ptr[11];
      const float w03 = weight_ptr[12], w13 = weight_ptr[13], w23 = weight_ptr[14], w33 = weight_ptr[15];
      for (int f = 0; f < num_frames; f++)
      {
        const float* __restrict__ in_col = input_ptr + f * in_stride;
        const float i0 = in_col[0];
        const float i1 = in_col[1];
        const float i2 = in_col[2];
        const float i3 = in_col[3];
        output_ptr[f * 4]     = w00 * i0 + w01 * i1 + w02 * i2 + w03 * i3;
        output_ptr[f * 4 + 1] = w10 * i0 + w11 * i1 + w12 * i2 + w13 * i3;
        output_ptr[f * 4 + 2] = w20 * i0 + w21 * i1 + w22 * i2 + w23 * i3;
        output_ptr[f * 4 + 3] = w30 * i0 + w31 * i1 + w32 * i2 + w33 * i3;
      }
    }
    else if (out_ch == 6 && in_ch == 6)
    {
      for (int f = 0; f < num_frames; f++)
      {
        const float* __restrict__ in_col = input_ptr + f * in_stride;
        float* __restrict__ out_col = output_ptr + f * 6;
        const float i0 = in_col[0], i1 = in_col[1], i2 = in_col[2];
        const float i3 = in_col[3], i4 = in_col[4], i5 = in_col[5];
        for (int o = 0; o < 6; o++)
        {
          out_col[o] = weight_ptr[o] * i0 + weight_ptr[6 + o] * i1 + weight_ptr[12 + o] * i2
                     + weight_ptr[18 + o] * i3 + weight_ptr[24 + o] * i4 + weight_ptr[30 + o] * i5;
        }
      }
    }
    else if (out_ch == 8 && in_ch == 8)
    {
      for (int f = 0; f < num_frames; f++)
      {
        const float* __restrict__ in_col = input_ptr + f * in_stride;
        float* __restrict__ out_col = output_ptr + f * 8;
        const float i0 = in_col[0], i1 = in_col[1], i2 = in_col[2], i3 = in_col[3];
        const float i4 = in_col[4], i5 = in_col[5], i6 = in_col[6], i7 = in_col[7];
        for (int o = 0; o < 8; o++)
        {
          out_col[o] = weight_ptr[o] * i0 + weight_ptr[8 + o] * i1 + weight_ptr[16 + o] * i2 + weight_ptr[24 + o] * i3
                     + weight_ptr[32 + o] * i4 + weight_ptr[40 + o] * i5 + weight_ptr[48 + o] * i6 + weight_ptr[56 + o] * i7;
        }
      }
    }
    else if (out_ch == 4 && in_ch == 8)
    {
      for (int f = 0; f < num_frames; f++)
      {
        const float* __restrict__ in_col = input_ptr + f * in_stride;
        float* __restrict__ out_col = output_ptr + f * 4;
        const float i0 = in_col[0], i1 = in_col[1], i2 = in_col[2], i3 = in_col[3];
        const float i4 = in_col[4], i5 = in_col[5], i6 = in_col[6], i7 = in_col[7];
        for (int o = 0; o < 4; o++)
        {
          out_col[o] = weight_ptr[o] * i0 + weight_ptr[4 + o] * i1 + weight_ptr[8 + o] * i2 + weight_ptr[12 + o] * i3
                     + weight_ptr[16 + o] * i4 + weight_ptr[20 + o] * i5 + weight_ptr[24 + o] * i6 + weight_ptr[28 + o] * i7;
        }
      }
    }
    else if (out_ch == 8 && in_ch == 4)
    {
      for (int f = 0; f < num_frames; f++)
      {
        const float* __restrict__ in_col = input_ptr + f * in_stride;
        float* __restrict__ out_col = output_ptr + f * 8;
        const float i0 = in_col[0], i1 = in_col[1], i2 = in_col[2], i3 = in_col[3];
        for (int o = 0; o < 8; o++)
        {
          out_col[o] = weight_ptr[o] * i0 + weight_ptr[8 + o] * i1 + weight_ptr[16 + o] * i2 + weight_ptr[24 + o] * i3;
        }
      }
    }
    else
    {
      // Fall back to Eigen for larger matrices where it's more efficient
      _output.leftCols(num_frames).noalias() = this->_weight * input.leftCols(num_frames);
    }
#else
    // Single GEMM for all cases - block-diagonal zero structure handles grouping
    _output.leftCols(num_frames).noalias() = this->_weight * input.leftCols(num_frames);
#endif
  }

  if (this->_do_bias)
  {
#ifdef NAM_USE_INLINE_GEMM
    const int out_ch = (int)get_out_channels();
    float* __restrict__ output_ptr = _output.data();
    const float* __restrict__ bias_ptr = this->_bias.data();

    // Specialized paths for common small channel counts
    if (out_ch == 2)
    {
      const float b0 = bias_ptr[0], b1 = bias_ptr[1];
      for (int f = 0; f < num_frames; f++)
      {
        const int off = f * 2;
        output_ptr[off] += b0;
        output_ptr[off + 1] += b1;
      }
    }
    else if (out_ch == 4)
    {
      const float b0 = bias_ptr[0], b1 = bias_ptr[1];
      const float b2 = bias_ptr[2], b3 = bias_ptr[3];
      for (int f = 0; f < num_frames; f++)
      {
        const int off = f * 4;
        output_ptr[off] += b0;
        output_ptr[off + 1] += b1;
        output_ptr[off + 2] += b2;
        output_ptr[off + 3] += b3;
      }
    }
    else
    {
      for (int f = 0; f < num_frames; f++)
      {
        float* __restrict__ out_col = output_ptr + f * out_ch;
        for (int o = 0; o < out_ch; o++)
        {
          out_col[o] += bias_ptr[o];
        }
      }
    }
#else
    _output.leftCols(num_frames).colwise() += this->_bias;
#endif
  }
}
