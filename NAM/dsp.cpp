#include <algorithm> // std::max_element
#include <algorithm>
#include <cmath> // pow, tanh, expf
#include <filesystem>
#include <fstream>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "dsp.h"
#include "json.hpp"
#include "util.h"

#define tanh_impl_ std::tanh
// #define tanh_impl_ fast_tanh_

constexpr const long _INPUT_BUFFER_SAFETY_FACTOR = 32;

DSP::DSP(const double expected_sample_rate)
: mLoudness(TARGET_DSP_LOUDNESS)
, mExpectedSampleRate(expected_sample_rate)
, mNormalizeOutputLoudness(false)
, _stale_params(true)
{
}

DSP::DSP(const double loudness, const double expected_sample_rate)
: mLoudness(loudness)
, mExpectedSampleRate(expected_sample_rate)
, mNormalizeOutputLoudness(false)
, _stale_params(true)
{
}

void DSP::process(NAM_SAMPLE** inputs, NAM_SAMPLE** outputs, const int num_channels, const int num_frames,
                  const double input_gain, const double output_gain,
                  const std::unordered_map<std::string, double>& params)
{
  this->_get_params_(params);
  this->_apply_input_level_(inputs, num_channels, num_frames, input_gain);
  this->_ensure_core_dsp_output_ready_();
  this->_process_core_();
  this->_apply_output_level_(outputs, num_channels, num_frames, output_gain);
}

void DSP::finalize_(const int num_frames) {}

void DSP::_get_params_(const std::unordered_map<std::string, double>& input_params)
{
  this->_stale_params = false;
  for (auto it = input_params.begin(); it != input_params.end(); ++it)
  {
    const std::string key = util::lowercase(it->first);
    const double value = it->second;
    if (this->_params.find(key) == this->_params.end()) // Not contained
      this->_stale_params = true;
    else if (this->_params[key] != value) // Contained but new value
      this->_stale_params = true;
    this->_params[key] = value;
  }
}

void DSP::_apply_input_level_(NAM_SAMPLE** inputs, const int num_channels, const int num_frames, const double gain)
{
  // Must match exactly; we're going to use the size of _input_post_gain later
  // for num_frames.
  if ((int)this->_input_post_gain.size() != num_frames)
    this->_input_post_gain.resize(num_frames);
  // MONO ONLY
  const int channel = 0;
  for (int i = 0; i < num_frames; i++)
    this->_input_post_gain[i] = float(gain * inputs[channel][i]);
}

void DSP::_ensure_core_dsp_output_ready_()
{
  if (this->_core_dsp_output.size() < this->_input_post_gain.size())
    this->_core_dsp_output.resize(this->_input_post_gain.size());
}

void DSP::_process_core_()
{
  // Default implementation is the null operation
  for (size_t i = 0; i < this->_input_post_gain.size(); i++)
    this->_core_dsp_output[i] = this->_input_post_gain[i];
}

void DSP::_apply_output_level_(NAM_SAMPLE** outputs, const int num_channels, const int num_frames,
                               const double gain)
{
  const double loudnessGain = pow(10.0, -(this->mLoudness - TARGET_DSP_LOUDNESS) / 20.0);
  const double finalGain = this->mNormalizeOutputLoudness ? gain * loudnessGain : gain;
  for (int c = 0; c < num_channels; c++)
    for (int s = 0; s < num_frames; s++)
      outputs[c][s] = (NAM_SAMPLE)(finalGain * this->_core_dsp_output[s]);
}

// Buffer =====================================================================

Buffer::Buffer(const int receptive_field, const double expected_sample_rate)
: Buffer(TARGET_DSP_LOUDNESS, receptive_field, expected_sample_rate)
{
}

Buffer::Buffer(const double loudness, const int receptive_field, const double expected_sample_rate)
: DSP(loudness, expected_sample_rate)
{
  this->_set_receptive_field(receptive_field);
}

void Buffer::_set_receptive_field(const int new_receptive_field)
{
  this->_set_receptive_field(new_receptive_field, _INPUT_BUFFER_SAFETY_FACTOR * new_receptive_field);
};

void Buffer::_set_receptive_field(const int new_receptive_field, const int input_buffer_size)
{
  this->_receptive_field = new_receptive_field;
  this->_input_buffer.resize(input_buffer_size);
  this->_reset_input_buffer();
}

void Buffer::_update_buffers_()
{
  const long int num_frames = this->_input_post_gain.size();
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
    }
  }

  // If we'd run off the end of the input buffer, then we need to move the data
  // back to the start of the buffer and start again.
  if (this->_input_buffer_offset + num_frames > (long)this->_input_buffer.size())
    this->_rewind_buffers_();
  // Put the new samples into the input buffer
  for (long i = this->_input_buffer_offset, j = 0; j < num_frames; i++, j++)
    this->_input_buffer[i] = this->_input_post_gain[j];
  // And resize the output buffer:
  this->_output_buffer.resize(num_frames);
}

void Buffer::_rewind_buffers_()
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

void Buffer::_reset_input_buffer()
{
  this->_input_buffer_offset = this->_receptive_field;
}

void Buffer::finalize_(const int num_frames)
{
  this->DSP::finalize_(num_frames);
  this->_input_buffer_offset += num_frames;
}

// Linear =====================================================================

Linear::Linear(const int receptive_field, const bool _bias, const std::vector<float>& params,
               const double expected_sample_rate)
: Linear(TARGET_DSP_LOUDNESS, receptive_field, _bias, params, expected_sample_rate)
{
}

Linear::Linear(const double loudness, const int receptive_field, const bool _bias, const std::vector<float>& params,
               const double expected_sample_rate)
: Buffer(loudness, receptive_field, expected_sample_rate)
{
  if ((int)params.size() != (receptive_field + (_bias ? 1 : 0)))
    throw std::runtime_error(
      "Params vector does not match expected size based "
      "on architecture parameters");

  this->_weight.resize(this->_receptive_field);
  // Pass in in reverse order so that dot products work out of the box.
  for (int i = 0; i < this->_receptive_field; i++)
    this->_weight(i) = params[receptive_field - 1 - i];
  this->_bias = _bias ? params[receptive_field] : (float)0.0;
}

void Linear::_process_core_()
{
  this->Buffer::_update_buffers_();

  // Main computation!
  for (size_t i = 0; i < this->_input_post_gain.size(); i++)
  {
    const size_t offset = this->_input_buffer_offset - this->_weight.size() + i + 1;
    auto input = Eigen::Map<const Eigen::VectorXf>(&this->_input_buffer[offset], this->_receptive_field);
    this->_core_dsp_output[i] = this->_bias + this->_weight.dot(input);
  }
}

// NN modules =================================================================

void Conv1D::set_params_(std::vector<float>::iterator& params)
{
  if (this->_weight.size() > 0)
  {
    const long out_channels = this->_weight[0].rows();
    const long in_channels = this->_weight[0].cols();
    // Crazy ordering because that's how it gets flattened.
    for (auto i = 0; i < out_channels; i++)
      for (auto j = 0; j < in_channels; j++)
        for (size_t k = 0; k < this->_weight.size(); k++)
          this->_weight[k](i, j) = *(params++);
  }
  for (long i = 0; i < this->_bias.size(); i++)
    this->_bias(i) = *(params++);
}

void Conv1D::set_size_(const int in_channels, const int out_channels, const int kernel_size, const bool do_bias,
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

void Conv1D::set_size_and_params_(const int in_channels, const int out_channels, const int kernel_size,
                                  const int _dilation, const bool do_bias, std::vector<float>::iterator& params)
{
  this->set_size_(in_channels, out_channels, kernel_size, do_bias, _dilation);
  this->set_params_(params);
}

void Conv1D::process_(const Eigen::MatrixXf& input, Eigen::MatrixXf& output, const long i_start, const long ncols,
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

long Conv1D::get_num_params() const
{
  long num_params = this->_bias.size();
  for (size_t i = 0; i < this->_weight.size(); i++)
    num_params += this->_weight[i].size();
  return num_params;
}

Conv1x1::Conv1x1(const int in_channels, const int out_channels, const bool _bias)
{
  this->_weight.resize(out_channels, in_channels);
  this->_do_bias = _bias;
  if (_bias)
    this->_bias.resize(out_channels);
}

void Conv1x1::set_params_(std::vector<float>::iterator& params)
{
  for (int i = 0; i < this->_weight.rows(); i++)
    for (int j = 0; j < this->_weight.cols(); j++)
      this->_weight(i, j) = *(params++);
  if (this->_do_bias)
    for (int i = 0; i < this->_bias.size(); i++)
      this->_bias(i) = *(params++);
}

Eigen::MatrixXf Conv1x1::process(const Eigen::MatrixXf& input) const
{
  if (this->_do_bias)
    return (this->_weight * input).colwise() + this->_bias;
  else
    return this->_weight * input;
}
