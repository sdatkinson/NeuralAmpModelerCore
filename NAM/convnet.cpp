#include <algorithm> // std::max_element
#include <algorithm>
#include <cmath> // pow, tanh, expf

#include <fstream>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "dsp.h"
#include "json.hpp"
#include "util.h"
#include "convnet.h"

convnet::BatchNorm::BatchNorm(const int dim, std::vector<float>::iterator& params)
{
  // Extract from param buffer
  Eigen::VectorXf running_mean(dim);
  Eigen::VectorXf running_var(dim);
  Eigen::VectorXf _weight(dim);
  Eigen::VectorXf _bias(dim);
  for (int i = 0; i < dim; i++)
    running_mean(i) = *(params++);
  for (int i = 0; i < dim; i++)
    running_var(i) = *(params++);
  for (int i = 0; i < dim; i++)
    _weight(i) = *(params++);
  for (int i = 0; i < dim; i++)
    _bias(i) = *(params++);
  float eps = *(params++);

  // Convert to scale & loc
  this->scale.resize(dim);
  this->loc.resize(dim);
  for (int i = 0; i < dim; i++)
    this->scale(i) = _weight(i) / sqrt(eps + running_var(i));
  this->loc = _bias - this->scale.cwiseProduct(running_mean);
}

void convnet::BatchNorm::process_(Eigen::MatrixXf& x, const long i_start, const long i_end) const
{
  // todo using colwise?
  // #speed but conv probably dominates
  for (auto i = i_start; i < i_end; i++)
  {
    x.col(i) = x.col(i).cwiseProduct(this->scale);
    x.col(i) += this->loc;
  }
}

void convnet::ConvNetBlock::set_params_(const int in_channels, const int out_channels, const int _dilation,
                                        const bool batchnorm, const std::string activation,
                                        std::vector<float>::iterator& params)
{
  this->_batchnorm = batchnorm;
  // HACK 2 kernel
  this->conv.set_size_and_params_(in_channels, out_channels, 2, _dilation, !batchnorm, params);
  if (this->_batchnorm)
    this->batchnorm = BatchNorm(out_channels, params);
  this->activation = activations::Activation::get_activation(activation);
}

void convnet::ConvNetBlock::process_(const Eigen::MatrixXf& input, Eigen::MatrixXf& output, const long i_start,
                                     const long i_end) const
{
  const long ncols = i_end - i_start;
  this->conv.process_(input, output, i_start, ncols, i_start);
  if (this->_batchnorm)
    this->batchnorm.process_(output, i_start, i_end);

  this->activation->apply(output.middleCols(i_start, ncols));
}

long convnet::ConvNetBlock::get_out_channels() const
{
  return this->conv.get_out_channels();
}

convnet::_Head::_Head(const int channels, std::vector<float>::iterator& params)
{
  this->_weight.resize(channels);
  for (int i = 0; i < channels; i++)
    this->_weight[i] = *(params++);
  this->_bias = *(params++);
}

void convnet::_Head::process_(const Eigen::MatrixXf& input, Eigen::VectorXf& output, const long i_start,
                              const long i_end) const
{
  const long length = i_end - i_start;
  output.resize(length);
  for (long i = 0, j = i_start; i < length; i++, j++)
    output(i) = this->_bias + input.col(j).dot(this->_weight);
}

convnet::ConvNet::ConvNet(const int channels, const std::vector<int>& dilations, const bool batchnorm,
                          const std::string activation, std::vector<float>& params)
: ConvNet(TARGET_DSP_LOUDNESS, channels, dilations, batchnorm, activation, params)
{
}

convnet::ConvNet::ConvNet(const double loudness, const int channels, const std::vector<int>& dilations,
                          const bool batchnorm, const std::string activation, std::vector<float>& params)
: Buffer(loudness, *std::max_element(dilations.begin(), dilations.end()))
{
  this->_verify_params(channels, dilations, batchnorm, params.size());
  this->_blocks.resize(dilations.size());
  std::vector<float>::iterator it = params.begin();
  for (int i = 0; i < dilations.size(); i++)
    this->_blocks[i].set_params_(i == 0 ? 1 : channels, channels, dilations[i], batchnorm, activation, it);
  this->_block_vals.resize(this->_blocks.size() + 1);
  this->_head = _Head(channels, it);
  if (it != params.end())
    throw std::runtime_error("Didn't touch all the params when initializing wavenet");
  this->_reset_anti_pop_();
}

void convnet::ConvNet::_process_core_()
{
  this->_update_buffers_();
  // Main computation!
  const long i_start = this->_input_buffer_offset;
  const long num_frames = this->_input_post_gain.size();
  const long i_end = i_start + num_frames;
  // TODO one unnecessary copy :/ #speed
  for (auto i = i_start; i < i_end; i++)
    this->_block_vals[0](0, i) = this->_input_buffer[i];
  for (auto i = 0; i < this->_blocks.size(); i++)
    this->_blocks[i].process_(this->_block_vals[i], this->_block_vals[i + 1], i_start, i_end);
  // TODO clean up this allocation
  this->_head.process_(this->_block_vals[this->_blocks.size()], this->_head_output, i_start, i_end);
  // Copy to required output array (TODO tighten this up)
  for (int s = 0; s < num_frames; s++)
    this->_core_dsp_output[s] = this->_head_output(s);
  // Apply anti-pop
  this->_anti_pop_();
}

void convnet::ConvNet::_verify_params(const int channels, const std::vector<int>& dilations, const bool batchnorm,
                                      const size_t actual_params)
{
  // TODO
}

void convnet::ConvNet::_update_buffers_()
{
  this->Buffer::_update_buffers_();
  const long buffer_size = this->_input_buffer.size();
  this->_block_vals[0].resize(1, buffer_size);
  for (long i = 1; i < this->_block_vals.size(); i++)
    this->_block_vals[i].resize(this->_blocks[i - 1].get_out_channels(), buffer_size);
}

void convnet::ConvNet::_rewind_buffers_()
{
  // Need to rewind the block vals first because Buffer::rewind_buffers()
  // resets the offset index
  // The last _block_vals is the output of the last block and doesn't need to be
  // rewound.
  for (long k = 0; k < this->_block_vals.size() - 1; k++)
  {
    // We actually don't need to pull back a lot...just as far as the first
    // input sample would grab from dilation
    const long _dilation = this->_blocks[k].conv.get_dilation();
    for (long i = this->_receptive_field - _dilation, j = this->_input_buffer_offset - _dilation;
         j < this->_input_buffer_offset; i++, j++)
      for (long r = 0; r < this->_block_vals[k].rows(); r++)
        this->_block_vals[k](r, i) = this->_block_vals[k](r, j);
  }
  // Now we can do the rest of the rewind
  this->Buffer::_rewind_buffers_();
}

void convnet::ConvNet::_anti_pop_()
{
  if (this->_anti_pop_countdown >= this->_anti_pop_ramp)
    return;
  const float slope = 1.0f / float(this->_anti_pop_ramp);
  for (int i = 0; i < this->_core_dsp_output.size(); i++)
  {
    if (this->_anti_pop_countdown >= this->_anti_pop_ramp)
      break;
    const float gain = std::max(slope * float(this->_anti_pop_countdown), float(0.0));
    this->_core_dsp_output[i] *= gain;
    this->_anti_pop_countdown++;
  }
}

void convnet::ConvNet::_reset_anti_pop_()
{
  // You need the "real" receptive field, not the buffers.
  long receptive_field = 1;
  for (int i = 0; i < this->_blocks.size(); i++)
    receptive_field += this->_blocks[i].conv.get_dilation();
  this->_anti_pop_countdown = -receptive_field;
}
