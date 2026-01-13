#include <algorithm> // std::max_element
#include <algorithm>
#include <cmath> // pow, tanh, expf
#include <filesystem>
#include <fstream>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "dsp.h"
#include "registry.h"
#include "convnet.h"

nam::convnet::BatchNorm::BatchNorm(const int dim, std::vector<float>::iterator& weights)
{
  // Extract from param buffer
  Eigen::VectorXf running_mean(dim);
  Eigen::VectorXf running_var(dim);
  Eigen::VectorXf _weight(dim);
  Eigen::VectorXf _bias(dim);
  for (int i = 0; i < dim; i++)
    running_mean(i) = *(weights++);
  for (int i = 0; i < dim; i++)
    running_var(i) = *(weights++);
  for (int i = 0; i < dim; i++)
    _weight(i) = *(weights++);
  for (int i = 0; i < dim; i++)
    _bias(i) = *(weights++);
  float eps = *(weights++);

  // Convert to scale & loc
  this->scale.resize(dim);
  this->loc.resize(dim);
  for (int i = 0; i < dim; i++)
    this->scale(i) = _weight(i) / sqrt(eps + running_var(i));
  this->loc = _bias - this->scale.cwiseProduct(running_mean);
}

void nam::convnet::BatchNorm::process_(Eigen::MatrixXf& x, const long i_start, const long i_end) const
{
  // todo using colwise?
  // #speed but conv probably dominates
  for (auto i = i_start; i < i_end; i++)
  {
    x.col(i) = x.col(i).cwiseProduct(this->scale);
    x.col(i) += this->loc;
  }
}

void nam::convnet::ConvNetBlock::set_weights_(const int in_channels, const int out_channels, const int _dilation,
                                              const bool batchnorm, const std::string activation,
                                              std::vector<float>::iterator& weights)
{
  this->_batchnorm = batchnorm;
  // HACK 2 kernel
  this->conv.set_size_and_weights_(in_channels, out_channels, 2, _dilation, !batchnorm, weights);
  if (this->_batchnorm)
    this->batchnorm = BatchNorm(out_channels, weights);
  this->activation = activations::Activation::get_activation(activation);
}

void nam::convnet::ConvNetBlock::Process(const Eigen::MatrixXf& input, const int num_frames)
{
  // Process input with Conv1D
  this->conv.Process(input, num_frames);

  // Get output from Conv1D (this is a block reference to _output buffer)
  auto conv_output_block = this->conv.GetOutput(num_frames);

  // For batchnorm, we need a matrix reference (not a block)
  // Create a temporary matrix from the block, process it, then copy back
  Eigen::MatrixXf temp_output = conv_output_block;

  // Apply batchnorm if needed
  if (this->_batchnorm)
  {
    // Batchnorm expects indices, so we use 0 to num_frames for our temp matrix
    this->batchnorm.process_(temp_output, 0, num_frames);
  }

  // Apply activation
  this->activation->apply(temp_output);

  // Copy back to Conv1D's output buffer (so GetOutput() returns the processed result)
  conv_output_block = temp_output;
}

Eigen::Block<Eigen::MatrixXf> nam::convnet::ConvNetBlock::GetOutput(const int num_frames)
{
  // FIXME needs to own this output.
  return this->conv_output_block.leftCols(num_frames);
}

void nam::convnet::ConvNetBlock::process_(const Eigen::MatrixXf& input, Eigen::MatrixXf& output, const long i_start,
                                          const long i_end)
{
  const long ncols = i_end - i_start;
  // Extract input slice and process with Conv1D
  Eigen::MatrixXf input_slice = input.middleCols(i_start, ncols);
  this->conv.Process(input_slice, (int)ncols);

  // Get output from Conv1D (this is a block reference to _output buffer)
  auto conv_output_block = this->conv.GetOutput((int)ncols);

  // For batchnorm, we need a matrix reference (not a block)
  // Create a temporary matrix from the block, process it, then copy back
  Eigen::MatrixXf temp_output = conv_output_block;

  // Apply batchnorm if needed
  if (this->_batchnorm)
  {
    // Batchnorm expects indices, so we use 0 to ncols for our temp matrix
    this->batchnorm.process_(temp_output, 0, ncols);
  }

  // Apply activation
  this->activation->apply(temp_output);

  // Copy to Conv1D's output buffer and to output matrix
  conv_output_block = temp_output;
  output.middleCols(i_start, ncols) = temp_output;
}

long nam::convnet::ConvNetBlock::get_out_channels() const
{
  return this->conv.get_out_channels();
}

nam::convnet::_Head::_Head(const int channels, std::vector<float>::iterator& weights)
{
  this->_weight.resize(channels);
  for (int i = 0; i < channels; i++)
    this->_weight[i] = *(weights++);
  this->_bias = *(weights++);
}

void nam::convnet::_Head::process_(const Eigen::MatrixXf& input, Eigen::VectorXf& output, const long i_start,
                                   const long i_end) const
{
  const long length = i_end - i_start;
  output.resize(length);
  for (long i = 0, j = i_start; i < length; i++, j++)
    output(i) = this->_bias + input.col(j).dot(this->_weight);
}

nam::convnet::ConvNet::ConvNet(const int channels, const std::vector<int>& dilations, const bool batchnorm,
                               const std::string activation, std::vector<float>& weights,
                               const double expected_sample_rate)
: Buffer(*std::max_element(dilations.begin(), dilations.end()), expected_sample_rate)
{
  this->_verify_weights(channels, dilations, batchnorm, weights.size());
  this->_blocks.resize(dilations.size());
  std::vector<float>::iterator it = weights.begin();
  for (size_t i = 0; i < dilations.size(); i++)
    this->_blocks[i].set_weights_(i == 0 ? 1 : channels, channels, dilations[i], batchnorm, activation, it);
  // Only need _block_vals for the head (one entry)
  // Conv1D layers manage their own buffers now
  this->_block_vals.resize(1);
  this->_block_vals[0].setZero();
  std::fill(this->_input_buffer.begin(), this->_input_buffer.end(), 0.0f);
  this->_head = _Head(channels, it);
  if (it != weights.end())
    throw std::runtime_error("Didn't touch all the weights when initializing ConvNet");

  mPrewarmSamples = 1;
  for (size_t i = 0; i < dilations.size(); i++)
    mPrewarmSamples += dilations[i];
}


void nam::convnet::ConvNet::process(NAM_SAMPLE* input, NAM_SAMPLE* output, const int num_frames)

{
  this->_update_buffers_(input, num_frames);
  // Main computation!
  const long i_start = this->_input_buffer_offset;
  const long i_end = i_start + num_frames;

  // Convert input buffer to matrix for first layer
  Eigen::MatrixXf input_matrix(1, num_frames);
  for (int i = 0; i < num_frames; i++)
    input_matrix(0, i) = this->_input_buffer[i_start + i];

  // Process through ConvNetBlock layers
  // Each block now uses Conv1D's internal buffers via Process() and GetOutput()
  for (size_t i = 0; i < this->_blocks.size(); i++)
  {
    // Get input for this block
    Eigen::MatrixXf block_input;
    if (i == 0)
    {
      // First block uses the input matrix
      block_input = input_matrix;
    }
    else
    {
      // Subsequent blocks use output from previous block
      auto prev_output = this->_blocks[i - 1].GetOutput(num_frames);
      block_input = prev_output; // Copy to matrix
    }

    // Process block (handles Conv1D, batchnorm, and activation internally)
    this->_blocks[i].Process(block_input, num_frames);
  }

  // Process head with output from last Conv1D
  // Head still needs the old interface, so we need to provide it via a matrix
  // We still need _block_vals[0] for the head interface
  if (this->_block_vals[0].rows() != this->_blocks.back().get_out_channels()
      || this->_block_vals[0].cols() != (long)this->_input_buffer.size())
  {
    this->_block_vals[0].resize(this->_blocks.back().get_out_channels(), this->_input_buffer.size());
  }
  // Copy last block output to _block_vals for head
  auto last_output = this->_blocks.back().GetOutput(num_frames);
  this->_block_vals[0].middleCols(i_start, num_frames) = last_output;

  this->_head.process_(this->_block_vals[0], this->_head_output, i_start, i_end);

  // Copy to required output array
  for (int s = 0; s < num_frames; s++)
    output[s] = this->_head_output(s);

  // Prepare for next call:
  nam::Buffer::_advance_input_buffer_(num_frames);
}

void nam::convnet::ConvNet::_verify_weights(const int channels, const std::vector<int>& dilations, const bool batchnorm,
                                            const size_t actual_weights)
{
  // TODO
}

void nam::convnet::ConvNet::SetMaxBufferSize(const int maxBufferSize)
{
  nam::Buffer::SetMaxBufferSize(maxBufferSize);

  // Reset all Conv1D instances with the new buffer size
  // Get sample rate from parent (or use -1.0 if not set)
  double sampleRate = GetExpectedSampleRate(); // Use the expected sample rate
  for (auto& block : _blocks)
  {
    block.conv.SetMaxBufferSize(maxBufferSize);
  }
}

void nam::convnet::ConvNet::_update_buffers_(NAM_SAMPLE* input, const int num_frames)
{
  this->Buffer::_update_buffers_(input, num_frames);

  const long buffer_size = (long)this->_input_buffer.size();

  // Only need _block_vals[0] for the head
  // Conv1D layers manage their own buffers now
  if (this->_block_vals[0].rows() != this->_blocks.back().get_out_channels()
      || this->_block_vals[0].cols() != buffer_size)
  {
    this->_block_vals[0].resize(this->_blocks.back().get_out_channels(), buffer_size);
    this->_block_vals[0].setZero();
  }
}

void nam::convnet::ConvNet::_rewind_buffers_()
{
  // Conv1D instances now manage their own ring buffers and handle rewinding internally
  // So we don't need to rewind _block_vals for Conv1D layers
  // We only need _block_vals for the head, and it doesn't need rewinding since it's only used
  // for the current frame range

  // Just rewind the input buffer (for Buffer base class)
  this->Buffer::_rewind_buffers_();
}

// Factory
std::unique_ptr<nam::DSP> nam::convnet::Factory(const nlohmann::json& config, std::vector<float>& weights,
                                                const double expectedSampleRate)
{
  const int channels = config["channels"];
  const std::vector<int> dilations = config["dilations"];
  const bool batchnorm = config["batchnorm"];
  const std::string activation = config["activation"];
  return std::make_unique<nam::convnet::ConvNet>(
    channels, dilations, batchnorm, activation, weights, expectedSampleRate);
}

namespace
{
static nam::factory::Helper _register_ConvNet("ConvNet", nam::convnet::Factory);
}
