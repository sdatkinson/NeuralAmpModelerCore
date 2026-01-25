#include "conv1d.h"
#include <stdexcept>

namespace nam
{
// Conv1D =====================================================================

void Conv1D::set_weights_(std::vector<float>::iterator& weights)
{
  if (this->_weight.size() > 0)
  {
    const long out_channels = this->_weight[0].rows();
    const long in_channels = this->_weight[0].cols();
    const int numGroups = this->_num_groups;
    const long out_per_group = out_channels / numGroups;
    const long in_per_group = in_channels / numGroups;

    // For grouped convolutions, weights are organized per group
    // Weight layout: for each kernel position k, weights are [group0, group1, ..., groupN-1]
    // Each group's weight matrix is (out_channels/numGroups, in_channels/numGroups)
    // Crazy ordering because that's how it gets flattened.
    for (int g = 0; g < numGroups; g++)
    {
      for (long i = 0; i < out_per_group; ++i)
      {
        for (long j = 0; j < in_per_group; ++j)
        {
          for (size_t k = 0; k < this->_weight.size(); ++k)
          {
            this->_weight[k](g * out_per_group + i, g * in_per_group + j) = *(weights++);
          }
        }
      }
    }
  }
  for (long i = 0; i < this->_bias.size(); i++)
    this->_bias(i) = *(weights++);
}

void Conv1D::set_size_(const int in_channels, const int out_channels, const int kernel_size, const bool do_bias,
                       const int _dilation, const int groups)
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

void Conv1D::set_size_and_weights_(const int in_channels, const int out_channels, const int kernel_size,
                                   const int _dilation, const bool do_bias, const int groups,
                                   std::vector<float>::iterator& weights)
{
  this->set_size_(in_channels, out_channels, kernel_size, do_bias, _dilation, groups);
  this->set_weights_(weights);
}

void Conv1D::SetMaxBufferSize(const int maxBufferSize)
{
  _max_buffer_size = maxBufferSize;

  // Calculate receptive field (maximum lookback needed)
  const long kernel_size = get_kernel_size();
  const long dilation = get_dilation();
  const long receptive_field = kernel_size > 0 ? (kernel_size - 1) * dilation : 0;

  const long in_channels = get_in_channels();

  // Initialize input ring buffer
  // Set max lookback before Reset so that Reset() can use it to calculate storage size
  // Reset() will calculate storage size as: 2 * max_lookback + max_buffer_size
  _input_buffer.SetMaxLookback(receptive_field);
  _input_buffer.Reset(in_channels, maxBufferSize);

  // Pre-allocate output matrix
  const long out_channels = get_out_channels();
  _output.resize(out_channels, maxBufferSize);
  _output.setZero();
}


void Conv1D::Process(const Eigen::MatrixXf& input, const nam::Index num_frames)
{
  // Write input to ring buffer
  _input_buffer.Write(input, (int)num_frames);

  // Zero output before processing
  _output.leftCols(num_frames).setZero();

  const int numGroups = this->_num_groups;
  const long in_channels = get_in_channels();
  const long out_channels = get_out_channels();
  const long in_per_group = in_channels / numGroups;
  const long out_per_group = out_channels / numGroups;

  // Process from ring buffer with dilation lookback
  // After Write(), data is at positions [_write_pos, _write_pos+num_frames-1]
  // For kernel tap k with offset, we need to read from _write_pos + offset
  // The offset is negative (looking back), so _write_pos + offset reads from earlier positions
  // The original process_() reads: input.middleCols(i_start + offset, ncols)
  // where i_start is the current position and offset is negative for lookback

  if (numGroups == 1)
  {
    // Standard convolution (no grouping)
    for (size_t k = 0; k < this->_weight.size(); k++)
    {
      const long offset = this->_dilation * (k + 1 - (long)this->_weight.size());
      const long lookback = -offset;
      auto input_block = _input_buffer.Read((int)num_frames, lookback);
      _output.leftCols(num_frames).noalias() += this->_weight[k] * input_block;
    }
  }
  else
  {
    // Grouped convolution: process each group separately
    for (int g = 0; g < numGroups; g++)
    {
      for (size_t k = 0; k < this->_weight.size(); k++)
      {
        const long offset = this->_dilation * (k + 1 - (long)this->_weight.size());
        const long lookback = -offset;
        auto input_block = _input_buffer.Read((int)num_frames, lookback);

        // Extract input slice for this group
        auto input_group = input_block.middleRows(g * in_per_group, in_per_group);

        // Extract weight slice for this group
        auto weight_group = this->_weight[k].block(g * out_per_group, g * in_per_group, out_per_group, in_per_group);

        // Extract output slice for this group
        auto output_group = _output.leftCols(num_frames).middleRows(g * out_per_group, out_per_group);

        // Perform grouped convolution: output_group += weight_group * input_group
        output_group.noalias() += weight_group * input_group;
      }
    }
  }

  // Add bias if present
  if (this->_bias.size() > 0)
  {
    _output.leftCols(num_frames).colwise() += this->_bias;
  }

  // Advance ring buffer write pointer after processing
  _input_buffer.Advance((int)num_frames);
}

void Conv1D::process_(const Eigen::MatrixXf& input, Eigen::MatrixXf& output, const long i_start, const long ncols,
                      const long j_start) const
{
  const int numGroups = this->_num_groups;
  const long in_channels = get_in_channels();
  const long out_channels = get_out_channels();
  const long in_per_group = in_channels / numGroups;
  const long out_per_group = out_channels / numGroups;

  if (numGroups == 1)
  {
    // Standard convolution (no grouping)
    for (size_t k = 0; k < this->_weight.size(); k++)
    {
      const long offset = this->_dilation * (k + 1 - this->_weight.size());
      if (k == 0)
        output.middleCols(j_start, ncols).noalias() = this->_weight[k] * input.middleCols(i_start + offset, ncols);
      else
        output.middleCols(j_start, ncols).noalias() += this->_weight[k] * input.middleCols(i_start + offset, ncols);
    }
  }
  else
  {
    // Grouped convolution: process each group separately
    for (int g = 0; g < numGroups; g++)
    {
      for (size_t k = 0; k < this->_weight.size(); k++)
      {
        const long offset = this->_dilation * (k + 1 - this->_weight.size());

        // Extract input slice for this group
        auto input_group = input.middleCols(i_start + offset, ncols).middleRows(g * in_per_group, in_per_group);

        // Extract weight slice for this group
        auto weight_group = this->_weight[k].block(g * out_per_group, g * in_per_group, out_per_group, in_per_group);

        // Extract output slice for this group
        auto output_group = output.middleCols(j_start, ncols).middleRows(g * out_per_group, out_per_group);

        // Perform grouped convolution
        if (k == 0)
          output_group.noalias() = weight_group * input_group;
        else
          output_group.noalias() += weight_group * input_group;
      }
    }
  }
  if (this->_bias.size() > 0)
  {
    output.middleCols(j_start, ncols).colwise() += this->_bias;
  }
}

long Conv1D::get_num_weights() const
{
  long num_weights = this->_bias.size();
  if (this->_weight.size() > 0)
  {
    const long out_channels = this->_weight[0].rows();
    const long in_channels = this->_weight[0].cols();
    // For grouped convolutions, the number of weights is reduced by numGroups
    num_weights += (out_channels * in_channels * this->_weight.size()) / this->_num_groups;
  }
  return num_weights;
}
} // namespace nam
