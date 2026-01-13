#include "conv1d.h"

namespace nam
{
// Conv1D =====================================================================

void Conv1D::set_weights_(std::vector<float>::iterator& weights)
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

void Conv1D::set_size_and_weights_(const int in_channels, const int out_channels, const int kernel_size,
                                   const int _dilation, const bool do_bias, std::vector<float>::iterator& weights)
{
  this->set_size_(in_channels, out_channels, kernel_size, do_bias, _dilation);
  this->set_weights_(weights);
}

void Conv1D::Reset(const double sampleRate, const int maxBufferSize)
{
  (void)sampleRate; // Unused, but kept for interface consistency

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

Eigen::Block<Eigen::MatrixXf> Conv1D::GetOutput(const int num_frames)
{
  return _output.block(0, 0, _output.rows(), num_frames);
}

void Conv1D::Process(const Eigen::MatrixXf& input, const int num_frames)
{
  // Write input to ring buffer
  _input_buffer.Write(input, num_frames);

  // Zero output before processing
  _output.leftCols(num_frames).setZero();

  // Process from ring buffer with dilation lookback
  // After Write(), data is at positions [_write_pos, _write_pos+num_frames-1]
  // For kernel tap k with offset, we need to read from _write_pos + offset
  // The offset is negative (looking back), so _write_pos + offset reads from earlier positions
  // The original process_() reads: input.middleCols(i_start + offset, ncols)
  // where i_start is the current position and offset is negative for lookback
  for (size_t k = 0; k < this->_weight.size(); k++)
  {
    const long offset = this->_dilation * (k + 1 - (long)this->_weight.size());
    // Offset is negative (looking back)
    // Read from position: _write_pos + offset
    // Since offset is negative, we compute lookback = -offset to read from _write_pos - lookback
    const long lookback = -offset;

    // Read num_frames starting from write_pos + offset (which is write_pos - lookback)
    auto input_block = _input_buffer.Read(num_frames, lookback);

    // Perform convolution: output += weight[k] * input_block
    _output.leftCols(num_frames).noalias() += this->_weight[k] * input_block;
  }

  // Add bias if present
  if (this->_bias.size() > 0)
  {
    _output.leftCols(num_frames).colwise() += this->_bias;
  }

  // Advance ring buffer write pointer after processing
  _input_buffer.Advance(num_frames);
}

void Conv1D::process_(const Eigen::MatrixXf& input, Eigen::MatrixXf& output, const long i_start, const long ncols,
                      const long j_start) const
{
  // This is the clever part ;)
  for (size_t k = 0; k < this->_weight.size(); k++)
  {
    const long offset = this->_dilation * (k + 1 - this->_weight.size());
    if (k == 0)
      output.middleCols(j_start, ncols).noalias() = this->_weight[k] * input.middleCols(i_start + offset, ncols);
    else
      output.middleCols(j_start, ncols).noalias() += this->_weight[k] * input.middleCols(i_start + offset, ncols);
  }
  if (this->_bias.size() > 0)
  {
    output.middleCols(j_start, ncols).colwise() += this->_bias;
  }
}

long Conv1D::get_num_weights() const
{
  long num_weights = this->_bias.size();
  for (size_t i = 0; i < this->_weight.size(); i++)
    num_weights += this->_weight[i].size();
  return num_weights;
}
} // namespace nam
