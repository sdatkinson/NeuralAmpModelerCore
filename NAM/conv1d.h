#pragma once

#include <Eigen/Dense>
#include <vector>
#include "ring_buffer.h"

namespace nam
{
class Conv1D
{
public:
  Conv1D()
  {
    this->_dilation = 1;
    this->_num_groups = 1;
  };
  Conv1D(const int in_channels, const int out_channels, const int kernel_size, const int bias, const int dilation,
         const int groups = 1)
  {
    set_size_(in_channels, out_channels, kernel_size, bias, dilation, groups);
  };
  void set_weights_(std::vector<float>::iterator& weights);
  void set_size_(const int in_channels, const int out_channels, const int kernel_size, const bool do_bias,
                 const int _dilation, const int groups = 1);
  void set_size_and_weights_(const int in_channels, const int out_channels, const int kernel_size, const int _dilation,
                             const bool do_bias, const int groups, std::vector<float>::iterator& weights);
  // Reset the ring buffer and pre-allocate output buffer
  // :param sampleRate: Unused, for interface consistency
  // :param maxBufferSize: Maximum buffer size for output buffer and to size ring buffer
  void SetMaxBufferSize(const int maxBufferSize);
  // Get the entire internal output buffer. This is intended for internal wiring
  // between layers; callers should treat the buffer as pre-allocated storage
  // and only consider the first `num_frames` columns valid for a given
  // processing call. Slice with .leftCols(num_frames) as needed.
  Eigen::MatrixXf& GetOutput() { return _output; }
  const Eigen::MatrixXf& GetOutput() const { return _output; }
  // Process input and write to internal output buffer
  // :param input: Input matrix (channels x num_frames)
  // :param num_frames: Number of frames to process
  void Process(const Eigen::MatrixXf& input, const int num_frames);
  // Process from input to output (legacy method, kept for compatibility)
  //  Rightmost indices of input go from i_start for ncols,
  //  Indices on output for from j_start (to j_start + ncols - i_start)
  void process_(const Eigen::MatrixXf& input, Eigen::MatrixXf& output, const long i_start, const long ncols,
                const long j_start) const;
  long get_in_channels() const { return this->_weight.size() > 0 ? this->_weight[0].cols() : 0; };
  long get_kernel_size() const { return this->_weight.size(); };
  long get_num_weights() const;
  long get_out_channels() const { return this->_weight.size() > 0 ? this->_weight[0].rows() : 0; };
  int get_dilation() const { return this->_dilation; };
  bool has_bias() const { return this->_bias.size() > 0; };

protected:
  // conv[kernel](cout, cin)
  std::vector<Eigen::MatrixXf> _weight;
  Eigen::VectorXf _bias;
  int _dilation;
  int _num_groups;

private:
  RingBuffer _input_buffer; // Ring buffer for input (channels x buffer_size)
  Eigen::MatrixXf _output; // Pre-allocated output buffer (out_channels x maxBufferSize)
  int _max_buffer_size = 0; // Stored maxBufferSize
};
} // namespace nam
