#pragma once

#include <Eigen/Dense>
#include <vector>
#include "ring_buffer.h"

namespace nam
{
class Conv1D
{
public:
  Conv1D() { this->_dilation = 1; };
  void set_weights_(std::vector<float>::iterator& weights);
  void set_size_(const int in_channels, const int out_channels, const int kernel_size, const bool do_bias,
                 const int _dilation);
  void set_size_and_weights_(const int in_channels, const int out_channels, const int kernel_size, const int _dilation,
                             const bool do_bias, std::vector<float>::iterator& weights);
  // Reset the ring buffer and pre-allocate output buffer
  // :param sampleRate: Unused, for interface consistency
  // :param maxBufferSize: Maximum buffer size for output buffer and to size ring buffer
  void Reset(const double sampleRate, const int maxBufferSize);
  // Get output buffer (similar to Conv1x1::GetOutput())
  // :param num_frames: Number of frames to return
  // :return: Block reference to output buffer
  Eigen::Block<Eigen::MatrixXf> GetOutput(const int num_frames);
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

protected:
  // conv[kernel](cout, cin)
  std::vector<Eigen::MatrixXf> _weight;
  Eigen::VectorXf _bias;
  int _dilation;

private:
  RingBuffer _input_buffer; // Ring buffer for input (channels x buffer_size)
  Eigen::MatrixXf _output; // Pre-allocated output buffer (out_channels x maxBufferSize)
  int _max_buffer_size = 0; // Stored maxBufferSize
};
} // namespace nam
