#pragma once

#include <Eigen/Dense>
#include <vector>
#include "ring_buffer.h"

namespace nam
{
/// \brief 1D dilated convolution layer
///
/// Implements a 1D convolution with support for dilation and grouped convolution.
/// Uses a ring buffer to maintain input history for efficient processing of
/// sequential audio frames.
class Conv1D
{
public:
  /// \brief Default constructor
  ///
  /// Initializes with dilation=1 and groups=1. Use set_size_() to configure.
  Conv1D()
  {
    this->_dilation = 1;
    this->_num_groups = 1;
  };

  /// \brief Constructor
  /// \param in_channels Number of input channels
  /// \param out_channels Number of output channels
  /// \param kernel_size Size of the convolution kernel
  /// \param bias Whether to use bias (1 for true, 0 for false)
  /// \param dilation Dilation factor for the convolution
  /// \param groups Number of groups for grouped convolution (default: 1)
  Conv1D(const int in_channels, const int out_channels, const int kernel_size, const int bias, const int dilation,
         const int groups = 1)
  {
    set_size_(in_channels, out_channels, kernel_size, bias, dilation, groups);
  };

  /// \brief Set the parameters (weights) of this module
  /// \param weights Iterator to the weights vector. Will be advanced as weights are consumed.
  void set_weights_(std::vector<float>::iterator& weights);

  /// \brief Set the size parameters of the convolution
  /// \param in_channels Number of input channels
  /// \param out_channels Number of output channels
  /// \param kernel_size Size of the convolution kernel
  /// \param do_bias Whether to use bias
  /// \param _dilation Dilation factor for the convolution
  /// \param groups Number of groups for grouped convolution
  void set_size_(const int in_channels, const int out_channels, const int kernel_size, const bool do_bias,
                 const int _dilation, const int groups = 1);

  /// \brief Set size and weights in one call
  /// \param in_channels Number of input channels
  /// \param out_channels Number of output channels
  /// \param kernel_size Size of the convolution kernel
  /// \param _dilation Dilation factor for the convolution
  /// \param do_bias Whether to use bias
  /// \param groups Number of groups for grouped convolution
  /// \param weights Iterator to the weights vector. Will be advanced as weights are consumed.
  void set_size_and_weights_(const int in_channels, const int out_channels, const int kernel_size, const int _dilation,
                             const bool do_bias, const int groups, std::vector<float>::iterator& weights);

  /// \brief Reset the ring buffer and pre-allocate output buffer
  /// \param maxBufferSize Maximum buffer size for output buffer and to size ring buffer
  void SetMaxBufferSize(const int maxBufferSize);

  /// \brief Get the entire internal output buffer
  ///
  /// This is intended for internal wiring between layers; callers should treat
  /// the buffer as pre-allocated storage and only consider the first num_frames columns
  /// valid for a given processing call. Slice with .leftCols(num_frames) as needed.
  /// \return Reference to the output buffer
  Eigen::MatrixXf& GetOutput() { return _output; }

  /// \brief Get the entire internal output buffer (const version)
  /// \return Const reference to the output buffer
  const Eigen::MatrixXf& GetOutput() const { return _output; }

  /// \brief Process input and write to internal output buffer
  /// \param input Input matrix (channels x num_frames)
  /// \param num_frames Number of frames to process
  void Process(const Eigen::MatrixXf& input, const int num_frames);

  /// \brief Process from input to output (legacy method, kept for compatibility)
  ///
  /// Rightmost indices of input go from i_start for ncols,
  /// Indices on output go from j_start (to j_start + ncols - i_start).
  /// \param input Input matrix
  /// \param output Output matrix
  /// \param i_start Starting index in input
  /// \param ncols Number of columns to process
  /// \param j_start Starting index in output
  void process_(const Eigen::MatrixXf& input, Eigen::MatrixXf& output, const long i_start, const long ncols,
                const long j_start) const;
  /// \brief Get the number of input channels
  /// \return Number of input channels
  long get_in_channels() const { return this->_weight.size() > 0 ? this->_weight[0].cols() : 0; };

  /// \brief Get the kernel size
  /// \return Kernel size
  long get_kernel_size() const { return this->_weight.size(); };

  /// \brief Get the total number of weights
  /// \return Total number of weight parameters
  long get_num_weights() const;

  /// \brief Get the number of output channels
  /// \return Number of output channels
  long get_out_channels() const { return this->_weight.size() > 0 ? this->_weight[0].rows() : 0; };

  /// \brief Get the dilation factor
  /// \return Dilation factor
  int get_dilation() const { return this->_dilation; };

  /// \brief Check if bias is used
  /// \return true if bias is present, false otherwise
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
