#pragma once


#include <iterator>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <Eigen/Dense>

namespace convnet
{
// Custom Conv that avoids re-computing on pieces of the input and trusts
// that the corresponding outputs are where they need to be.
// Beware: this is clever!

// Batch normalization
// In prod mode, so really just an elementwise affine layer.
class BatchNorm
{
public:
  BatchNorm(){};
  BatchNorm(const int dim, std::vector<float>::iterator& params);
  void process_(Eigen::MatrixXf& input, const long i_start, const long i_end) const;

private:
  // TODO simplify to just ax+b
  // y = (x-m)/sqrt(v+eps) * w + bias
  // y = ax+b
  // a = w / sqrt(v+eps)
  // b = a * m + bias
  Eigen::VectorXf scale;
  Eigen::VectorXf loc;
};

class ConvNetBlock
{
public:
  ConvNetBlock() { this->_batchnorm = false; };
  void set_params_(const int in_channels, const int out_channels, const int _dilation, const bool batchnorm,
                   const std::string activation, std::vector<float>::iterator& params);
  void process_(const Eigen::MatrixXf& input, Eigen::MatrixXf& output, const long i_start, const long i_end) const;
  long get_out_channels() const;
  Conv1D conv;

private:
  BatchNorm batchnorm;
  bool _batchnorm;
  activations::Activation* activation;
};

class _Head
{
public:
  _Head() { this->_bias = (float)0.0; };
  _Head(const int channels, std::vector<float>::iterator& params);
  void process_(const Eigen::MatrixXf& input, Eigen::VectorXf& output, const long i_start, const long i_end) const;

private:
  Eigen::VectorXf _weight;
  float _bias;
};

class ConvNet : public Buffer
{
public:
  ConvNet(const int channels, const std::vector<int>& dilations, const bool batchnorm, const std::string activation,
          std::vector<float>& params);
  ConvNet(const double loudness, const int channels, const std::vector<int>& dilations, const bool batchnorm,
          const std::string activation, std::vector<float>& params);

protected:
  std::vector<ConvNetBlock> _blocks;
  std::vector<Eigen::MatrixXf> _block_vals;
  Eigen::VectorXf _head_output;
  _Head _head;
  void _verify_params(const int channels, const std::vector<int>& dilations, const bool batchnorm,
                      const size_t actual_params);
  void _update_buffers_() override;
  void _rewind_buffers_() override;

  void _process_core_() override;

  // The net starts with random parameters inside; we need to wait for a full
  // receptive field to pass through before we can count on the output being
  // ok. This implements a gentle "ramp-up" so that there's no "pop" at the
  // start.
  long _anti_pop_countdown;
  const long _anti_pop_ramp = 100;
  void _anti_pop_();
  void _reset_anti_pop_();
};
}; // namespace convnet
