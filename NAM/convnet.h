#pragma once

#include <filesystem>
#include <iterator>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <Eigen/Dense>

namespace nam
{
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
  BatchNorm(const int dim, weights_it& weights);
  void process_(Eigen::Ref<Eigen::MatrixXf> input, const long i_start, const long i_end) const;

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
  ConvNetBlock(){};
  void set_weights_(const int in_channels, const int out_channels, const int _dilation, const bool batchnorm,
                    const std::string activation, weights_it& weights);
  void process_(const Eigen::Ref<const Eigen::MatrixXf> input, Eigen::Ref<Eigen::MatrixXf> output, const long i_start, const long i_end) const;
  long get_out_channels() const;
  Conv1D conv;

private:
  BatchNorm batchnorm;
  bool _batchnorm = false;
  activations::Activation* activation = nullptr;
};

class _Head
{
public:
  _Head(){};
  _Head(const int channels, weights_it& weights);
  void process_(const Eigen::Ref<const Eigen::MatrixXf> input, Eigen::VectorXf& output, const long i_start, const long i_end) const;

private:
  Eigen::VectorXf _weight;
  float mBias = 0.0f;
};

class ConvNet : public Buffer
{
public:
  ConvNet(const int channels, const std::vector<int>& dilations, const bool batchnorm, const std::string activation,
          std::vector<float>& weights, const double expectedSampleRate = -1.0);
  ~ConvNet() = default;

protected:
  std::vector<ConvNetBlock> _blocks;
  std::vector<Eigen::MatrixXf> _block_vals;
  Eigen::VectorXf _head_output;
  _Head _head;
  void _verify_weights(const int channels, const std::vector<int>& dilations, const bool batchnorm,
                       const size_t actual_weights);
  void UpdateBuffers(float* input, const int numFrames) override;
  void RewindBuffers() override;

  void Process(float* input, float* output, const int numFrames) override;
};
}; // namespace convnet
}; // namespace nam
