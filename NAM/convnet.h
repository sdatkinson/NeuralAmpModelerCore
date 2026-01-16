#pragma once

#include <filesystem>
#include <iterator>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <Eigen/Dense>

#include "conv1d.h"
#include "dsp.h"

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
  BatchNorm() {};
  BatchNorm(const int dim, std::vector<float>::iterator& weights);
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
  ConvNetBlock() {};
  void set_weights_(const int in_channels, const int out_channels, const int _dilation, const bool batchnorm,
                    const std::string activation, const int groups, std::vector<float>::iterator& weights);
  void SetMaxBufferSize(const int maxBufferSize);
  // Process input matrix directly (new API, similar to WaveNet)
  void Process(const Eigen::MatrixXf& input, const int num_frames);
  // Legacy method for compatibility (uses indices)
  void process_(const Eigen::MatrixXf& input, Eigen::MatrixXf& output, const long i_start, const long i_end);
  // Get output from last Process() call
  Eigen::Block<Eigen::MatrixXf> GetOutput(const int num_frames);
  long get_out_channels() const;
  Conv1D conv;

private:
  BatchNorm batchnorm;
  bool _batchnorm = false;
  activations::Activation* activation = nullptr;
  Eigen::MatrixXf _output; // Output buffer owned by the block
};

class _Head
{
public:
  _Head() {};
  _Head(const int in_channels, const int out_channels, std::vector<float>::iterator& weights);
  void process_(const Eigen::MatrixXf& input, Eigen::MatrixXf& output, const long i_start, const long i_end) const;

private:
  Eigen::MatrixXf _weight; // (out_channels, in_channels)
  Eigen::VectorXf _bias; // (out_channels,)
};

class ConvNet : public Buffer
{
public:
  ConvNet(const int in_channels, const int out_channels, const int channels, const std::vector<int>& dilations,
          const bool batchnorm, const std::string activation, std::vector<float>& weights,
          const double expected_sample_rate = -1.0, const int groups = 1);
  ~ConvNet() = default;

  void process(NAM_SAMPLE** input, NAM_SAMPLE** output, const int num_frames) override;
  void SetMaxBufferSize(const int maxBufferSize) override;

protected:
  std::vector<ConvNetBlock> _blocks;
  std::vector<Eigen::MatrixXf> _block_vals;
  Eigen::MatrixXf _head_output; // (out_channels, num_frames)
  _Head _head;
  void _verify_weights(const int channels, const std::vector<int>& dilations, const bool batchnorm,
                       const size_t actual_weights);
  void _update_buffers_(NAM_SAMPLE** input, const int num_frames) override;
  void _rewind_buffers_() override;

  int mPrewarmSamples = 0; // Pre-compute during initialization
  int PrewarmSamples() override { return mPrewarmSamples; };
};

// Factory
std::unique_ptr<DSP> Factory(const nlohmann::json& config, std::vector<float>& weights,
                             const double expectedSampleRate);

}; // namespace convnet
}; // namespace nam
