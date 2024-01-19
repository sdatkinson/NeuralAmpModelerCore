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
  void Process(Eigen::Ref<Eigen::MatrixXf> input, const long i_start, const long i_end) const;

private:
  // TODO simplify to just ax+b
  // y = (x-m)/sqrt(v+eps) * w + bias
  // y = ax+b
  // a = w / sqrt(v+eps)
  // b = a * m + bias
  Eigen::VectorXf mScale;
  Eigen::VectorXf loc;
};

class ConvNetBlock
{
public:
  ConvNetBlock(){};
  void SetWeights(const int inChannels, const int outChannels, const int dilation, const bool batchnorm,
                    const std::string activation, weights_it& weights);
  void Process(const Eigen::Ref<const Eigen::MatrixXf> input, Eigen::Ref<Eigen::MatrixXf> output, const long i_start, const long i_end) const;
  long GetOutChannels() const;
  Conv1D conv;

private:
  BatchNorm mBatchnorm;
  bool mDoBatchNorm = false;
  activations::Activation* activation = nullptr;
};

class Head
{
public:
  Head(){};
  Head(const int channels, weights_it& weights);
  void Process(const Eigen::Ref<const Eigen::MatrixXf> input, Eigen::VectorXf& output, const long i_start, const long i_end) const;

private:
  Eigen::VectorXf mWeight;
  float mBias = 0.0f;
};

class ConvNet : public Buffer
{
public:
  ConvNet(const int channels, const std::vector<int>& dilations, const bool batchnorm, const std::string activation,
          std::vector<float>& weights, const double expectedSampleRate = -1.0);
  ~ConvNet() = default;

protected:
  std::vector<ConvNetBlock> mBlocks;
  std::vector<Eigen::MatrixXf> mBlockVals;
  Eigen::VectorXf mHeadOutput;
  Head mHead;
  void VerifyWeights(const int channels, const std::vector<int>& dilations, const bool batchnorm,
                       const size_t actualWeights);
  void UpdateBuffers(float* input, const int numFrames) override;
  void RewindBuffers() override;

  void Process(float* input, float* output, const int numFrames) override;
};
}; // namespace convnet
}; // namespace nam
