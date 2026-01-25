#pragma once

#include <filesystem>
#include <iterator>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <Eigen/Dense>
#include "index.h"

#include "activations.h"
#include "conv1d.h"
#include "dsp.h"
#include "json.hpp"

namespace nam
{
namespace convnet
{
/// \brief Batch normalization layer
///
/// In production mode, so really just an elementwise affine layer.
/// Applies: y = (x - mean) / sqrt(variance + eps) * weight + bias
/// which simplifies to: y = scale * x + loc
class BatchNorm
{
public:
  /// \brief Default constructor
  BatchNorm() {};

  /// \brief Constructor with weights
  /// \param dim Dimension of the input
  /// \param weights Iterator to the weights vector. Will be advanced as weights are consumed.
  BatchNorm(const int dim, std::vector<float>::iterator& weights);

  /// \brief Process input in-place
  /// \param input Input matrix to process
  /// \param i_start Start index
  /// \param i_end End index
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

/// \brief A single block in a ConvNet
///
/// Consists of a dilated convolution, optional batch normalization, and activation.
class ConvNetBlock
{
public:
  /// \brief Default constructor
  ConvNetBlock() {};

  /// \brief Set the parameters (weights) of this block
  /// \param in_channels Number of input channels
  /// \param out_channels Number of output channels
  /// \param _dilation Dilation factor for the convolution
  /// \param batchnorm Whether to use batch normalization
  /// \param activation_config Activation function configuration
  /// \param groups Number of groups for grouped convolution
  /// \param weights Iterator to the weights vector. Will be advanced as weights are consumed.
  void set_weights_(const int in_channels, const int out_channels, const int _dilation, const bool batchnorm,
                    const activations::ActivationConfig& activation_config, const int groups,
                    std::vector<float>::iterator& weights);

  /// \brief Resize buffers to handle maxBufferSize frames
  /// \param maxBufferSize Maximum number of frames to process in a single call
  void SetMaxBufferSize(const int maxBufferSize);

  /// \brief Process input matrix directly (new API, similar to WaveNet)
  /// \param input Input matrix (channels x num_frames)
  /// \param num_frames Number of frames to process
  void Process(const Eigen::MatrixXf& input, const nam::Index num_frames);

  /// \brief Process input (legacy method for compatibility, uses indices)
  /// \param input Input matrix
  /// \param output Output matrix
  /// \param i_start Start index in input
  /// \param i_end End index in input
  void process_(const Eigen::MatrixXf& input, Eigen::MatrixXf& output, const long i_start, const long i_end);

  /// \brief Get output from last Process() call
  /// \param num_frames Number of frames to return
  /// \return Block reference to the output
  Eigen::Block<Eigen::MatrixXf> GetOutput(const nam::Index num_frames);

  /// \brief Get the number of output channels
  /// \return Number of output channels
  long get_out_channels() const;

  Conv1D conv; ///< The dilated convolution layer

private:
  BatchNorm batchnorm;
  bool _batchnorm = false;
  activations::Activation::Ptr activation;
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

/// \brief Convolutional neural network model
///
/// A ConvNet consists of multiple ConvNetBlocks with increasing dilation factors,
/// followed by a head layer that produces the final output.
class ConvNet : public Buffer
{
public:
  /// \brief Constructor
  /// \param in_channels Number of input channels
  /// \param out_channels Number of output channels
  /// \param channels Number of channels in the hidden layers
  /// \param dilations Vector of dilation factors, one per block
  /// \param batchnorm Whether to use batch normalization
  /// \param activation_config Activation function configuration
  /// \param weights Model weights vector
  /// \param expected_sample_rate Expected sample rate in Hz (-1.0 if unknown)
  /// \param groups Number of groups for grouped convolution
  ConvNet(const int in_channels, const int out_channels, const int channels, const std::vector<int>& dilations,
          const bool batchnorm, const activations::ActivationConfig& activation_config, std::vector<float>& weights,
          const double expected_sample_rate = -1.0, const int groups = 1);

  /// \brief Destructor
  ~ConvNet() = default;

  /// \brief Process audio frames
  /// \param input Input audio buffers
  /// \param output Output audio buffers
  /// \param num_frames Number of frames to process
  void process(NAM_SAMPLE** input, NAM_SAMPLE** output, const int num_frames) override;

  /// \brief Resize all buffers to handle maxBufferSize frames
  /// \param maxBufferSize Maximum number of frames to process in a single call
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

/// \brief Factory function to instantiate ConvNet from JSON
/// \param config JSON configuration object
/// \param weights Model weights vector
/// \param expectedSampleRate Expected sample rate in Hz (-1.0 if unknown)
/// \return Unique pointer to a DSP object (ConvNet instance)
std::unique_ptr<DSP> Factory(const nlohmann::json& config, std::vector<float>& weights,
                             const double expectedSampleRate);

}; // namespace convnet
}; // namespace nam
