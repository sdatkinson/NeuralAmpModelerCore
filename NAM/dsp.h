#pragma once

#include <filesystem>
#include <iterator>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <Eigen/Dense>

#include "activations.h"
#include "json.hpp"
#include "model_config.h"

#ifdef NAM_SAMPLE_FLOAT
  #define NAM_SAMPLE float
#else
  #define NAM_SAMPLE double
#endif
/// \brief Use a sample rate of -1 if we don't know what the model expects to be run at
#define NAM_UNKNOWN_EXPECTED_SAMPLE_RATE -1.0

namespace nam
{
namespace wavenet
{
/// Forward declaration to allow WaveNet to access protected members of DSP
class WaveNet;
} // namespace wavenet


/// \brief Base class for all DSP models
///
/// DSP provides the common interface for all neural network-based audio processing models.
/// It handles:
/// - Input/output channel management
/// - Sample rate tracking
/// - Level management (input/output levels and loudness)
/// - Prewarm functionality for settling initial conditions
/// - Buffer size management
///
/// Subclasses should override process() to implement the actual processing algorithm.
class DSP
{
public:
  /// \brief Constructor
  ///
  /// \param in_channels Number of input channels
  /// \param out_channels Number of output channels
  /// \param expected_sample_rate Expected sample rate in Hz (-1.0 if unknown)
  DSP(const int in_channels, const int out_channels, const double expected_sample_rate);

  /// \brief Virtual destructor
  virtual ~DSP() = default;

  /// \brief Prewarm the model to settle initial conditions
  ///
  /// This can be somewhat expensive, so should not be called during real-time audio processing.
  /// Important: don't expect the model to be outputting zeroes after this. Neural networks
  /// don't know that there's anything special about "zero", and forcing this gets rid of
  /// some possibilities (e.g. models that "are noisy").
  virtual void prewarm();

  /// \brief Process audio frames
  ///
  /// \param input Input audio buffers. Double pointer where the first pointer indexes channels
  ///              and the second indexes frames: input[channel][frame]
  /// \param output Output audio buffers. Same structure as input.
  /// \param num_frames Number of frames to process
  virtual void process(NAM_SAMPLE** input, NAM_SAMPLE** output, const int num_frames);
  /// \brief Get the expected sample rate
  /// \return Expected sample rate in Hz (-1.0 if unknown)
  double GetExpectedSampleRate() const { return mExpectedSampleRate; };

  /// \brief Get the number of input channels
  /// \return Number of input channels
  int NumInputChannels() const { return mInChannels; };

  /// \brief Get the number of output channels
  /// \return Number of output channels
  int NumOutputChannels() const { return mOutChannels; };

  /// \brief Get the input level
  ///
  /// Input level is in dBu RMS, corresponding to 0 dBFS peak for a 1 kHz sine wave.
  /// You should call HasInputLevel() first to be safe.
  /// Note: input level is assumed global over all inputs.
  /// \return Input level in dBu
  double GetInputLevel();

  /// \brief Get how loud this model's output is, in dB, if a "typical" input is processed.
  /// This can be used to normalize the output level of the object.
  ///
  /// Throws a std::runtime_error if the model doesn't know how loud it is.
  /// Note: loudness is assumed global over all outputs.
  /// \return Loudness in dB
  /// \throws std::runtime_error If the model doesn't know its loudness
  double GetLoudness() const;

  /// \brief Get the output level
  ///
  /// Output level is in dBu RMS, corresponding to 0 dBFS peak for a 1 kHz sine wave.
  /// You should call HasOutputLevel() first to be safe.
  /// Note: output level is assumed global over all outputs.
  /// \return Output level in dBu
  double GetOutputLevel();

  /// \brief Check if this model knows its input level
  ///
  /// Note: input level is assumed global over all inputs.
  /// \return true if input level is known, false otherwise
  bool HasInputLevel();

  /// \brief Check if the model knows how loud it is
  /// \return true if loudness is known, false otherwise
  bool HasLoudness() const { return mHasLoudness; };

  /// \brief Check if this model knows its output level
  ///
  /// Note: output level is assumed global over all outputs.
  /// \return true if output level is known, false otherwise
  bool HasOutputLevel();

  /// \brief General function for resetting the DSP unit
  ///
  /// This doesn't call prewarm(). If you want to do that, then you might want to use ResetAndPrewarm().
  /// See https://github.com/sdatkinson/NeuralAmpModelerCore/issues/96 for the reasoning.
  /// \param sampleRate Current sample rate
  /// \param maxBufferSize Maximum buffer size to process
  virtual void Reset(const double sampleRate, const int maxBufferSize);

  /// \brief Reset the DSP unit, then prewarm
  /// \param sampleRate Current sample rate
  /// \param maxBufferSize Maximum buffer size to process
  void ResetAndPrewarm(const double sampleRate, const int maxBufferSize)
  {
    Reset(sampleRate, maxBufferSize);
    prewarm();
  }

  /// \brief Set the input level
  /// \param inputLevel Input level in dBu
  void SetInputLevel(const double inputLevel);

  /// \brief Set the loudness
  ///
  /// This is usually defined to be the loudness to a standardized input. The trainer has its own,
  /// but you can always use this to define it a different way if you like yours better.
  /// Note: loudness is assumed global over all outputs.
  /// \param loudness Loudness in dB
  void SetLoudness(const double loudness);

  /// \brief Set the output level
  /// \param outputLevel Output level in dBu
  void SetOutputLevel(const double outputLevel);

protected:
  friend class wavenet::WaveNet; // Allow WaveNet to access protected members. Used in condition DSP.

  bool mHasLoudness = false;
  // How loud is the model? In dB
  double mLoudness = 0.0;
  // What sample rate does the model expect?
  double mExpectedSampleRate;
  // Have we been told what the external sample rate is? If so, what is it?
  bool mHaveExternalSampleRate = false;
  double mExternalSampleRate = -1.0;
  // The largest buffer I expect to be told to process:
  int mMaxBufferSize = 0;

  /// \brief Get how many samples should be processed for the model to be considered "warmed up"
  ///
  /// Override this in subclasses to specify prewarm requirements.
  /// \return Number of samples needed for prewarm
  virtual int PrewarmSamples() { return 0; };

  /// \brief Set the maximum buffer size
  /// \param maxBufferSize Maximum number of frames to process in a single call
  virtual void SetMaxBufferSize(const int maxBufferSize);

  /// \brief Get the maximum buffer size
  /// \return Maximum buffer size
  int GetMaxBufferSize() const { return mMaxBufferSize; };

private:
  const int mInChannels;
  const int mOutChannels;
  struct Level
  {
    bool haveLevel = false;
    float level = 0.0;
  };
  // Note: input/output levels are assumed global over all inputs/outputs
  Level mInputLevel;
  Level mOutputLevel;
};

/// \brief Base class for DSP models that require input buffering
/// This class is deprecated and will be removed in a future version.
///
/// Class where an input buffer is kept so that long-time effects can be captured.
/// (e.g. conv nets or impulse responses, where we need history that's longer than
/// the sample buffer that's coming in.)
class Buffer : public DSP
{
public:
  /// \brief Constructor
  /// \param in_channels Number of input channels
  /// \param out_channels Number of output channels
  /// \param receptive_field Size of the receptive field (buffer size needed)
  /// \param expected_sample_rate Expected sample rate in Hz (-1.0 if unknown)
  Buffer(const int in_channels, const int out_channels, const int receptive_field,
         const double expected_sample_rate = -1.0);

protected:
  int _receptive_field;
  // First location where we add new samples from the input (same for all channels)
  long _input_buffer_offset;
  // Per-channel input buffers
  std::vector<std::vector<float>> _input_buffers;
  std::vector<std::vector<float>> _output_buffers;

  void _advance_input_buffer_(const int num_frames);
  void _set_receptive_field(const int new_receptive_field, const int input_buffer_size);
  void _set_receptive_field(const int new_receptive_field);
  void _reset_input_buffer();
  // Use this->_input_post_gain
  virtual void _update_buffers_(NAM_SAMPLE** input, int num_frames);
  virtual void _rewind_buffers_();
};

/// \brief Basic linear model
///
/// Implements a simple linear convolution, (i.e. an impulse response).
class Linear : public Buffer
{
public:
  /// \brief Constructor
  /// \param in_channels Number of input channels
  /// \param out_channels Number of output channels
  /// \param receptive_field Size of the impulse response
  /// \param _bias Whether to use bias
  /// \param weights Model weights (impulse response coefficients)
  /// \param expected_sample_rate Expected sample rate in Hz (-1.0 if unknown)
  Linear(const int in_channels, const int out_channels, const int receptive_field, const bool _bias,
         const std::vector<float>& weights, const double expected_sample_rate = -1.0);

  /// \brief Process audio frames
  /// \param input Input audio buffers
  /// \param output Output audio buffers
  /// \param num_frames Number of frames to process
  void process(NAM_SAMPLE** input, NAM_SAMPLE** output, const int num_frames) override;

protected:
  Eigen::VectorXf _weight;
  float _bias;
};

namespace linear
{

/// \brief Configuration for a Linear model
struct LinearConfig : public ModelConfig
{
  int receptive_field;
  bool bias;
  int in_channels;
  int out_channels;

  std::unique_ptr<DSP> create(std::vector<float> weights, double sampleRate) override;
};

/// \brief Parse Linear configuration from JSON
/// \param config JSON configuration object
/// \return LinearConfig
LinearConfig parse_config_json(const nlohmann::json& config);

/// \brief Config parser for ConfigParserRegistry
/// \param config JSON configuration object
/// \param sampleRate Expected sample rate in Hz
/// \return unique_ptr<ModelConfig> wrapping a LinearConfig
std::unique_ptr<ModelConfig> create_config(const nlohmann::json& config, double sampleRate);
} // namespace linear

// NN modules =================================================================

/// \brief 1x1 convolution (really just a fully-connected linear layer operating per-sample)
///
/// Performs a pointwise convolution, which is equivalent to a fully connected layer
/// applied independently to each time step. Supports grouped convolution for efficiency.
class Conv1x1
{
public:
  /// \brief Constructor
  /// \param in_channels Number of input channels
  /// \param out_channels Number of output channels
  /// \param _bias Whether to use bias
  /// \param groups Number of groups for grouped convolution (default: 1)
  Conv1x1(const int in_channels, const int out_channels, const bool _bias, const int groups = 1);

  /// \brief Get the entire internal output buffer
  ///
  /// This is intended for internal wiring between layers/arrays; callers should treat
  /// the buffer as pre-allocated storage and only consider the first num_frames columns
  /// valid for a given processing call. Slice with .leftCols(num_frames) as needed.
  /// \return Reference to the output buffer
  Eigen::MatrixXf& GetOutput() { return _output; }

  /// \brief Get the entire internal output buffer (const version)
  /// \return Const reference to the output buffer
  const Eigen::MatrixXf& GetOutput() const { return _output; }

  /// \brief Resize the output buffer to handle maxBufferSize frames
  /// \param maxBufferSize Maximum number of frames to process in a single call
  void SetMaxBufferSize(const int maxBufferSize);

  /// \brief Set the parameters (weights) of this module
  /// \param weights Iterator to the weights vector. Will be advanced as weights are consumed.
  void set_weights_(std::vector<float>::iterator& weights);

  /// \brief Process input and return output matrix
  ///
  /// \param input Input matrix (channels x num_frames) or (channels,)
  /// \return Output matrix (channels x num_frames) or (channels,), respectively
  Eigen::MatrixXf process(const Eigen::MatrixXf& input) const { return process(input, (int)input.cols()); };

  /// \brief Process input and return output matrix
  /// \param input Input matrix (channels x num_frames)
  /// \param num_frames Number of frames to process
  /// \return Output matrix (channels x num_frames)
  Eigen::MatrixXf process(const Eigen::MatrixXf& input, const int num_frames) const;

  /// \brief Process input and store output to pre-allocated buffer
  ///
  /// Uses Eigen::Ref to accept matrices and block expressions without creating
  /// temporaries (real-time safe). Access output via GetOutput().
  /// \param input Input matrix (channels x num_frames)
  /// \param num_frames Number of frames to process
  void process_(const Eigen::Ref<const Eigen::MatrixXf>& input, const int num_frames);

  long get_out_channels() const;
  long get_in_channels() const;

protected:
  // Non-depthwise: full weight matrix (out_channels x in_channels)
  Eigen::MatrixXf _weight;
  // For depthwise convolution (groups == in_channels == out_channels):
  // stores one weight per channel
  Eigen::VectorXf _depthwise_weight;
  bool _is_depthwise = false;
  int _channels = 0; // Used for depthwise case (in_channels == out_channels)
  Eigen::VectorXf _bias;
  int _num_groups;

private:
  Eigen::MatrixXf _output;
  bool _do_bias;
};

// Utilities ==================================================================
// Implemented in get_dsp.cpp

/// \brief Data structure for a DSP object
///
/// Contains all information needed to instantiate and configure a DSP model.
struct dspData
{
  std::string version; ///< Data version. Follows conventions established in trainer code.
  std::string architecture; ///< High-level architecture. Supported: "ConvNet", "LSTM", "Linear", "WaveNet"
  nlohmann::json config; ///< Model configuration JSON
  nlohmann::json metadata; ///< Model metadata JSON
  std::vector<float> weights; ///< Model weights
  double expected_sample_rate; ///< Expected sample rate in Hz. Most NAM models implicitly assume data at some sample
                               ///< rate. Use -1.0 for "I don't know".
};

/// \brief Verify that the config version is supported by this plugin version
/// \param version Config version string to verify
void verify_config_version(const std::string version);

/// \brief Legacy loader for directory-style DSPs
///
/// Loads models from a directory structure (older format).
/// \param dirname Path to the directory containing the model
/// \return Unique pointer to a DSP object
std::unique_ptr<DSP> get_dsp_legacy(const std::filesystem::path dirname);
}; // namespace nam
