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

#ifdef NAM_SAMPLE_FLOAT
  #define NAM_SAMPLE float
#else
  #define NAM_SAMPLE double
#endif
// Use a sample rate of -1 if we don't know what the model expects to be run at.
// TODO clean this up and track a bool for whether it knows.
#define NAM_UNKNOWN_EXPECTED_SAMPLE_RATE -1.0

namespace nam
{
namespace wavenet
{
// Forward declaration to allow WaveNet to access protected members of DSP
// Not sure I like this.
class WaveNet;
} // namespace wavenet

enum EArchitectures
{
  kLinear = 0,
  kConvNet,
  kLSTM,
  kCatLSTM,
  kWaveNet,
  kCatWaveNet,
  kNumModels
};

class DSP
{
public:
  // Older models won't know, but newer ones will come with a loudness from the training based on their response to a
  // standardized input.
  // We may choose to have the models figure out for themselves how loud they are in here in the future.
  DSP(const int in_channels, const int out_channels, const double expected_sample_rate);
  virtual ~DSP() = default;
  // prewarm() does any required intial work required to "settle" model initial conditions
  // it can be somewhat expensive, so should not be called during realtime audio processing
  // Important: don't expect the model to be outputting zeroes after this. Neural networks
  // Don't know that there's anything special about "zero", and forcing this gets rid of
  // some possibilities that I dont' want to rule out (e.g. models that "are noisy").
  virtual void prewarm();
  // process() does all of the processing requried to take `input` array and
  // fill in the required values on `output`.
  // To do this:
  // 1. The core DSP algorithm is run (This is what should probably be
  //    overridden in subclasses).
  // 2. The output level is applied and the result stored to `output`.
  // `input` and `output` are double pointers where the first pointer indexes channels
  // and the second indexes frames: input[channel][frame]
  virtual void process(NAM_SAMPLE** input, NAM_SAMPLE** output, const int num_frames);
  // Expected sample rate, in Hz.
  // TODO throw if it doesn't know.
  double GetExpectedSampleRate() const { return mExpectedSampleRate; };
  // Number of input channels
  int NumInputChannels() const { return mInChannels; };
  // Number of output channels
  int NumOutputChannels() const { return mOutChannels; };
  // Input Level, in dBu, corresponding to 0 dBFS for a sine wave
  // You should call HasInputLevel() first to be safe.
  // Note: input level is assumed global over all inputs.
  double GetInputLevel();
  // Get how loud this model is, in dB.
  // Throws a std::runtime_error if the model doesn't know how loud it is.
  // Note: loudness is assumed global over all outputs.
  double GetLoudness() const;
  // Output Level, in dBu, corresponding to 0 dBFS for a sine wave
  // You should call HasOutputLevel() first to be safe.
  // Note: output level is assumed global over all outputs.
  double GetOutputLevel();
  // Does this model know its input level?
  // Note: input level is assumed global over all inputs.
  bool HasInputLevel();
  // Get whether the model knows how loud it is.
  bool HasLoudness() const { return mHasLoudness; };
  // Does this model know its output level?
  // Note: output level is assumed global over all outputs.
  bool HasOutputLevel();
  // General function for resetting the DSP unit.
  // This doesn't call prewarm(). If you want to do that, then you might want to use ResetAndPrewarm().
  // See https://github.com/sdatkinson/NeuralAmpModelerCore/issues/96 for the reasoning.
  virtual void Reset(const double sampleRate, const int maxBufferSize);
  // Reset(), then prewarm()
  void ResetAndPrewarm(const double sampleRate, const int maxBufferSize)
  {
    Reset(sampleRate, maxBufferSize);
    prewarm();
  }
  void SetInputLevel(const double inputLevel);
  // Set the loudness, in dB.
  // This is usually defined to be the loudness to a standardized input. The trainer has its own, but you can always
  // use this to define it a different way if you like yours better.
  // Note: loudness is assumed global over all outputs.
  void SetLoudness(const double loudness);
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

  // How many samples should be processed for me to be considered "warmed up"?
  virtual int PrewarmSamples() { return 0; };

  virtual void SetMaxBufferSize(const int maxBufferSize);
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

// Class where an input buffer is kept so that long-time effects can be
// captured. (e.g. conv nets or impulse responses, where we need history that's
// longer than the sample buffer that's coming in.)
class Buffer : public DSP
{
public:
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

// Basic linear model (an IR!)
class Linear : public Buffer
{
public:
  Linear(const int in_channels, const int out_channels, const int receptive_field, const bool _bias,
         const std::vector<float>& weights, const double expected_sample_rate = -1.0);
  void process(NAM_SAMPLE** input, NAM_SAMPLE** output, const int num_frames) override;

protected:
  Eigen::VectorXf _weight;
  float _bias;
};

namespace linear
{
std::unique_ptr<DSP> Factory(const nlohmann::json& config, std::vector<float>& weights,
                             const double expectedSampleRate);
} // namespace linear

// NN modules =================================================================

// Really just a linear layer
class Conv1x1
{
public:
  Conv1x1(const int in_channels, const int out_channels, const bool _bias, const int groups = 1);
  // Get the entire internal output buffer. This is intended for internal wiring
  // between layers/arrays; callers should treat the buffer as pre-allocated
  // storage and only consider the first `num_frames` columns valid for a given
  // processing call. Slice with .leftCols(num_frames) as needed.
  Eigen::MatrixXf& GetOutput() { return _output; }
  const Eigen::MatrixXf& GetOutput() const { return _output; }
  void SetMaxBufferSize(const int maxBufferSize);
  void set_weights_(std::vector<float>::iterator& weights);
  // :param input: (N,Cin) or (Cin,)
  // :return: (N,Cout) or (Cout,), respectively
  Eigen::MatrixXf process(const Eigen::MatrixXf& input) const { return process(input, (int)input.cols()); };
  Eigen::MatrixXf process(const Eigen::MatrixXf& input, const int num_frames) const;
  // Store output to pre-allocated _output; access with GetOutput()
  // Uses Eigen::Ref to accept matrices and block expressions without creating temporaries (real-time safe)
  void process_(const Eigen::Ref<const Eigen::MatrixXf>& input, const int num_frames);

  long get_out_channels() const { return this->_weight.rows(); };
  long get_in_channels() const { return this->_weight.cols(); };

protected:
  Eigen::MatrixXf _weight;
  Eigen::VectorXf _bias;
  int _num_groups;

private:
  Eigen::MatrixXf _output;
  bool _do_bias;
};

// Utilities ==================================================================
// Implemented in get_dsp.cpp

// Data for a DSP object
// :param version: Data version. Follows the conventions established in the trainer code.
// :param architecture: Defines the high-level architecture. Supported are (as per `get-dsp()` in get_dsp.cpp):
//     * "CatLSTM"
//     * "CatWaveNet"
//     * "ConvNet"
//     * "LSTM"
//     * "Linear"
//     * "WaveNet"
// :param config:
// :param metadata:
// :param weights: The model weights
// :param expected_sample_rate: Most NAM models implicitly assume that data will be provided to them at some sample
//     rate. This captures it for other components interfacing with the model to understand its needs. Use -1.0 for "I
//     don't know".
struct dspData
{
  std::string version;
  std::string architecture;
  nlohmann::json config;
  nlohmann::json metadata;
  std::vector<float> weights;
  double expected_sample_rate;
};

// Verify that the config that we are building our model from is supported by
// this plugin version.
void verify_config_version(const std::string version);

// Takes the model file and uses it to instantiate an instance of DSP.
std::unique_ptr<DSP> get_dsp(const std::filesystem::path model_file);
// Creates an instance of DSP. Also returns a dspData struct that holds the data of the model.
std::unique_ptr<DSP> get_dsp(const std::filesystem::path model_file, dspData& returnedConfig);
// Instantiates a DSP object from dsp_config struct.
std::unique_ptr<DSP> get_dsp(dspData& conf);
// Legacy loader for directory-type DSPs
std::unique_ptr<DSP> get_dsp_legacy(const std::filesystem::path dirname);
}; // namespace nam
