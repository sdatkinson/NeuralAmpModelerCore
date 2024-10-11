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
  DSP(const double expected_sample_rate);
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
  virtual void process(NAM_SAMPLE* input, NAM_SAMPLE* output, const int num_frames);
  // Expected sample rate, in Hz.
  // TODO throw if it doesn't know.
  double GetExpectedSampleRate() const { return mExpectedSampleRate; };
  // Input Level, in dBu, corresponding to 0 dBFS for a sine wave
  // You should call HasInputLevel() first to be safe.
  double GetInputLevel() { return mInputLevel.level; };
  // Get how loud this model is, in dB.
  // Throws a std::runtime_error if the model doesn't know how loud it is.
  double GetLoudness() const;
  // Output Level, in dBu, corresponding to 0 dBFS for a sine wave
  // You should call HasOutputLevel() first to be safe.
  double GetOutputLevel() { return mOutputLevel.level; };
  // Does this model know its output level?
  bool HasInputLevel() { return mInputLevel.haveLevel; };
  // Get whether the model knows how loud it is.
  bool HasLoudness() const { return mHasLoudness; };
  // Does this model know its output level?
  bool HasOutputLevel() { return mOutputLevel.haveLevel; };
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
  void SetInputLevel(const double inputLevel)
  {
    mInputLevel.haveLevel = true;
    mInputLevel.level = inputLevel;
  };
  // Set the loudness, in dB.
  // This is usually defined to be the loudness to a standardized input. The trainer has its own, but you can always
  // use this to define it a different way if you like yours better.
  void SetLoudness(const double loudness);
  void SetOutputLevel(const double outputLevel)
  {
    mOutputLevel.haveLevel = true;
    mOutputLevel.level = outputLevel;
  };

protected:
  bool mHasLoudness = false;
  // How loud is the model? In dB
  double mLoudness = 0.0;
  // What sample rate does the model expect?
  double mExpectedSampleRate;
  // Have we been told what the external sample rate is? If so, what is it?
  bool mHaveExternalSampleRate = false;
  double mExternalSampleRate = -1.0;
  // The largest buffer I expect to be told to process:
  int mMaxBufferSize = 512;

  // How many samples should be processed for me to be considered "warmed up"?
  virtual int PrewarmSamples() { return 0; };

private:
  struct Level
  {
    bool haveLevel = false;
    float level = 0.0;
  };
  Level mInputLevel;
  Level mOutputLevel;
};

// Class where an input buffer is kept so that long-time effects can be
// captured. (e.g. conv nets or impulse responses, where we need history that's
// longer than the sample buffer that's coming in.)
class Buffer : public DSP
{
public:
  Buffer(const int receptive_field, const double expected_sample_rate = -1.0);

protected:
  // Input buffer
  const int _input_buffer_channels = 1; // Mono
  int _receptive_field;
  // First location where we add new samples from the input
  long _input_buffer_offset;
  std::vector<float> _input_buffer;
  std::vector<float> _output_buffer;

  void _advance_input_buffer_(const int num_frames);
  void _set_receptive_field(const int new_receptive_field, const int input_buffer_size);
  void _set_receptive_field(const int new_receptive_field);
  void _reset_input_buffer();
  // Use this->_input_post_gain
  virtual void _update_buffers_(NAM_SAMPLE* input, int num_frames);
  virtual void _rewind_buffers_();
};

// Basic linear model (an IR!)
class Linear : public Buffer
{
public:
  Linear(const int receptive_field, const bool _bias, const std::vector<float>& weights,
         const double expected_sample_rate = -1.0);
  void process(NAM_SAMPLE* input, NAM_SAMPLE* output, const int num_frames) override;

protected:
  Eigen::VectorXf _weight;
  float _bias;
};

// NN modules =================================================================

// TODO conv could take care of its own ring buffer.
class Conv1D
{
public:
  Conv1D() { this->_dilation = 1; };
  void set_weights_(std::vector<float>::iterator& weights);
  void set_size_(const int in_channels, const int out_channels, const int kernel_size, const bool do_bias,
                 const int _dilation);
  void set_size_and_weights_(const int in_channels, const int out_channels, const int kernel_size, const int _dilation,
                             const bool do_bias, std::vector<float>::iterator& weights);
  // Process from input to output
  //  Rightmost indices of input go from i_start for ncols,
  //  Indices on output for from j_start (to j_start + ncols - i_start)
  void process_(const Eigen::MatrixXf& input, Eigen::MatrixXf& output, const long i_start, const long ncols,
                const long j_start) const;
  long get_in_channels() const { return this->_weight.size() > 0 ? this->_weight[0].cols() : 0; };
  long get_kernel_size() const { return this->_weight.size(); };
  long get_num_weights() const;
  long get_out_channels() const { return this->_weight.size() > 0 ? this->_weight[0].rows() : 0; };
  int get_dilation() const { return this->_dilation; };

private:
  // Gonna wing this...
  // conv[kernel](cout, cin)
  std::vector<Eigen::MatrixXf> _weight;
  Eigen::VectorXf _bias;
  int _dilation;
};

// Really just a linear layer
class Conv1x1
{
public:
  Conv1x1(const int in_channels, const int out_channels, const bool _bias);
  void set_weights_(std::vector<float>::iterator& weights);
  // :param input: (N,Cin) or (Cin,)
  // :return: (N,Cout) or (Cout,), respectively
  Eigen::MatrixXf process(const Eigen::MatrixXf& input) const;

  long get_out_channels() const { return this->_weight.rows(); };

private:
  Eigen::MatrixXf _weight;
  Eigen::VectorXf _bias;
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
