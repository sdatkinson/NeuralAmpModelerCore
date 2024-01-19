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

// Use a sample rate of -1 if we don't know what the model expects to be run at.
// TODO clean this up and track a bool for whether it knows.
#define NAM_UNKNOWN_EXPECTED_SAMPLE_RATE -1.0

namespace nam
{
using weights_it = std::vector<float>::const_iterator;

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
  DSP(const double expectedSampleRate);
  virtual ~DSP() = default;
  // Prewarm() does any required intial work required to "settle" model initial conditions
  // it can be somewhat expensive, so should not be called during realtime audio processing
  virtual void Prewarm();
  // Process() does all of the processing requried to take `input` array and
  // fill in the required values on `output`.
  // To do this:
  // 1. The core DSP algorithm is run (This is what should probably be
  //    overridden in subclasses).
  // 2. The output level is applied and the result stored to `output`.
  virtual void Process(float* input, float* output, const int numFrames);
  // Anything to take care of before next buffer comes in.
  // For example:
  // * Move the buffer index forward
  virtual void Finalize(const int numFrames);
  // Expected sample rate, in Hz.
  // TODO throw if it doesn't know.
  double GetExpectedSampleRate() const { return mExpectedSampleRate; };
  // Get how loud this model is, in dB.
  // Throws a std::runtime_error if the model doesn't know how loud it is.
  double GetLoudness() const;
  // Get whether the model knows how loud it is.
  bool HasLoudness() const { return mHasLoudness; };
  // Set the loudness, in dB.
  // This is usually defined to be the loudness to a standardized input. The trainer has its own, but you can always
  // use this to define it a different way if you like yours better.
  void SetLoudness(const double loudness);

protected:
  bool mHasLoudness = false;
  // How loud is the model? In dB
  double mLoudness = 0.0;
  // What sample rate does the model expect?
  double mExpectedSampleRate;
  // How many samples should be processed during "pre-warming"
  int mPrewarmSamples = 0;
};

// Class where an input buffer is kept so that long-time effects can be
// captured. (e.g. conv nets or impulse responses, where we need history that's
// longer than the sample buffer that's coming in.)
class Buffer : public DSP
{
public:
  Buffer(const int receptiveField, const double expectedSampleRate = -1.0);
  void Finalize(const int numFrames);

protected:
  int mReceptiveField;
  // First location where we add new samples from the input
  long mInputBufferOffset;
  std::vector<float> mInputBuffer;
  std::vector<float> mOutputBuffer;

  void SetReceptiveField(const int newReceptiveField, const int inputBufferSize);
  void SetReceptiveField(const int newReceptiveField);
  void ResetInputBuffer();
  // Use this->_input_post_gain
  virtual void UpdateBuffers(float* input, int numFrames);
  virtual void RewindBuffers();
};

// Basic linear model (an IR!)
class Linear : public Buffer
{
public:
  Linear(const int receptiveField, const bool bias, const std::vector<float>& weights,
         const double expectedSampleRate = -1.0);
  void Process(float* input, float* output, const int numFrames) override;

protected:
  Eigen::VectorXf mWeight;
  float mBias;
};

// NN modules =================================================================

class Conv1D
{
public:
  Conv1D() { this->mDilation = 1; };
  void SetWeights(weights_it& weights);
  void SetSize(const int inChannels, const int outChannels, const int kernelSize, const bool doBias,
                 const int dilation);
  void SetSizeAndWeights(const int inChannels, const int outChannels, const int kernelSize, const int dilation,
                             const bool doBias, weights_it& weights);
  // Process from input to output
  //  Rightmost indices of input go from i_start to i_end,
  //  Indices on output for from j_start (to j_start + i_end - i_start)
  void Process(const Eigen::Ref<const Eigen::MatrixXf> input, Eigen::Ref<Eigen::MatrixXf> output, const long i_start, const long i_end,
                const long j_start) const;
  long GetInChannels() const { return this->mWeight.size() > 0 ? this->mWeight[0].cols() : 0; };
  long get_kernel_size() const { return this->mWeight.size(); };
  long get_num_weights() const;
  long get_out_channels() const { return this->mWeight.size() > 0 ? this->mWeight[0].rows() : 0; };
  int get_dilation() const { return this->mDilation; };

private:
  // Gonna wing this...
  // conv[kernel](cout, cin)
  std::vector<Eigen::MatrixXf> mWeight;
  Eigen::VectorXf mBias;
  int mDilation;
};

// Really just a linear layer
class Conv1x1
{
public:
  Conv1x1(const int inChannels, const int outChannels, const bool bias);
  void SetWeights(weights_it& weights);
  // :param input: (N,Cin) or (Cin,)
  // :return: (N,Cout) or (Cout,), respectively
  Eigen::MatrixXf Process(const Eigen::Ref<const Eigen::MatrixXf> input) const;

  long get_out_channels() const { return this->mWeight.rows(); };

private:
  Eigen::MatrixXf mWeight;
  Eigen::VectorXf mBias;
  bool mDoBias;
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
// :param expectedSampleRate: Most NAM models implicitly assume that data will be provided to them at some sample
//     rate. This captures it for other components interfacing with the model to understand its needs. Use -1.0 for "I
//     don't know".
struct dspData
{
  std::string version;
  std::string architecture;
  nlohmann::json config;
  nlohmann::json metadata;
  std::vector<float> weights;
  double expectedSampleRate;
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
