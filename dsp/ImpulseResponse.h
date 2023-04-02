//
//  ImpulseResponse.h
//  NeuralAmpModeler-macOS
//
//  Created by Steven Atkinson on 12/30/22.
//
// Impulse response processing

#pragma once

#include <filesystem>

#include <Eigen/Dense>

#include "dsp.h"
#include "wav.h"

namespace dsp
{
class ImpulseResponse : public History
{
public:
  ImpulseResponse(const char* fileName, const double sampleRate);
  double** Process(double** inputs, const size_t numChannels, const size_t numFrames) override;
  // TODO states for the IR class
  dsp::wav::LoadReturnCode GetWavState() const { return this->mWavState; };

private:
  // Set the weights, given that the plugin is running at the provided sample
  // rate.
  void _SetWeights(const double sampleRate);

  // State of audio
  dsp::wav::LoadReturnCode mWavState;
  // Keep a copy of the raw audio that was loaded so that it can be resampled
  std::vector<float> mRawAudio;
  double mRawAudioSampleRate;
  // Resampled to the required sample rate.
  std::vector<float> mResampled;

  const size_t mMaxLength = 8192;
  // The weights
  Eigen::VectorXf mWeight;
};
}; // namespace dsp
