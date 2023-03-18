//
//  ImpulseResponse.h
//  NeuralAmpModeler-macOS
//
//  Created by Steven Atkinson on 12/30/22.
//
// Impulse response processing

#ifndef ImpulseResponse_h
#define ImpulseResponse_h

#include <filesystem>

#include <Eigen/Dense>

#include "IPlugConstants.h" // sample
#include "dsp.h"
#include "wav.h"
#include "wdlstring.h" // WDL_String

namespace dsp {
class ImpulseResponse : public History {
public:
  ImpulseResponse(const WDL_String &fileName);
  ImpulseResponse(const std::vector<float> &rawAudio,
                  const double rawAudioSampleRate);
  dsp::wav::LoadReturnCode GetWavState() const { return this->mWavState; };
  iplug::sample **Process(iplug::sample **inputs, const size_t numChannels,
                          const size_t numFrames) override;
  void SetSampleRate(const double sampleRate);

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
  // Sample rate it was resampled to.
  double mSampleRate;

  const size_t mMaxLength = 8192;
  // The weights
  Eigen::VectorXf mWeight;
};
}; // namespace dsp

#endif /* ImpulseResponse_h */