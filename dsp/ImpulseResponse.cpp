//
//  ImpulseResponse.cpp
//  NeuralAmpModeler-macOS
//
//  Created by Steven Atkinson on 12/30/22.
//

#include "Resample.h"
#include "wav.h"

#include "ImpulseResponse.h"

dsp::ImpulseResponse::ImpulseResponse(const WDL_String &fileName)
    : mSampleRate(0.0), mWavState(dsp::wav::LoadReturnCode::ERROR_OTHER) {
  this->mWavState =
      dsp::wav::Load(fileName, this->mRawAudio, this->mRawAudioSampleRate);
}

dsp::ImpulseResponse::ImpulseResponse(const std::vector<float> &rawAudio,
                                      const double rawAudioSampleRate)
    : mSampleRate(0.0), mWavState(dsp::wav::LoadReturnCode::ERROR_OTHER) {
  this->mRawAudio.resize(rawAudio.size());
  for (auto i = 0; i < rawAudio.size(); i++)
    this->mRawAudio[i] = rawAudio[i];
  this->mRawAudioSampleRate = rawAudioSampleRate;
  this->mWavState = dsp::wav::LoadReturnCode::SUCCESS;
}

iplug::sample **dsp::ImpulseResponse::Process(iplug::sample **inputs,
                                              const size_t numChannels,
                                              const size_t numFrames) {
  this->_PrepareBuffers(numChannels, numFrames);
  this->_UpdateHistory(inputs, numChannels, numFrames);

  for (size_t i = 0, j = this->mHistoryIndex - this->mHistoryRequired;
       i < numFrames; i++, j++) {
    auto input = Eigen::Map<const Eigen::VectorXf>(&this->mHistory[j],
                                                   this->mHistoryRequired + 1);
    this->mOutputs[0][i] = (double)this->mWeight.dot(input);
  }
  // Copy out for more-than-mono.
  for (size_t c = 1; c < numChannels; c++)
    for (size_t i = 0; i < numFrames; i++)
      this->mOutputs[c][i] = this->mOutputs[0][i];

  this->_AdvanceHistoryIndex(numFrames);
  return this->_GetPointers();
}

void dsp::ImpulseResponse::SetSampleRate(const double sampleRate) {
  if (sampleRate != this->mSampleRate)
    // Set the weights based on the raw audio.
    this->_SetWeights(sampleRate);
}

void dsp::ImpulseResponse::_SetWeights(const double sampleRate) {
  if (this->mRawAudioSampleRate == sampleRate) {
    // Simple implementation w/ no resample...
    this->mResampled.resize(this->mRawAudio.size());
    memcpy(this->mResampled.data(), this->mRawAudio.data(),
           this->mResampled.size());
  } else {
    // Cubic resampling
    std::vector<float> padded;
    padded.resize(this->mRawAudio.size() + 2);
    padded[0] = 0.0f;
    padded[padded.size() - 1] = 0.0f;
    memcpy(padded.data() + 1, this->mRawAudio.data(), this->mRawAudio.size());
    dsp::ResampleCubic<float>(padded, this->mRawAudioSampleRate, sampleRate,
                              0.0, this->mResampled);
  }
  const size_t irLength = std::min(this->mResampled.size(), this->mMaxLength);
  this->mWeight.resize(irLength);
  for (size_t i = 0, j = irLength - 1; i < irLength; i++, j--)
    this->mWeight[j] = this->mResampled[i];
  this->mHistoryRequired = irLength - 1;
  // And remember for the future
  this->mSampleRate = sampleRate;
}