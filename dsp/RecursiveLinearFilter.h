//
//  RecursiveLinearFilter.h
//
//
//  Created by Steven Atkinson on 12/28/22.
//
// Recursive linear filters (LPF, HPF, Peaking, Shelving)

#pragma once

#include "dsp.h"
#include <cmath> // pow, sin
#include <vector>

#define MATH_PI 3.14159265358979323846

// TODO refactor base DSP into a common abstraction.

namespace recursive_linear_filter
{
class Base : public dsp::DSP
{
public:
  Base(const size_t inputDegree, const size_t outputDegree);
  DSP_SAMPLE** Process(DSP_SAMPLE** inputs, const size_t numChannels, const size_t numFrames) override;

protected:
  // Methods
  size_t _GetInputDegree() const { return this->mInputCoefficients.size(); };
  size_t _GetOutputDegree() const { return this->mOutputCoefficients.size(); };
  // Additionally prepares mInputHistory and mOutputHistory.
  void _PrepareBuffers(const size_t numChannels, const size_t numFrames) override;

  // Coefficients for the DSP filter
  // [0] is for the current sample
  // [1] is for the previous
  // [2] before that
  // (mOutputCoefficients[0] should always be zero. It'll never be used.)
  std::vector<double> mInputCoefficients;
  std::vector<double> mOutputCoefficients;

  // Arrays holding the history on which the filter depends recursively.
  // First index is channel
  // Second index, [0] is the current input/output, [1] is the previous, [2] is
  // before that, etc.
  std::vector<std::vector<DSP_SAMPLE>> mInputHistory;
  std::vector<std::vector<DSP_SAMPLE>> mOutputHistory;
  // Indices for history.
  // Designates which index is currently "0". Use modulus to wrap around.
  long mInputStart;
  long mOutputStart;
};

class LevelParams : public dsp::Params
{
public:
  LevelParams(const double gain)
  : Params()
  , mGain(gain){};
  double GetGain() const { return this->mGain; };

private:
  // The gain (multiplicative, i.e. not dB)
  double mGain;
};

class Level : public Base
{
public:
  Level()
  : Base(1, 0){};
  // Invalid usage: require a pointer to recursive_linear_filter::Params so
  // that SetCoefficients() is defined.
  void SetParams(const LevelParams& params) { this->mInputCoefficients[0] = params.GetGain(); };
  ;
};

// The same 3 params (frequency, quality, gain) describe a bunch of filters.
// (Low shelf, high shelf, peaking)
class BiquadParams : public dsp::Params
{
public:
  BiquadParams(const double sampleRate, const double frequency, const double quality, const double gainDB)
  : dsp::Params()
  , mFrequency(frequency)
  , mGainDB(gainDB)
  , mQuality(quality)
  , mSampleRate(sampleRate){};

  // Parameters defined in
  // https://webaudio.github.io/Audio-EQ-Cookbook/audio-eq-cookbook.html
  double GetA() const { return pow(10.0, this->mGainDB / 40.0); };
  double GetOmega0() const { return 2.0 * MATH_PI * this->mFrequency / this->mSampleRate; };
  double GetAlpha(const double omega_0) const { return sin(omega_0) / (2.0 * this->mQuality); };
  double GetCosW(const double omega_0) const { return cos(omega_0); };

private:
  double mFrequency;
  double mGainDB;
  double mQuality;
  double mSampleRate;
};

class Biquad : public Base
{
public:
  Biquad()
  : Base(3, 3){};
  virtual void SetParams(const BiquadParams& params) = 0;

protected:
  void _AssignCoefficients(const double a0, const double a1, const double a2, const double b0, const double b1,
                           const double b2);
};

class LowShelf : public Biquad
{
public:
  void SetParams(const BiquadParams& params) override;
};

class Peaking : public Biquad
{
public:
  void SetParams(const BiquadParams& params) override;
};

class HighShelf : public Biquad
{
public:
  void SetParams(const BiquadParams& params) override;
};

// HPF only has one param: frequency
// TODO LPF (alpha calculation is different though)
class HighPassParams : public dsp::Params
{
public:
  HighPassParams(const double sampleRate, const double frequency)
  : dsp::Params()
  , mFrequency(frequency)
  , mSampleRate(sampleRate){};

  double GetAlpha() const
  {
    const double c = 2.0 * MATH_PI * mFrequency / mSampleRate;
    return 1.0 / (c + 1.0);
  };

private:
  double mFrequency;
  double mSampleRate;
};

class HighPass : public Base
{
public:
  HighPass()
  : Base(2, 2){};
  void SetParams(const HighPassParams& params)
  {
    const double alpha = params.GetAlpha();
    // y[i] = alpha * y[i-1] + alpha * (x[i]-x[i-1])
    mInputCoefficients[0] = alpha;
    mInputCoefficients[1] = -alpha;
    mOutputCoefficients[0] = 0.0;
    mOutputCoefficients[1] = alpha;
  }
};

class LowPassParams : public dsp::Params
{
public:
  LowPassParams(const double sampleRate, const double frequency)
  : dsp::Params()
  , mFrequency(frequency)
  , mSampleRate(sampleRate){};

  double GetAlpha() const
  {
    const double c = 2.0 * MATH_PI * mFrequency / mSampleRate;
    return c / (c + 1.0);
  };

private:
  double mFrequency;
  double mSampleRate;
};

class LowPass : public Base
{
public:
  LowPass()
  : Base(1, 2){};
  void SetParams(const LowPassParams& params)
  {
    const double alpha = params.GetAlpha();
    // y[i] = alpha * x[i] + (1-alpha) * y[i-1]
    mInputCoefficients[0] = alpha;
    mOutputCoefficients[0] = 0.0;
    mOutputCoefficients[1] = 1.0 - alpha;
  }
};

}; // namespace recursive_linear_filter
