#pragma once

#include <memory>
#include <vector>

#include "dsp.h"

namespace nam
{

// Sequential model that composes a set of DSP models and processes audio through them in sequence
class Sequential : public DSP
{
public:
  // Constructor takes ownership of the models via move semantics
  Sequential(std::vector<std::unique_ptr<DSP>>&& models);
  virtual ~Sequential() = default;

  // Override DSP interface methods
  void process(NAM_SAMPLE* input, NAM_SAMPLE* output, const int num_frames) override;
  void Reset(const double sampleRate, const int maxBufferSize) override;
  void SetMaxBufferSize(const int maxBufferSize) override;

protected:
  int PrewarmSamples() override;

private:
  std::vector<std::unique_ptr<DSP>> mModels;
  std::vector<NAM_SAMPLE> mIntermediateBuffer;

  // Helper methods
  void ValidateModels() const;
  void InitializeLevelsAndLoudness();
};

} // namespace nam