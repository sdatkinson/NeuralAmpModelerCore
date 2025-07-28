#include "sequential.h"

#include <algorithm>
#include <stdexcept>

namespace nam
{

Sequential::Sequential(std::vector<std::unique_ptr<DSP>>&& models)
: DSP(models.empty() ? NAM_UNKNOWN_EXPECTED_SAMPLE_RATE : models[0]->GetExpectedSampleRate())
, mModels(std::move(models))
{
  ValidateModels();
  InitializeLevelsAndLoudness();

  // Now that we've validated the models, we know that the sample rates are the same.
  Reset(mModels.empty() ? NAM_UNKNOWN_EXPECTED_SAMPLE_RATE : mModels[0]->GetExpectedSampleRate(), 512);
}

void Sequential::ValidateModels() const
{
  if (!mModels.empty())
  {
    const double expectedSampleRate = mModels[0]->GetExpectedSampleRate();

    for (size_t i = 1; i < mModels.size(); ++i)
    {
      const double modelSampleRate = mModels[i]->GetExpectedSampleRate();
      if (std::abs(modelSampleRate - expectedSampleRate) > 1e-6 && modelSampleRate != NAM_UNKNOWN_EXPECTED_SAMPLE_RATE
          && expectedSampleRate != NAM_UNKNOWN_EXPECTED_SAMPLE_RATE)
      {
        throw std::invalid_argument("All models in Sequential must have the same expected sample rate");
      }
    }
  }
}

void Sequential::InitializeLevelsAndLoudness()
{
  if (mModels.empty())
  {
    return;
  }

  // Set input level from the first model
  if (mModels[0]->HasInputLevel())
  {
    SetInputLevel(mModels[0]->GetInputLevel());
  }

  // Set output level from the last model
  if (mModels.back()->HasOutputLevel())
  {
    SetOutputLevel(mModels.back()->GetOutputLevel());
  }

  // Set loudness from the last model
  // TODO: Implement a function to compute loudness for an arbitrary NAM::DSP instance
  // and use it to get the combined loudness of the entire sequence
  if (mModels.back()->HasLoudness())
  {
    SetLoudness(mModels.back()->GetLoudness());
  }
}

double dBToGain(const double dB)
{
  return std::pow(10.0, dB / 20.0);
}

void Sequential::process(NAM_SAMPLE* input, NAM_SAMPLE* output, const int num_frames)
{
  if (mModels.empty())
  {
    // No models, just copy input to output
    for (int i = 0; i < num_frames; ++i)
    {
      output[i] = input[i];
    }
    return;
  }

  // Ensure intermediate buffer is large enough
  if (mIntermediateBuffer.size() < static_cast<size_t>(num_frames))
  {
    throw std::runtime_error("Intermediate buffer is too small");
  }

  // Process through the first model
  mModels[0]->process(input, output, num_frames);

  // Process through remaining models, using intermediate buffer to avoid overwriting
  for (size_t i = 1; i < mModels.size(); ++i)
  {
    // Calibration gain:
    const double calibrationGain = mModels[i]->HasInputLevel() && mModels[i - 1]->HasOutputLevel()
                                     ? dBToGain(mModels[i - 1]->GetOutputLevel() - mModels[i]->GetInputLevel())
                                     : 1.0;

    // Copy current output to intermediate buffer
    for (int j = 0; j < num_frames; ++j)
    {
      mIntermediateBuffer[j] = output[j] * calibrationGain;
    }

    // Process through the next model
    mModels[i]->process(mIntermediateBuffer.data(), output, num_frames);
  }
}

void Sequential::Reset(const double sampleRate, const int maxBufferSize)
{
  // Reset all sub-models
  for (auto& model : mModels)
  {
    model->Reset(sampleRate, maxBufferSize);
  }

  // Call base class Reset first
  DSP::Reset(sampleRate, maxBufferSize);
}

void Sequential::SetMaxBufferSize(const int maxBufferSize)
{
  DSP::SetMaxBufferSize(maxBufferSize);
  for (auto& model : mModels)
  {
    model->SetMaxBufferSize(maxBufferSize);
  }
  mIntermediateBuffer.resize(maxBufferSize);
}

int Sequential::PrewarmSamples()
{
  int receptiveField = 1;
  for (const auto& model : mModels)
  {
    receptiveField += model->GetReceptiveField() - 1;
  }
  return receptiveField;
}

} // namespace nam