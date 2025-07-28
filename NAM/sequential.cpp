#include "sequential.h"

#include <algorithm>
#include <stdexcept>

namespace nam
{

Sequential::Sequential(std::vector<std::unique_ptr<DSP>>&& models)
: DSP(models.empty() ? NAM_UNKNOWN_EXPECTED_SAMPLE_RATE : models[0]->GetExpectedSampleRate())
, mModels(std::move(models))
{
  if (mModels.empty())
  {
    throw std::invalid_argument("Sequential model cannot be constructed with an empty vector of models");
  }

  ValidateModels();
  InitializeLevelsAndLoudness();
}

void Sequential::ValidateModels() const
{
  const double expectedSampleRate = mModels[0]->GetExpectedSampleRate();
  
  for (size_t i = 1; i < mModels.size(); ++i)
  {
    const double modelSampleRate = mModels[i]->GetExpectedSampleRate();
    if (std::abs(modelSampleRate - expectedSampleRate) > 1e-6 && 
        modelSampleRate != NAM_UNKNOWN_EXPECTED_SAMPLE_RATE &&
        expectedSampleRate != NAM_UNKNOWN_EXPECTED_SAMPLE_RATE)
    {
      throw std::invalid_argument("All models in Sequential must have the same expected sample rate");
    }
  }
}

void Sequential::InitializeLevelsAndLoudness()
{
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

int Sequential::ComputeMaxBufferSize() const
{
  int maxBufferSize = 0;
  for (const auto& model : mModels)
  {
    // Access the protected member through a friend-like approach or use a getter if available
    // For now, we'll use a reasonable default and let SetMaxBufferSize handle the propagation
    maxBufferSize = std::max(maxBufferSize, 4096); // Default buffer size from DSP::prewarm()
  }
  return maxBufferSize;
}

void Sequential::prewarm()
{
  // Prewarm all models in sequence
  for (auto& model : mModels)
  {
    model->prewarm();
  }
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
    mIntermediateBuffer.resize(num_frames);
  }

  // Process through the first model
  mModels[0]->process(input, output, num_frames);

  // Process through remaining models, using intermediate buffer to avoid overwriting
  for (size_t i = 1; i < mModels.size(); ++i)
  {
    // Copy current output to intermediate buffer
    for (int j = 0; j < num_frames; ++j)
    {
      mIntermediateBuffer[j] = output[j];
    }
    
    // Process through the next model
    mModels[i]->process(mIntermediateBuffer.data(), output, num_frames);
  }
}

void Sequential::Reset(const double sampleRate, const int maxBufferSize)
{
  // Call base class Reset first
  DSP::Reset(sampleRate, maxBufferSize);

  // Reset all sub-models
  for (auto& model : mModels)
  {
    model->Reset(sampleRate, maxBufferSize);
  }
}

int Sequential::PrewarmSamples()
{
  // Return the maximum prewarm samples needed by any model
  int maxPrewarmSamples = 0;
  for (const auto& model : mModels)
  {
    // We can't directly access the protected PrewarmSamples() method from other instances
    // For now, return a reasonable default. This could be improved with a public getter
    maxPrewarmSamples = std::max(maxPrewarmSamples, 4096);
  }
  return maxPrewarmSamples;
}

void Sequential::SetMaxBufferSize(const int maxBufferSize)
{
  // Call base class method
  DSP::SetMaxBufferSize(maxBufferSize);

  // Note: We don't call SetMaxBufferSize on sub-models here because it's protected.
  // Instead, this will be handled when Reset() is called on each sub-model,
  // which will in turn call SetMaxBufferSize on each one.

  // Ensure our intermediate buffer can handle the max buffer size
  mIntermediateBuffer.reserve(maxBufferSize);
}

} // namespace nam