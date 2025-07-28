// Tests for Sequential class

#include <cassert>
#include <memory>
#include <vector>
#include <stdexcept>

#include "NAM/sequential.h"
#include "NAM/dsp.h"

namespace test_sequential
{

// Mock DSP class for testing
class MockDSP : public nam::DSP
{
public:
  MockDSP(double sampleRate = 48000.0, bool hasInputLevel = false, bool hasOutputLevel = false,
          bool hasLoudness = false, double inputLevel = 0.0, double outputLevel = 0.0, double loudness = 0.0)
  : nam::DSP(sampleRate)
  , mProcessCallCount(0)
  , mPrewarmCallCount(0)
  , mResetCallCount(0)
  {
    if (hasInputLevel)
    {
      SetInputLevel(inputLevel);
    }
    if (hasOutputLevel)
    {
      SetOutputLevel(outputLevel);
    }
    if (hasLoudness)
    {
      SetLoudness(loudness);
    }
  }

  void process(NAM_SAMPLE* input, NAM_SAMPLE* output, const int num_frames) override
  {
    mProcessCallCount++;
    // Simple gain processing for testing
    for (int i = 0; i < num_frames; i++)
    {
      output[i] = input[i] * mGain;
    }
  }

  void prewarm() override
  {
    mPrewarmCallCount++;
    nam::DSP::prewarm();
  }

  void Reset(const double sampleRate, const int maxBufferSize) override
  {
    mResetCallCount++;
    nam::DSP::Reset(sampleRate, maxBufferSize);
  }

  // Test accessors
  int GetProcessCallCount() const { return mProcessCallCount; }
  int GetPrewarmCallCount() const { return mPrewarmCallCount; }
  int GetResetCallCount() const { return mResetCallCount; }
  void SetGain(double gain) { mGain = gain; }

protected:
  int PrewarmSamples() override { return 1; }

private:
  int mProcessCallCount;
  int mPrewarmCallCount;
  int mResetCallCount;
  double mGain = 1.0;
};

// Test: Can construct Sequential with valid models
void test_construct_with_models()
{
  std::vector<std::unique_ptr<nam::DSP>> models;
  models.push_back(std::make_unique<MockDSP>(48000.0));
  models.push_back(std::make_unique<MockDSP>(48000.0));

  nam::Sequential sequential(std::move(models));

  assert(sequential.GetExpectedSampleRate() == 48000.0);
}

// Test: Can construct Sequential with empty vector
void test_construct_empty()
{
  std::vector<std::unique_ptr<nam::DSP>> models; // Empty vector

  nam::Sequential sequential(std::move(models));

  assert(true);
}

// Test: Cannot construct Sequential with mismatched sample rates
void test_construct_mismatched_sample_rates_throws()
{
  std::vector<std::unique_ptr<nam::DSP>> models;
  models.push_back(std::make_unique<MockDSP>(48000.0));
  models.push_back(std::make_unique<MockDSP>(44100.0)); // Different sample rate

  bool threw = false;
  try
  {
    nam::Sequential sequential(std::move(models));
  }
  catch (const std::invalid_argument&)
  {
    threw = true;
  }

  assert(threw);
}

// Test: Can construct with unknown sample rates
void test_construct_with_unknown_sample_rates()
{
  std::vector<std::unique_ptr<nam::DSP>> models;
  models.push_back(std::make_unique<MockDSP>(NAM_UNKNOWN_EXPECTED_SAMPLE_RATE));
  models.push_back(std::make_unique<MockDSP>(48000.0));

  // Should not throw - unknown sample rates are allowed
  nam::Sequential sequential(std::move(models));

  assert(sequential.GetExpectedSampleRate() == NAM_UNKNOWN_EXPECTED_SAMPLE_RATE);
}

// Test: Input level comes from first model
void test_input_level_from_first_model()
{
  std::vector<std::unique_ptr<nam::DSP>> models;
  const double inputLevel1 = 18.0;
  const double inputLevel2 = inputLevel1 + 2.0;
  models.push_back(std::make_unique<MockDSP>(48000.0, true, false, false, inputLevel1, 0.0, 0.0)); // Has input level
  models.push_back(
    std::make_unique<MockDSP>(48000.0, true, false, false, inputLevel2, 0.0, 0.0)); // Different input level

  nam::Sequential sequential(std::move(models));

  assert(sequential.HasInputLevel());
  assert(sequential.GetInputLevel() == inputLevel1); // Should be from first model
}

// Test: Output level comes from last model
void test_output_level_from_last_model()
{
  std::vector<std::unique_ptr<nam::DSP>> models;
  const double outputLevel1 = 12.0;
  const double outputLevel2 = outputLevel1 + 3.0;
  models.push_back(std::make_unique<MockDSP>(48000.0, false, true, false, 0.0, outputLevel1, 0.0)); // Has output level
  models.push_back(
    std::make_unique<MockDSP>(48000.0, false, true, false, 0.0, outputLevel2, 0.0)); // Different output level

  nam::Sequential sequential(std::move(models));

  assert(sequential.HasOutputLevel());
  assert(sequential.GetOutputLevel() == outputLevel2); // Should be from last model
}

// Test: Loudness comes from last model
void test_loudness()
{
  std::vector<std::unique_ptr<nam::DSP>> models;
  const double loudness1 = -20.0;
  const double loudness2 = loudness1 + 5.0;
  models.push_back(std::make_unique<MockDSP>(48000.0, false, false, true, 0.0, 0.0, loudness1)); // Has loudness
  models.push_back(std::make_unique<MockDSP>(48000.0, false, false, true, 0.0, 0.0, loudness2)); // Different loudness

  nam::Sequential sequential(std::move(models));

  assert(sequential.HasLoudness());
  // It's not quite true that the loudness is simply that of the last model.
  // So this isn't actually correct, but it's what we have for now.
  // FIXME
  assert(sequential.GetLoudness() == loudness2);
}

// Test: No levels when models don't have them
void test_no_levels_when_models_dont_have_them()
{
  std::vector<std::unique_ptr<nam::DSP>> models;
  models.push_back(std::make_unique<MockDSP>(48000.0)); // No levels
  models.push_back(std::make_unique<MockDSP>(48000.0)); // No levels

  nam::Sequential sequential(std::move(models));

  assert(!sequential.HasInputLevel());
  assert(!sequential.HasOutputLevel());
  assert(!sequential.HasLoudness());
}

// Test: Process calls all models in sequence
void test_process_calls_all_models()
{
  std::vector<std::unique_ptr<nam::DSP>> models;
  auto mock1 = std::make_unique<MockDSP>(48000.0);
  auto mock2 = std::make_unique<MockDSP>(48000.0);

  // Keep pointers to check call counts
  MockDSP* mock1_ptr = mock1.get();
  MockDSP* mock2_ptr = mock2.get();

  models.push_back(std::move(mock1));
  models.push_back(std::move(mock2));

  nam::Sequential sequential(std::move(models));

  // Process some audio
  const int numFrames = 64;
  std::vector<NAM_SAMPLE> input(numFrames, 0.1);
  std::vector<NAM_SAMPLE> output(numFrames);

  sequential.process(input.data(), output.data(), numFrames);

  assert(mock1_ptr->GetProcessCallCount() > 0);
  assert(mock2_ptr->GetProcessCallCount() > 0);
}

// Test: Process chains models correctly
void test_process_chains_models()
{
  std::vector<std::unique_ptr<nam::DSP>> models;
  auto mock1 = std::make_unique<MockDSP>(48000.0);
  auto mock2 = std::make_unique<MockDSP>(48000.0);

  // Set different gains to test chaining
  mock1->SetGain(2.0);
  mock2->SetGain(3.0);

  models.push_back(std::move(mock1));
  models.push_back(std::move(mock2));

  nam::Sequential sequential(std::move(models));

  // Process audio
  const int numFrames = 1;
  NAM_SAMPLE input[1] = {1.0};
  NAM_SAMPLE output[1];

  sequential.process(input, output, numFrames);

  // Should be 1.0 * 2.0 * 3.0 = 6.0
  assert(output[0] == 6.0);
}

// Test: Prewarm calls all models
void test_prewarm_calls_all_models()
{
  std::vector<std::unique_ptr<nam::DSP>> models;
  auto mock1 = std::make_unique<MockDSP>(48000.0);
  auto mock2 = std::make_unique<MockDSP>(48000.0);

  // Keep pointers to check call counts
  MockDSP* mock1_ptr = mock1.get();
  MockDSP* mock2_ptr = mock2.get();

  models.push_back(std::move(mock1));
  models.push_back(std::move(mock2));

  nam::Sequential sequential(std::move(models));

  sequential.prewarm();

  assert(mock1_ptr->GetProcessCallCount() > 0);
  assert(mock2_ptr->GetProcessCallCount() > 0);
}

// Test: Reset calls all models
void test_reset_calls_all_models()
{
  std::vector<std::unique_ptr<nam::DSP>> models;
  auto mock1 = std::make_unique<MockDSP>(48000.0);
  auto mock2 = std::make_unique<MockDSP>(48000.0);

  // Keep pointers to check call counts
  MockDSP* mock1_ptr = mock1.get();
  MockDSP* mock2_ptr = mock2.get();

  models.push_back(std::move(mock1));
  models.push_back(std::move(mock2));

  nam::Sequential sequential(std::move(models));

  sequential.Reset(48000.0, 512);

  assert(mock1_ptr->GetResetCallCount() > 0);
  assert(mock2_ptr->GetResetCallCount() > 0);
}

// Test: Single model works correctly
void test_single_model()
{
  std::vector<std::unique_ptr<nam::DSP>> models;
  const double expectedSampleRate = 48000.0;
  const double inputLevel = 18.0;
  const double outputLevel = 12.0;
  const double loudness = -20.0;
  auto mock = std::make_unique<MockDSP>(expectedSampleRate, true, true, true, inputLevel, outputLevel, loudness);
  const double gain = 2.0;
  mock->SetGain(gain);

  models.push_back(std::move(mock));

  nam::Sequential sequential(std::move(models));

  // Check properties are passed through
  assert(sequential.GetExpectedSampleRate() == expectedSampleRate);
  assert(sequential.HasInputLevel() && sequential.GetInputLevel() == inputLevel);
  assert(sequential.HasOutputLevel() && sequential.GetOutputLevel() == outputLevel);
  assert(sequential.HasLoudness() && sequential.GetLoudness() == loudness);

  // Check processing works
  const int numFrames = 1;
  NAM_SAMPLE input[1] = {1.0};
  NAM_SAMPLE output[1];

  sequential.process(input, output, numFrames);
  assert(output[0] == gain);
}

} // namespace test_sequential