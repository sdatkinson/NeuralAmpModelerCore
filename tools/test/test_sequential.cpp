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
        : nam::DSP(sampleRate), mProcessCallCount(0), mPrewarmCallCount(0), mResetCallCount(0)
    {
        if (hasInputLevel) {
            SetInputLevel(inputLevel);
        }
        if (hasOutputLevel) {
            SetOutputLevel(outputLevel);
        }
        if (hasLoudness) {
            SetLoudness(loudness);
        }
    }

    void process(NAM_SAMPLE* input, NAM_SAMPLE* output, const int num_frames) override
    {
        mProcessCallCount++;
        // Simple gain processing for testing
        for (int i = 0; i < num_frames; i++) {
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

// Test: Cannot construct Sequential with empty vector
void test_construct_empty_throws()
{
    std::vector<std::unique_ptr<nam::DSP>> models; // Empty vector
    
    bool threw = false;
    try {
        nam::Sequential sequential(std::move(models));
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    
    assert(threw);
}

// Test: Cannot construct Sequential with mismatched sample rates
void test_construct_mismatched_sample_rates_throws()
{
    std::vector<std::unique_ptr<nam::DSP>> models;
    models.push_back(std::make_unique<MockDSP>(48000.0));
    models.push_back(std::make_unique<MockDSP>(44100.0)); // Different sample rate
    
    bool threw = false;
    try {
        nam::Sequential sequential(std::move(models));
    } catch (const std::invalid_argument&) {
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
    models.push_back(std::make_unique<MockDSP>(48000.0, true, false, false, 18.0, 0.0, 0.0)); // Has input level
    models.push_back(std::make_unique<MockDSP>(48000.0, true, false, false, 20.0, 0.0, 0.0)); // Different input level
    
    nam::Sequential sequential(std::move(models));
    
    assert(sequential.HasInputLevel());
    assert(sequential.GetInputLevel() == 18.0); // Should be from first model
}

// Test: Output level comes from last model
void test_output_level_from_last_model()
{
    std::vector<std::unique_ptr<nam::DSP>> models;
    models.push_back(std::make_unique<MockDSP>(48000.0, false, true, false, 0.0, 12.0, 0.0)); // Has output level
    models.push_back(std::make_unique<MockDSP>(48000.0, false, true, false, 0.0, 15.0, 0.0)); // Different output level
    
    nam::Sequential sequential(std::move(models));
    
    assert(sequential.HasOutputLevel());
    assert(sequential.GetOutputLevel() == 15.0); // Should be from last model
}

// Test: Loudness comes from last model
void test_loudness_from_last_model()
{
    std::vector<std::unique_ptr<nam::DSP>> models;
    models.push_back(std::make_unique<MockDSP>(48000.0, false, false, true, 0.0, 0.0, -20.0)); // Has loudness
    models.push_back(std::make_unique<MockDSP>(48000.0, false, false, true, 0.0, 0.0, -25.0)); // Different loudness
    
    nam::Sequential sequential(std::move(models));
    
    assert(sequential.HasLoudness());
    assert(sequential.GetLoudness() == -25.0); // Should be from last model
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
    std::vector<NAM_SAMPLE> input(numFrames, 1.0);
    std::vector<NAM_SAMPLE> output(numFrames);
    
    sequential.process(input.data(), output.data(), numFrames);
    
    assert(mock1_ptr->GetProcessCallCount() == 1);
    assert(mock2_ptr->GetProcessCallCount() == 1);
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
    
    assert(mock1_ptr->GetPrewarmCallCount() == 1);
    assert(mock2_ptr->GetPrewarmCallCount() == 1);
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
    
    assert(mock1_ptr->GetResetCallCount() == 1);
    assert(mock2_ptr->GetResetCallCount() == 1);
}

// Test: Single model works correctly
void test_single_model()
{
    std::vector<std::unique_ptr<nam::DSP>> models;
    auto mock = std::make_unique<MockDSP>(48000.0, true, true, true, 18.0, 12.0, -20.0);
    mock->SetGain(2.0);
    
    models.push_back(std::move(mock));
    
    nam::Sequential sequential(std::move(models));
    
    // Check properties are passed through
    assert(sequential.GetExpectedSampleRate() == 48000.0);
    assert(sequential.HasInputLevel() && sequential.GetInputLevel() == 18.0);
    assert(sequential.HasOutputLevel() && sequential.GetOutputLevel() == 12.0);
    assert(sequential.HasLoudness() && sequential.GetLoudness() == -20.0);
    
    // Check processing works
    const int numFrames = 1;
    NAM_SAMPLE input[1] = {1.0};
    NAM_SAMPLE output[1];
    
    sequential.process(input, output, numFrames);
    assert(output[0] == 2.0);
}

} // namespace test_sequential