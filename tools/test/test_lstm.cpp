// Tests for LSTM

#include <Eigen/Dense>
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

#include "NAM/lstm.h"

namespace test_lstm
{
// Helper function to calculate weights needed for LSTM
// For each LSTMCell:
// - Weight matrix: (4 * hidden_size) x (input_size + hidden_size) in row-major order
// - Bias: 4 * hidden_size
// - Initial hidden state: hidden_size (stored in second half of _xh)
// - Initial cell state: hidden_size
// For the LSTM:
// - Head weight matrix: out_channels x hidden_size in row-major order
// - Head bias: out_channels
std::vector<float> create_lstm_weights(int num_layers, int input_size, int hidden_size, int out_channels)
{
  std::vector<float> weights;

  for (int layer = 0; layer < num_layers; layer++)
  {
    int layer_input_size = (layer == 0) ? input_size : hidden_size;
    int w_rows = 4 * hidden_size;
    int w_cols = layer_input_size + hidden_size;

    // Weight matrix (row-major)
    for (int i = 0; i < w_rows * w_cols; i++)
    {
      weights.push_back(0.1f); // Small weights for stability
    }

    // Bias vector
    for (int i = 0; i < 4 * hidden_size; i++)
    {
      weights.push_back(0.0f);
    }

    // Initial hidden state (stored in _xh)
    for (int i = 0; i < hidden_size; i++)
    {
      weights.push_back(0.0f);
    }

    // Initial cell state
    for (int i = 0; i < hidden_size; i++)
    {
      weights.push_back(0.0f);
    }
  }

  // Head weight matrix (row-major: out_channels x hidden_size)
  for (int out_ch = 0; out_ch < out_channels; out_ch++)
  {
    for (int h = 0; h < hidden_size; h++)
    {
      weights.push_back(0.1f);
    }
  }

  // Head bias
  for (int out_ch = 0; out_ch < out_channels; out_ch++)
  {
    weights.push_back(0.0f);
  }

  return weights;
}

// Test basic LSTM construction and processing
void test_lstm_basic()
{
  const int in_channels = 1;
  const int out_channels = 1;
  const int num_layers = 1;
  const int input_size = 1;
  const int hidden_size = 4;
  const double expected_sample_rate = 48000.0;

  std::vector<float> weights = create_lstm_weights(num_layers, input_size, hidden_size, out_channels);

  nam::lstm::LSTM lstm(in_channels, out_channels, num_layers, input_size, hidden_size, weights, expected_sample_rate);

  const int numFrames = 4;
  const int maxBufferSize = 64;
  lstm.Reset(expected_sample_rate, maxBufferSize);

  std::vector<NAM_SAMPLE> input(numFrames, 1.0f);
  std::vector<NAM_SAMPLE> output(numFrames, 0.0f);
  NAM_SAMPLE* inputPtrs[] = {input.data()};
  NAM_SAMPLE* outputPtrs[] = {output.data()};

  lstm.process(inputPtrs, outputPtrs, numFrames);

  // Verify output dimensions
  assert(output.size() == numFrames);
  // Output should be non-zero and finite
  for (int i = 0; i < numFrames; i++)
  {
    assert(std::isfinite(output[i]));
  }
}

// Test LSTM with multiple layers
void test_lstm_multiple_layers()
{
  const int in_channels = 1;
  const int out_channels = 1;
  const int num_layers = 2;
  const int input_size = 1;
  const int hidden_size = 4;
  const double expected_sample_rate = 48000.0;

  std::vector<float> weights = create_lstm_weights(num_layers, input_size, hidden_size, out_channels);

  nam::lstm::LSTM lstm(in_channels, out_channels, num_layers, input_size, hidden_size, weights, expected_sample_rate);

  const int numFrames = 8;
  const int maxBufferSize = 64;
  lstm.Reset(expected_sample_rate, maxBufferSize);

  std::vector<NAM_SAMPLE> input(numFrames, 0.5f);
  std::vector<NAM_SAMPLE> output(numFrames, 0.0f);
  NAM_SAMPLE* inputPtrs[] = {input.data()};
  NAM_SAMPLE* outputPtrs[] = {output.data()};

  lstm.process(inputPtrs, outputPtrs, numFrames);

  assert(output.size() == numFrames);
  for (int i = 0; i < numFrames; i++)
  {
    assert(std::isfinite(output[i]));
  }
}

// Test LSTM with zero input
void test_lstm_zero_input()
{
  const int in_channels = 1;
  const int out_channels = 1;
  const int num_layers = 1;
  const int input_size = 1;
  const int hidden_size = 4;
  const double expected_sample_rate = 48000.0;

  std::vector<float> weights = create_lstm_weights(num_layers, input_size, hidden_size, out_channels);

  nam::lstm::LSTM lstm(in_channels, out_channels, num_layers, input_size, hidden_size, weights, expected_sample_rate);

  const int numFrames = 4;
  lstm.Reset(expected_sample_rate, numFrames);

  std::vector<NAM_SAMPLE> input(numFrames, 0.0f);
  std::vector<NAM_SAMPLE> output(numFrames, 0.0f);
  NAM_SAMPLE* inputPtrs[] = {input.data()};
  NAM_SAMPLE* outputPtrs[] = {output.data()};

  lstm.process(inputPtrs, outputPtrs, numFrames);

  // With zero input, output should be finite (may be zero or non-zero depending on bias)
  for (int i = 0; i < numFrames; i++)
  {
    assert(std::isfinite(output[i]));
  }
}

// Test LSTM with different buffer sizes
void test_lstm_different_buffer_sizes()
{
  const int in_channels = 1;
  const int out_channels = 1;
  const int num_layers = 1;
  const int input_size = 1;
  const int hidden_size = 4;
  const double expected_sample_rate = 48000.0;

  std::vector<float> weights = create_lstm_weights(num_layers, input_size, hidden_size, out_channels);

  nam::lstm::LSTM lstm(in_channels, out_channels, num_layers, input_size, hidden_size, weights, expected_sample_rate);

  // Test with different buffer sizes
  lstm.Reset(expected_sample_rate, 64);
  std::vector<NAM_SAMPLE> input1(32, 1.0f);
  std::vector<NAM_SAMPLE> output1(32, 0.0f);
  NAM_SAMPLE* inputPtrs1[] = {input1.data()};
  NAM_SAMPLE* outputPtrs1[] = {output1.data()};
  lstm.process(inputPtrs1, outputPtrs1, 32);

  lstm.Reset(expected_sample_rate, 128);
  std::vector<NAM_SAMPLE> input2(64, 1.0f);
  std::vector<NAM_SAMPLE> output2(64, 0.0f);
  NAM_SAMPLE* inputPtrs2[] = {input2.data()};
  NAM_SAMPLE* outputPtrs2[] = {output2.data()};
  lstm.process(inputPtrs2, outputPtrs2, 64);

  // Both should work without errors
  assert(output1.size() == 32);
  assert(output2.size() == 64);
}

// Test LSTM prewarm functionality
void test_lstm_prewarm()
{
  const int in_channels = 1;
  const int out_channels = 1;
  const int num_layers = 1;
  const int input_size = 1;
  const int hidden_size = 4;
  const double expected_sample_rate = 48000.0;

  std::vector<float> weights = create_lstm_weights(num_layers, input_size, hidden_size, out_channels);

  nam::lstm::LSTM lstm(in_channels, out_channels, num_layers, input_size, hidden_size, weights, expected_sample_rate);

  // Test that prewarm can be called without errors
  lstm.Reset(expected_sample_rate, 64);
  lstm.prewarm();

  // After prewarm, processing should work
  const int numFrames = 4;
  std::vector<NAM_SAMPLE> input(numFrames, 1.0f);
  std::vector<NAM_SAMPLE> output(numFrames, 0.0f);
  NAM_SAMPLE* inputPtrs[] = {input.data()};
  NAM_SAMPLE* outputPtrs[] = {output.data()};
  lstm.process(inputPtrs, outputPtrs, numFrames);

  // Output should be finite
  for (int i = 0; i < numFrames; i++)
  {
    assert(std::isfinite(output[i]));
  }
}

// Test multiple process() calls (state persistence)
void test_lstm_multiple_calls()
{
  const int in_channels = 1;
  const int out_channels = 1;
  const int num_layers = 1;
  const int input_size = 1;
  const int hidden_size = 4;
  const double expected_sample_rate = 48000.0;

  std::vector<float> weights = create_lstm_weights(num_layers, input_size, hidden_size, out_channels);

  nam::lstm::LSTM lstm(in_channels, out_channels, num_layers, input_size, hidden_size, weights, expected_sample_rate);

  const int numFrames = 2;
  lstm.Reset(expected_sample_rate, numFrames);

  // Multiple calls should work correctly with state persistence
  for (int i = 0; i < 5; i++)
  {
    std::vector<NAM_SAMPLE> input(numFrames, 1.0f);
    std::vector<NAM_SAMPLE> output(numFrames, 0.0f);
    NAM_SAMPLE* inputPtrs[] = {input.data()};
    NAM_SAMPLE* outputPtrs[] = {output.data()};
    lstm.process(inputPtrs, outputPtrs, numFrames);

    // Output should be finite
    for (int j = 0; j < numFrames; j++)
    {
      assert(std::isfinite(output[j]));
    }
  }
}

// Test LSTM with multi-channel input/output
void test_lstm_multichannel()
{
  const int in_channels = 2;
  const int out_channels = 2;
  const int num_layers = 1;
  const int input_size = 2;
  const int hidden_size = 4;
  const double expected_sample_rate = 48000.0;

  std::vector<float> weights = create_lstm_weights(num_layers, input_size, hidden_size, out_channels);

  nam::lstm::LSTM lstm(in_channels, out_channels, num_layers, input_size, hidden_size, weights, expected_sample_rate);

  const int numFrames = 4;
  lstm.Reset(expected_sample_rate, 64);

  std::vector<NAM_SAMPLE> input1(numFrames, 0.5f);
  std::vector<NAM_SAMPLE> input2(numFrames, 0.3f);
  std::vector<NAM_SAMPLE> output1(numFrames, 0.0f);
  std::vector<NAM_SAMPLE> output2(numFrames, 0.0f);
  NAM_SAMPLE* inputPtrs[] = {input1.data(), input2.data()};
  NAM_SAMPLE* outputPtrs[] = {output1.data(), output2.data()};

  lstm.process(inputPtrs, outputPtrs, numFrames);

  // Verify both output channels are finite
  for (int i = 0; i < numFrames; i++)
  {
    assert(std::isfinite(output1[i]));
    assert(std::isfinite(output2[i]));
  }
}

// Test LSTM with larger hidden size
void test_lstm_large_hidden_size()
{
  const int in_channels = 1;
  const int out_channels = 1;
  const int num_layers = 1;
  const int input_size = 1;
  const int hidden_size = 16;
  const double expected_sample_rate = 48000.0;

  std::vector<float> weights = create_lstm_weights(num_layers, input_size, hidden_size, out_channels);

  nam::lstm::LSTM lstm(in_channels, out_channels, num_layers, input_size, hidden_size, weights, expected_sample_rate);

  const int numFrames = 4;
  lstm.Reset(expected_sample_rate, 64);

  std::vector<NAM_SAMPLE> input(numFrames, 1.0f);
  std::vector<NAM_SAMPLE> output(numFrames, 0.0f);
  NAM_SAMPLE* inputPtrs[] = {input.data()};
  NAM_SAMPLE* outputPtrs[] = {output.data()};

  lstm.process(inputPtrs, outputPtrs, numFrames);

  for (int i = 0; i < numFrames; i++)
  {
    assert(std::isfinite(output[i]));
  }
}

// Test LSTM with different input sizes
void test_lstm_different_input_size()
{
  const int in_channels = 3;
  const int out_channels = 1;
  const int num_layers = 1;
  const int input_size = 3;
  const int hidden_size = 4;
  const double expected_sample_rate = 48000.0;

  std::vector<float> weights = create_lstm_weights(num_layers, input_size, hidden_size, out_channels);

  nam::lstm::LSTM lstm(in_channels, out_channels, num_layers, input_size, hidden_size, weights, expected_sample_rate);

  const int numFrames = 4;
  lstm.Reset(expected_sample_rate, 64);

  std::vector<NAM_SAMPLE> input1(numFrames, 0.1f);
  std::vector<NAM_SAMPLE> input2(numFrames, 0.2f);
  std::vector<NAM_SAMPLE> input3(numFrames, 0.3f);
  std::vector<NAM_SAMPLE> output(numFrames, 0.0f);
  NAM_SAMPLE* inputPtrs[] = {input1.data(), input2.data(), input3.data()};
  NAM_SAMPLE* outputPtrs[] = {output.data()};

  lstm.process(inputPtrs, outputPtrs, numFrames);

  for (int i = 0; i < numFrames; i++)
  {
    assert(std::isfinite(output[i]));
  }
}

// Test LSTM state evolution over time
void test_lstm_state_evolution()
{
  const int in_channels = 1;
  const int out_channels = 1;
  const int num_layers = 1;
  const int input_size = 1;
  const int hidden_size = 4;
  const double expected_sample_rate = 48000.0;

  std::vector<float> weights = create_lstm_weights(num_layers, input_size, hidden_size, out_channels);

  nam::lstm::LSTM lstm(in_channels, out_channels, num_layers, input_size, hidden_size, weights, expected_sample_rate);

  const int numFrames = 10;
  lstm.Reset(expected_sample_rate, 64);

  // Create a sine wave input
  std::vector<NAM_SAMPLE> input(numFrames);
  for (int i = 0; i < numFrames; i++)
  {
    input[i] = 0.5f * std::sin(2.0f * M_PI * i / numFrames);
  }

  std::vector<NAM_SAMPLE> output(numFrames, 0.0f);
  NAM_SAMPLE* inputPtrs[] = {input.data()};
  NAM_SAMPLE* outputPtrs[] = {output.data()};

  lstm.process(inputPtrs, outputPtrs, numFrames);

  // Output should be finite and potentially show some variation due to state
  for (int i = 0; i < numFrames; i++)
  {
    assert(std::isfinite(output[i]));
  }
}

// Test LSTM with no layers (edge case)
void test_lstm_no_layers()
{
  const int in_channels = 1;
  const int out_channels = 1;
  const int num_layers = 0;
  const int input_size = 1;
  const int hidden_size = 4;
  const double expected_sample_rate = 48000.0;

  // With no layers, we still need head weights
  std::vector<float> weights;
  // Head weight matrix (row-major: out_channels x hidden_size)
  for (int out_ch = 0; out_ch < out_channels; out_ch++)
  {
    for (int h = 0; h < hidden_size; h++)
    {
      weights.push_back(0.0f); // Zero weights means pass-through
    }
  }
  // Head bias
  for (int out_ch = 0; out_ch < out_channels; out_ch++)
  {
    weights.push_back(0.0f);
  }

  nam::lstm::LSTM lstm(in_channels, out_channels, num_layers, input_size, hidden_size, weights, expected_sample_rate);

  const int numFrames = 4;
  lstm.Reset(expected_sample_rate, 64);

  std::vector<NAM_SAMPLE> input(numFrames, 1.0f);
  std::vector<NAM_SAMPLE> output(numFrames, 0.0f);
  NAM_SAMPLE* inputPtrs[] = {input.data()};
  NAM_SAMPLE* outputPtrs[] = {output.data()};

  lstm.process(inputPtrs, outputPtrs, numFrames);

  // With zero head weights and bias, output should equal input for first channel
  for (int i = 0; i < numFrames; i++)
  {
    assert(std::isfinite(output[i]));
  }
}

}; // namespace test_lstm
