// Tests for dsp

#include "NAM/dsp.h"
#include <vector>

namespace test_dsp
{
// Simplest test: can I construct something!
void test_construct()
{
  const int in_channels = 1;
  const int out_channels = 1;
  nam::DSP myDsp(in_channels, out_channels, 48000.0);
}

void test_channels()
{
  const int in_channels = 2;
  const int out_channels = 3;
  nam::DSP myDsp(in_channels, out_channels, 48000.0);
  assert(myDsp.NumInputChannels() == in_channels);
  assert(myDsp.NumOutputChannels() == out_channels);
}

void test_get_input_level()
{
  const int in_channels = 2;
  const int out_channels = 1;
  nam::DSP myDsp(in_channels, out_channels, 48000.0);
  const double expected = 19.0;
  myDsp.SetInputLevel(expected);
  assert(myDsp.HasInputLevel());
  const double actual = myDsp.GetInputLevel();

  assert(actual == expected);
}

void test_get_output_level()
{
  const int in_channels = 1;
  const int out_channels = 2;
  nam::DSP myDsp(in_channels, out_channels, 48000.0);
  const double expected = 12.0;
  myDsp.SetOutputLevel(expected);
  assert(myDsp.HasOutputLevel());
  const double actual = myDsp.GetOutputLevel();

  assert(actual == expected);
}

// Test correct function of DSP::HasInputLevel()
void test_has_input_level()
{
  const int in_channels = 2;
  const int out_channels = 1;
  nam::DSP myDsp(in_channels, out_channels, 48000.0);
  assert(!myDsp.HasInputLevel());

  const double level = 19.0;
  myDsp.SetInputLevel(level);
  assert(myDsp.HasInputLevel());
}

void test_has_output_level()
{
  const int in_channels = 1;
  const int out_channels = 2;
  nam::DSP myDsp(in_channels, out_channels, 48000.0);

  assert(!myDsp.HasOutputLevel());

  const double level = 12.0;
  myDsp.SetOutputLevel(level);
  assert(myDsp.HasOutputLevel());
}

// Test correct function of DSP::HasInputLevel()
void test_set_input_level()
{
  const int in_channels = 2;
  const int out_channels = 1;
  nam::DSP myDsp(in_channels, out_channels, 48000.0);
  myDsp.SetInputLevel(19.0);
}

void test_set_output_level()
{
  const int in_channels = 1;
  const int out_channels = 2;
  nam::DSP myDsp(in_channels, out_channels, 48000.0);
  myDsp.SetOutputLevel(19.0);
}

void test_process_multi_channel()
{
  const int in_channels = 2;
  const int out_channels = 2;
  nam::DSP myDsp(in_channels, out_channels, 48000.0);
  const int num_frames = 64;

  // Allocate buffers
  std::vector<std::vector<double>> inputBuffers(in_channels);
  std::vector<std::vector<double>> outputBuffers(out_channels);
  std::vector<double*> inputPtrs(in_channels);
  std::vector<double*> outputPtrs(out_channels);

  for (int ch = 0; ch < in_channels; ch++)
  {
    inputBuffers[ch].resize(num_frames);
    outputBuffers[ch].resize(num_frames);
    inputPtrs[ch] = inputBuffers[ch].data();
    outputPtrs[ch] = outputBuffers[ch].data();

    // Fill input with test data
    for (int i = 0; i < num_frames; i++)
    {
      inputBuffers[ch][i] = (ch + 1) * 0.5 + i * 0.01;
    }
  }
  for (int ch = 0; ch < out_channels; ch++)
  {
    outputBuffers[ch].resize(num_frames);
    outputPtrs[ch] = outputBuffers[ch].data();
  }

  // Process
  myDsp.process(inputPtrs.data(), outputPtrs.data(), num_frames);

  // Check that default implementation copied input to output
  const int channelsToCheck = std::min(in_channels, out_channels);
  for (int ch = 0; ch < channelsToCheck; ch++)
  {
    for (int i = 0; i < num_frames; i++)
    {
      assert(outputBuffers[ch][i] == inputBuffers[ch][i]);
    }
  }
}
}; // namespace test_dsp
