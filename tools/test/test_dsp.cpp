// Tests for dsp

#include "NAM/dsp.h"
#include <vector>

namespace test_dsp
{
// Simplest test: can I construct something!
void test_construct()
{
  nam::DSP myDsp(1, 1, 48000.0);
}

void test_channels()
{
  nam::DSP myDsp(2, 3, 48000.0);
  assert(myDsp.NumInputChannels() == 2);
  assert(myDsp.NumOutputChannels() == 3);
}

void test_get_input_level()
{
  nam::DSP myDsp(2, 1, 48000.0);
  const double expected = 19.0;
  myDsp.SetInputLevel(0, expected);
  assert(myDsp.HasInputLevel(0));
  const double actual = myDsp.GetInputLevel(0);

  assert(actual == expected);
}

void test_get_output_level()
{
  nam::DSP myDsp(1, 2, 48000.0);
  const double expected = 12.0;
  myDsp.SetOutputLevel(1, expected);
  assert(myDsp.HasOutputLevel(1));
  const double actual = myDsp.GetOutputLevel(1);

  assert(actual == expected);
}

// Test correct function of DSP::HasInputLevel()
void test_has_input_level()
{
  nam::DSP myDsp(2, 1, 48000.0);
  myDsp.SetInputLevel(0, 19.0);
  assert(myDsp.HasInputLevel(0));
  assert(!myDsp.HasInputLevel(1));
}

void test_has_output_level()
{
  nam::DSP myDsp(1, 2, 48000.0);

  assert(!myDsp.HasOutputLevel(0));
  assert(!myDsp.HasOutputLevel(1));

  myDsp.SetOutputLevel(1, 12.0);
  assert(!myDsp.HasOutputLevel(0));
  assert(myDsp.HasOutputLevel(1));
}

// Test correct function of DSP::HasInputLevel()
void test_set_input_level()
{
  nam::DSP myDsp(2, 1, 48000.0);
  myDsp.SetInputLevel(0, 19.0);
  myDsp.SetInputLevel(1, 20.0);
}

void test_set_output_level()
{
  nam::DSP myDsp(1, 2, 48000.0);
  myDsp.SetOutputLevel(0, 19.0);
  myDsp.SetOutputLevel(1, 20.0);
}

void test_process_multi_channel()
{
  nam::DSP myDsp(2, 2, 48000.0);
  const int num_frames = 64;

  // Allocate buffers
  std::vector<std::vector<double>> inputBuffers(2);
  std::vector<std::vector<double>> outputBuffers(2);
  std::vector<double*> inputPtrs(2);
  std::vector<double*> outputPtrs(2);

  for (int ch = 0; ch < 2; ch++)
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

  // Process
  myDsp.process(inputPtrs.data(), outputPtrs.data(), num_frames);

  // Check that default implementation copied input to output
  for (int ch = 0; ch < 2; ch++)
  {
    for (int i = 0; i < num_frames; i++)
    {
      assert(outputBuffers[ch][i] == inputBuffers[ch][i]);
    }
  }
}
}; // namespace test_dsp
