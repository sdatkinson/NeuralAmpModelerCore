// Tests for dsp

#include "NAM/dsp.h"

namespace test_dsp
{
// Simplest test: can I construct something!
void test_construct()
{
  nam::DSP myDsp(48000.0);
}

void test_get_input_level()
{
  nam::DSP myDsp(48000.0);
  const double expected = 19.0;
  myDsp.SetInputLevel(expected);
  assert(myDsp.HasInputLevel());
  const double actual = myDsp.GetInputLevel();

  assert(actual == expected);
}

void test_get_output_level()
{
  nam::DSP myDsp(48000.0);
  const double expected = 12.0;
  myDsp.SetOutputLevel(expected);
  assert(myDsp.HasOutputLevel());
  const double actual = myDsp.GetOutputLevel();

  assert(actual == expected);
}

// Test correct function of DSP::HasInputLevel()
void test_has_input_level()
{
  nam::DSP myDsp(48000.0);
  assert(!myDsp.HasInputLevel());

  myDsp.SetInputLevel(19.0);
  assert(myDsp.HasInputLevel());
}

void test_has_output_level()
{
  nam::DSP myDsp(48000.0);
  assert(!myDsp.HasOutputLevel());

  myDsp.SetOutputLevel(12.0);
  assert(myDsp.HasOutputLevel());
}

// Test correct function of DSP::HasInputLevel()
void test_set_input_level()
{
  nam::DSP myDsp(48000.0);
  myDsp.SetInputLevel(19.0);
}

void test_set_output_level()
{
  nam::DSP myDsp(48000.0);
  myDsp.SetOutputLevel(19.0);
}
}; // namespace test_dsp
