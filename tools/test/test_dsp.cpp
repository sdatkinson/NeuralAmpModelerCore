#include <catch2/catch_test_macros.hpp>

#include "NAM/dsp.h"

TEST_CASE("DSP can be constructed", "[dsp]")
{
  nam::DSP myDsp(48000.0);
}

TEST_CASE("DSP GetInputLevel returns the value set by SetInputLevel", "[dsp]")
{
  nam::DSP myDsp(48000.0);
  const double expected = 19.0;
  myDsp.SetInputLevel(expected);
  REQUIRE(myDsp.HasInputLevel());
  const double actual = myDsp.GetInputLevel();

  REQUIRE(actual == expected);
}

TEST_CASE("DSP GetOutputLevel returns the value set by SetOutputLevel", "[dsp]")
{
  nam::DSP myDsp(48000.0);
  const double expected = 12.0;
  myDsp.SetOutputLevel(expected);
  REQUIRE(myDsp.HasOutputLevel());
  const double actual = myDsp.GetOutputLevel();

  REQUIRE(actual == expected);
}

TEST_CASE("DSP HasInputLevel is true after SetInputLevel", "[dsp]")
{
  nam::DSP myDsp(48000.0);
  REQUIRE(!myDsp.HasInputLevel());

  myDsp.SetInputLevel(19.0);
  REQUIRE(myDsp.HasInputLevel());
}

TEST_CASE("DSP HasOutputLevel is true after SetOutputLevel", "[dsp]")
{
  nam::DSP myDsp(48000.0);
  REQUIRE(!myDsp.HasOutputLevel());

  myDsp.SetOutputLevel(12.0);
  REQUIRE(myDsp.HasOutputLevel());
}

TEST_CASE("DSP SetInputLevel does not crash", "[dsp]")
{
  nam::DSP myDsp(48000.0);
  myDsp.SetInputLevel(19.0);
}

TEST_CASE("DSP SetOutputLevel does not crash", "[dsp]")
{
  nam::DSP myDsp(48000.0);
  myDsp.SetOutputLevel(19.0);
}