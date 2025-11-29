#include <catch2/catch_test_macros.hpp>
#include "NAM/dsp.h"

TEST_CASE("DSP construct", "[dsp]") {
  nam::DSP myDsp(48000.0);
}

TEST_CASE("DSP get input level", "[dsp]") {
  nam::DSP myDsp(48000.0);
  const double expected = 19.0;
  myDsp.SetInputLevel(expected);
  REQUIRE(myDsp.HasInputLevel());
  REQUIRE(myDsp.GetInputLevel() == expected);
}

TEST_CASE("DSP get output level", "[dsp]") {
  nam::DSP myDsp(48000.0);
  const double expected = 12.0;
  myDsp.SetOutputLevel(expected);
  REQUIRE(myDsp.HasOutputLevel());
  REQUIRE(myDsp.GetOutputLevel() == expected);
}

TEST_CASE("DSP has input level", "[dsp]") {
  nam::DSP myDsp(48000.0);
  REQUIRE(!myDsp.HasInputLevel());
  myDsp.SetInputLevel(19.0);
  REQUIRE(myDsp.HasInputLevel());
}

TEST_CASE("DSP has output level", "[dsp]") {
  nam::DSP myDsp(48000.0);
  REQUIRE(!myDsp.HasOutputLevel());
  myDsp.SetOutputLevel(12.0);
  REQUIRE(myDsp.HasOutputLevel());
}

TEST_CASE("DSP set input level", "[dsp]") {
  nam::DSP myDsp(48000.0);
  myDsp.SetInputLevel(19.0);
}

TEST_CASE("DSP set output level", "[dsp]") {
  nam::DSP myDsp(48000.0);
  myDsp.SetOutputLevel(19.0);
}
