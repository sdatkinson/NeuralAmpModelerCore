#include <catch2/catch_test_macros.hpp>
#include <Eigen/Dense>
#include "NAM/wavenet.h"

TEST_CASE("WaveNet gated activation", "[wavenet]") {
  const int conditionSize = 1;
  const int channels = 1;
  const int kernelSize = 1;
  const int dilation = 1;
  const std::string activation = "ReLU";
  const bool gated = true;
  auto layer = nam::wavenet::_Layer(conditionSize, channels, kernelSize, dilation, activation, gated);

  std::vector<float> weights{
    1.0f, 1.0f, 0.0f, 0.0f,
    1.0f, -1.0f,
    1.0f, 0.0f};
  auto it = weights.begin();
  layer.set_weights_(it);
  REQUIRE(it == weights.end());

  const long numFrames = 4;
  layer.SetMaxBufferSize(numFrames);

  Eigen::MatrixXf input, condition, headInput, output;
  input.resize(channels, numFrames);
  condition.resize(channels, numFrames);
  headInput.resize(channels, numFrames);
  output.resize(channels, numFrames);

  const float signalValue = 0.25f;
  input.fill(signalValue);
  condition.fill(signalValue);
  headInput.setZero();
  output.setZero();

  layer.process_(input, condition, headInput, output, 0, 0, (int)numFrames);

  const float expectedOutput = 0.5;
  const float expectedHeadInput = 0.25;
  for (int i = 0; i < numFrames; i++) {
    REQUIRE(output(0, i) == expectedOutput);
    REQUIRE(headInput(0, i) == expectedHeadInput);
  }
}