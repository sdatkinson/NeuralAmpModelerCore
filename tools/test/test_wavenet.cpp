// Tests for the WaveNet

#include <Eigen/Dense>
#include <cassert>
#include <iostream>

#include "NAM/wavenet.h"

namespace test_wavenet
{
void test_gated()
{
  // Assert correct nuemrics of the gating activation.
  // Issue 101
  const int conditionSize = 1;
  const int channels = 1;
  const int kernelSize = 1;
  const int dilation = 1;
  const std::string activation = "ReLU";
  const bool gated = true;
  auto layer = nam::wavenet::_Layer(conditionSize, channels, kernelSize, dilation, activation, gated);

  // Conv, input mixin, 1x1
  std::vector<float> weights{
    // Conv (weight, bias)  NOTE: 2 channels out bc gated, so shapes are (2,1,1), (2,)
    1.0f, 1.0f, 0.0f, 0.0f,
    // Input mixin (weight only: (2,1,1))
    1.0f, -1.0f,
    // 1x1 (weight (1,1,1), bias (1,))
    // NOTE: Weights are (1,1) on conv, (1,-1), so the inputs sum on the upper channel and cancel on the lower.
    // This should give us a nice zero if the input & condition are the same, so that'll sigmoid to 0.5 for the
    // gate.
    1.0f, 0.0f};
  auto it = weights.begin();
  layer.set_weights_(it);
  assert(it == weights.end());

  const long numFrames = 4;
  layer.set_num_frames_(numFrames);

  Eigen::MatrixXf input, condition, headInput, output;
  input.resize(channels, numFrames);
  condition.resize(channels, numFrames);
  headInput.resize(channels, numFrames);
  output.resize(channels, numFrames);

  const float signalValue = 0.25f;
  input.fill(signalValue);
  condition.fill(signalValue);
  // So input & condition will sum to 0.5 on the top channel (-> ReLU), cancel to 0 on bottom (-> sigmoid)

  headInput.setZero();
  output.setZero();

  layer.process_(input, condition, headInput, output, 0, 0);

  // 0.25 + 0.25 -> 0.5 for conv & input mixin top channel
  // (0 on bottom channel)
  // Top ReLU -> preseves 0.5
  // Bottom sigmoid 0->0.5
  // Product is 0.25
  // 1x1 is unity
  // Skip-connect -> 0.25 (input) + 0.25 (output) -> 0.5 output
  // head output gets 0+0.25 = 0.25
  const float expectedOutput = 0.5;
  const float expectedHeadInput = 0.25;
  for (int i = 0; i < numFrames; i++)
  {
    const float actualOutput = output(0, i);
    const float actualHeadInput = headInput(0, i);
    // std::cout << actualOutput << std::endl;
    assert(actualOutput == expectedOutput);
    assert(actualHeadInput == expectedHeadInput);
  }
}
}; // namespace test_wavenet