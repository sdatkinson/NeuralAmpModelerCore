// Test to verify activation apply() overloads are real-time safe (no allocations/frees)

#include <Eigen/Dense>
#include <cassert>
#include <cmath>
#include <vector>

#include "NAM/activations.h"
#include "allocation_tracking.h"

namespace test_activations_realtime_safe
{
using namespace allocation_tracking;

// PReLU's matrix overload is reached once per sample from the gating/blending activations, so a
// heap allocation here lands directly on the audio thread.
void test_prelu_apply_matrix_realtime_safe()
{
  const int channels = 8;
  const int time_steps = 64;

  std::vector<float> slopes;
  for (int i = 0; i < channels; i++)
    slopes.push_back(0.01f * static_cast<float>(i + 1));

  nam::activations::ActivationPReLU activation(slopes);

  Eigen::MatrixXf matrix(channels, time_steps);
  matrix.setConstant(-1.0f);

  run_allocation_test_no_allocations(
    nullptr, // No setup needed
    [&]() { activation.apply(matrix); }, nullptr, // No teardown needed
    "test_prelu_apply_matrix_realtime_safe");

  // Each channel should have been scaled by its own slope.
  for (int c = 0; c < channels; c++)
    assert(std::abs(matrix(c, 0) - (-slopes[c])) < 1e-6f);
}

// The pointer/length overload is the one used by the plain (non-gated) path.
void test_prelu_apply_pointer_realtime_safe()
{
  const int channels = 4;
  const int time_steps = 32;

  std::vector<float> slopes;
  for (int i = 0; i < channels; i++)
    slopes.push_back(0.05f * static_cast<float>(i + 1));

  nam::activations::ActivationPReLU activation(slopes);

  Eigen::MatrixXf matrix(channels, time_steps);
  matrix.setConstant(-2.0f);

  run_allocation_test_no_allocations(
    nullptr, // No setup needed
    [&]() { activation.apply(matrix.data(), static_cast<long>(channels) * time_steps); },
    nullptr, // No teardown needed
    "test_prelu_apply_pointer_realtime_safe");

  for (int c = 0; c < channels; c++)
    assert(std::abs(matrix(c, 0) - (-2.0f * slopes[c])) < 1e-6f);
}
} // namespace test_activations_realtime_safe
