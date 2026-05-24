// Correctness check for grouped Conv1D templated tap kernel.
// Compares Conv1D::Process output against a reference dense block-diagonal GEMM
// for every (channels, groups, kernel_size, dilation) shape that the templated
// dispatch registers. Exits non-zero if any shape exceeds float-rounding tolerance.

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

#include "NAM/conv1d.h"

namespace
{
Eigen::MatrixXf reference_conv1d(int channels, int groups, int kernel_size, int dilation,
                                 const std::vector<float>& weights, const Eigen::MatrixXf& padded_input,
                                 int num_frames, int lookback_max)
{
  // Reconstruct dense block-diagonal weight matrices, one per kernel tap.
  const int per_group = channels / groups;
  std::vector<Eigen::MatrixXf> W(kernel_size, Eigen::MatrixXf::Zero(channels, channels));
  // Conv1D weight layout (set_weights_): for each group, for each out-in pair,
  // for each kernel position, one weight.
  size_t idx = 0;
  for (int g = 0; g < groups; g++)
  {
    for (int i = 0; i < per_group; i++)
    {
      for (int j = 0; j < per_group; j++)
      {
        for (int k = 0; k < kernel_size; k++)
        {
          W[k](g * per_group + i, g * per_group + j) = weights[idx++];
        }
      }
    }
  }
  Eigen::MatrixXf out(channels, num_frames);
  out.setZero();
  // padded_input has lookback_max columns of history prepended.
  for (int k = 0; k < kernel_size; k++)
  {
    const long offset = dilation * (k + 1 - kernel_size); // <= 0
    const long lookback = -offset;
    out.noalias() += W[k] * padded_input.middleCols(lookback_max - lookback, num_frames);
  }
  return out;
}

double check_one(int channels, int groups, int kernel_size, int dilation, int frames, std::mt19937& rng)
{
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  const int per_group = channels / groups;
  std::vector<float> weights(groups * per_group * per_group * kernel_size + channels);
  for (auto& w : weights)
    w = dist(rng);

  nam::Conv1D conv(channels, channels, kernel_size, /*bias=*/true, dilation, groups);
  auto it = weights.begin();
  conv.set_weights_(it);
  conv.SetMaxBufferSize(frames);

  // Generate an input matrix; also build a padded version with lookback history (zeros)
  // so the reference can do the same dilated taps.
  Eigen::MatrixXf input(channels, frames);
  for (int r = 0; r < channels; r++)
    for (int c = 0; c < frames; c++)
      input(r, c) = dist(rng);

  conv.Process(input, frames);
  const Eigen::MatrixXf& got = conv.GetOutput();

  const int lookback_max = (kernel_size - 1) * dilation;
  Eigen::MatrixXf padded(channels, lookback_max + frames);
  padded.setZero();
  padded.rightCols(frames) = input;

  Eigen::MatrixXf want = reference_conv1d(channels, groups, kernel_size, dilation, weights, padded, frames, lookback_max);
  // Add bias.
  const float* bias = weights.data() + (weights.size() - channels);
  for (int r = 0; r < channels; r++)
    for (int c = 0; c < frames; c++)
      want(r, c) += bias[r];

  double max_diff = 0.0;
  for (int r = 0; r < channels; r++)
    for (int c = 0; c < frames; c++)
      max_diff = std::max(max_diff, (double)std::abs(got(r, c) - want(r, c)));
  return max_diff;
}
} // namespace

int main()
{
  struct Shape
  {
    int channels;
    std::vector<int> groups;
  };
  const std::vector<Shape> shapes = {
    {4, {1, 2, 4}},
    {6, {1, 2, 3, 6}},
    {8, {1, 2, 4, 8}},
    {12, {1, 2, 3, 4, 6, 12}},
    {16, {1, 2, 4, 8, 16}},
  };
  const std::vector<int> kernel_sizes = {1, 2, 3};
  const std::vector<int> dilations = {1, 2, 7};
  const int frames = 64;
  const double tol = 1e-4;

  std::mt19937 rng(0xBADCAB);
  bool ok = true;
  int n = 0;
  for (const auto& s : shapes)
  {
    for (int g : s.groups)
    {
      for (int K : kernel_sizes)
      {
        for (int D : dilations)
        {
          const double diff = check_one(s.channels, g, K, D, frames, rng);
          const bool pass = diff < tol;
          std::cout << "ch=" << s.channels << " G=" << g << " K=" << K << " D=" << D << "  diff=" << diff
                    << (pass ? "  OK" : "  FAIL") << "\n";
          if (!pass)
            ok = false;
          n++;
        }
      }
    }
  }
  std::cout << (ok ? "ALL OK" : "FAIL") << " (" << n << " shapes)\n";
  return ok ? 0 : 1;
}
