// Microbenchmark and correctness check for grouped Conv1x1.
// Sweeps registered (channels, groups) combinations.
// Mirrors the "1x1_groups" plot in issue #215.

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

#include "NAM/dsp.h"

using clk = std::chrono::high_resolution_clock;

// Reference implementation: dense block-diagonal GEMM. Should match templated kernel
// bit-for-bit when groups==1, and within float-rounding tolerance otherwise (different
// accumulation order). Used by check_correctness().
static Eigen::MatrixXf reference_conv1x1(int channels, int groups, const std::vector<float>& weights,
                                         const Eigen::MatrixXf& input)
{
  Eigen::MatrixXf W(channels, channels);
  W.setZero();
  const int per_group = channels / groups;
  size_t idx = 0;
  for (int g = 0; g < groups; g++)
  {
    for (int i = 0; i < per_group; i++)
    {
      for (int j = 0; j < per_group; j++)
      {
        W(g * per_group + i, g * per_group + j) = weights[idx++];
      }
    }
  }
  return W * input;
}

// Verify templated kernel produces identical output (within tolerance) to a plain
// dense GEMM reference for the given (channels, groups). Returns max abs diff.
static double check_correctness(int channels, int groups, int frames, std::mt19937& rng)
{
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  const int per_group = channels / groups;
  std::vector<float> weights(groups * per_group * per_group);
  for (auto& w : weights)
    w = dist(rng);

  nam::Conv1x1 conv(channels, channels, /*bias=*/false, groups);
  auto it = weights.begin();
  conv.set_weights_(it);
  conv.SetMaxBufferSize(frames);

  Eigen::MatrixXf input(channels, frames);
  for (int r = 0; r < channels; r++)
    for (int c = 0; c < frames; c++)
      input(r, c) = dist(rng);

  conv.process_(input, frames);
  const Eigen::MatrixXf& got = conv.GetOutput();
  Eigen::MatrixXf want = reference_conv1x1(channels, groups, weights, input);

  double max_diff = 0.0;
  for (int r = 0; r < channels; r++)
    for (int c = 0; c < frames; c++)
      max_diff = std::max(max_diff, (double)std::abs(got(r, c) - want(r, c)));
  return max_diff;
}

static double bench_one(int channels, int groups, int frames, int iters)
{
  std::vector<float> weights;
  // Per-group block has (channels/groups) * (channels/groups) weights.
  const int per_group = channels / groups;
  weights.resize(groups * per_group * per_group, 0.123f);

  nam::Conv1x1 conv(channels, channels, /*bias=*/false, groups);
  auto it = weights.begin();
  conv.set_weights_(it);
  conv.SetMaxBufferSize(frames);

  Eigen::MatrixXf input(channels, frames);
  input.setRandom();

  // Warmup
  for (int i = 0; i < 100; i++)
    conv.process_(input, frames);

  auto t1 = clk::now();
  for (int i = 0; i < iters; i++)
    conv.process_(input, frames);
  auto t2 = clk::now();

  // Read output to defeat dead-store elimination.
  volatile float sink = conv.GetOutput().sum();
  (void)sink;

  return std::chrono::duration<double, std::nano>(t2 - t1).count() / iters;
}

int main(int argc, char** argv)
{
  const int frames = 64;
  const int iters = 2'000'000;

  std::cout << "Conv1x1 microbench: frames=" << frames << " iters=" << iters << "\n";
#ifdef NAM_USE_INLINE_GEMM
  std::cout << "Build: NAM_USE_INLINE_GEMM\n";
#else
  std::cout << "Build: standard (Eigen GEMM)\n";
#endif

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

  // Correctness gate: compare templated kernel against reference dense GEMM for every
  // registered shape with random weights and random input. Bail with non-zero exit if
  // any shape fails the tolerance check.
  std::cout << "\n== Correctness check (templated vs reference dense GEMM) ==\n";
  std::mt19937 rng(0xC0FFEE);
  const double tol = 1e-4; // accounts for accumulation-order rounding
  bool ok = true;
  for (const auto& s : shapes)
  {
    for (int g : s.groups)
    {
      double diff = check_correctness(s.channels, g, frames, rng);
      const bool pass = diff < tol;
      std::cout << "  ch=" << s.channels << " G=" << g << "  max_abs_diff=" << diff << (pass ? "  OK" : "  FAIL")
                << "\n";
      if (!pass)
        ok = false;
    }
  }
  if (!ok)
  {
    std::cerr << "FAIL: at least one shape exceeded tolerance " << tol << "\n";
    return 2;
  }

  for (const auto& s : shapes)
  {
    std::cout << "\n-- channels=" << s.channels << " --\n";
    for (int g : s.groups)
    {
      double best = 1e18;
      for (int r = 0; r < 3; r++)
      {
        double ns = bench_one(s.channels, g, frames, iters);
        if (ns < best)
          best = ns;
      }
      std::cout << "groups=" << g << "  per_call=" << best << " ns"
                << "  per_frame=" << (best / frames) << " ns\n";
    }
  }
  return 0;
}
