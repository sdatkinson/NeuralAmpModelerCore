// Benchmark for fully fixed convolution implementations
// Compares Conv1x1FullyFixed and Conv1DFullyFixed (all dimensions fixed) vs dynamic implementations

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include "NAM/conv1d.h"
#include "NAM/conv1d_fixed.h"
#include "NAM/conv1x1_fixed.h"
#include "NAM/dsp.h"

using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::nanoseconds;

constexpr int NUM_WARMUP = 100;
constexpr int NUM_ITERATIONS = 1000;

struct Result
{
  double mean_ns;
  double stddev_ns;
};

Result calculate_stats(const std::vector<double>& samples)
{
  double sum = 0.0;
  for (double s : samples)
    sum += s;
  double mean = sum / samples.size();

  double sq_sum = 0.0;
  for (double s : samples)
  {
    double diff = s - mean;
    sq_sum += diff * diff;
  }
  return {mean, std::sqrt(sq_sum / samples.size())};
}

// Benchmark Conv1x1FullyFixed vs Conv1x1 (dynamic)
template <int Channels, int MaxFrames, int Groups, bool HasBias>
void benchmark_conv1x1_fully_fixed(std::mt19937& rng)
{
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  // Generate weights
  constexpr int in_per_group = Channels / Groups;
  constexpr int out_per_group = Channels / Groups;
  std::vector<float> weights;
  for (int g = 0; g < Groups; g++)
    for (int i = 0; i < out_per_group; i++)
      for (int j = 0; j < in_per_group; j++)
        weights.push_back(dist(rng));
  if constexpr (HasBias)
    for (int i = 0; i < Channels; i++)
      weights.push_back(dist(rng));

  // Create input (dynamic for interface, but we'll also create fixed version)
  Eigen::MatrixXf input_dynamic(Channels, MaxFrames);
  for (int i = 0; i < Channels; i++)
    for (int j = 0; j < MaxFrames; j++)
      input_dynamic(i, j) = dist(rng);

  // Fixed-size input
  Eigen::Matrix<float, Channels, MaxFrames> input_fixed = input_dynamic;

  // ========== FULLY FIXED ==========
  nam::Conv1x1FullyFixed<Channels, Channels, MaxFrames, Groups, HasBias> conv_fixed;
  auto it1 = weights.begin();
  conv_fixed.set_weights_(it1);
  conv_fixed.SetMaxBufferSize(MaxFrames);

  // Warmup
  for (int i = 0; i < NUM_WARMUP; i++)
    conv_fixed.process_(input_dynamic, MaxFrames);

  std::vector<double> fixed_samples;
  fixed_samples.reserve(NUM_ITERATIONS);
  for (int i = 0; i < NUM_ITERATIONS; i++)
  {
    auto t1 = high_resolution_clock::now();
    conv_fixed.process_(input_dynamic, MaxFrames);
    auto t2 = high_resolution_clock::now();
    fixed_samples.push_back(static_cast<double>(duration_cast<nanoseconds>(t2 - t1).count()));
  }

  // ========== DYNAMIC ==========
  nam::Conv1x1 conv_dynamic(Channels, Channels, HasBias, Groups);
  auto it2 = weights.begin();
  conv_dynamic.set_weights_(it2);
  conv_dynamic.SetMaxBufferSize(MaxFrames);

  // Warmup
  for (int i = 0; i < NUM_WARMUP; i++)
    conv_dynamic.process_(input_dynamic, MaxFrames);

  std::vector<double> dynamic_samples;
  dynamic_samples.reserve(NUM_ITERATIONS);
  for (int i = 0; i < NUM_ITERATIONS; i++)
  {
    auto t1 = high_resolution_clock::now();
    conv_dynamic.process_(input_dynamic, MaxFrames);
    auto t2 = high_resolution_clock::now();
    dynamic_samples.push_back(static_cast<double>(duration_cast<nanoseconds>(t2 - t1).count()));
  }

  Result fixed_result = calculate_stats(fixed_samples);
  Result dynamic_result = calculate_stats(dynamic_samples);

  double speedup = dynamic_result.mean_ns / fixed_result.mean_ns;

  std::cout << "Conv1x1," << Channels << "," << Groups << "," << (HasBias ? "true" : "false") << "," << MaxFrames << ","
            << std::fixed << std::setprecision(1) << dynamic_result.mean_ns << "," << fixed_result.mean_ns << ","
            << std::setprecision(2) << speedup << "x\n";
}

// Benchmark Conv1DFullyFixed vs Conv1D (dynamic)
template <int Channels, int KernelSize, int MaxFrames, int Groups, bool HasBias>
void benchmark_conv1d_fully_fixed(std::mt19937& rng)
{
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  // Generate weights
  constexpr int in_per_group = Channels / Groups;
  constexpr int out_per_group = Channels / Groups;
  std::vector<float> weights;
  for (int g = 0; g < Groups; g++)
    for (int i = 0; i < out_per_group; i++)
      for (int j = 0; j < in_per_group; j++)
        for (int k = 0; k < KernelSize; k++)
          weights.push_back(dist(rng));
  if constexpr (HasBias)
    for (int i = 0; i < Channels; i++)
      weights.push_back(dist(rng));

  // Create input
  Eigen::MatrixXf input(Channels, MaxFrames);
  for (int i = 0; i < Channels; i++)
    for (int j = 0; j < MaxFrames; j++)
      input(i, j) = dist(rng);

  const int dilation = 1;

  // ========== FULLY FIXED ==========
  nam::Conv1DFullyFixed<Channels, Channels, KernelSize, MaxFrames, Groups, HasBias> conv_fixed(dilation);
  auto it1 = weights.begin();
  conv_fixed.set_weights_(it1);
  conv_fixed.SetMaxBufferSize(MaxFrames);

  // Warmup
  for (int i = 0; i < NUM_WARMUP; i++)
    conv_fixed.Process(input, MaxFrames);

  std::vector<double> fixed_samples;
  fixed_samples.reserve(NUM_ITERATIONS);
  for (int i = 0; i < NUM_ITERATIONS; i++)
  {
    auto t1 = high_resolution_clock::now();
    conv_fixed.Process(input, MaxFrames);
    auto t2 = high_resolution_clock::now();
    fixed_samples.push_back(static_cast<double>(duration_cast<nanoseconds>(t2 - t1).count()));
  }

  // ========== DYNAMIC ==========
  nam::Conv1D conv_dynamic;
  conv_dynamic.set_size_(Channels, Channels, KernelSize, HasBias, dilation, Groups);
  auto it2 = weights.begin();
  conv_dynamic.set_weights_(it2);
  conv_dynamic.SetMaxBufferSize(MaxFrames);

  // Warmup
  for (int i = 0; i < NUM_WARMUP; i++)
    conv_dynamic.Process(input, MaxFrames);

  std::vector<double> dynamic_samples;
  dynamic_samples.reserve(NUM_ITERATIONS);
  for (int i = 0; i < NUM_ITERATIONS; i++)
  {
    auto t1 = high_resolution_clock::now();
    conv_dynamic.Process(input, MaxFrames);
    auto t2 = high_resolution_clock::now();
    dynamic_samples.push_back(static_cast<double>(duration_cast<nanoseconds>(t2 - t1).count()));
  }

  Result fixed_result = calculate_stats(fixed_samples);
  Result dynamic_result = calculate_stats(dynamic_samples);

  double speedup = dynamic_result.mean_ns / fixed_result.mean_ns;

  std::cout << "Conv1D," << Channels << "," << Groups << "," << KernelSize << "," << (HasBias ? "true" : "false") << ","
            << MaxFrames << "," << std::fixed << std::setprecision(1) << dynamic_result.mean_ns << ","
            << fixed_result.mean_ns << "," << std::setprecision(2) << speedup << "x\n";
}

int main()
{
  std::mt19937 rng(42);

  std::cout << "================================================================================\n";
  std::cout << "CONV1X1: Fully Fixed (all dimensions) vs Dynamic\n";
  std::cout << "================================================================================\n";
  std::cout << "Type,Channels,Groups,Bias,Frames,Dynamic(ns),FullyFixed(ns),Speedup\n";

  // Common audio buffer sizes: 32, 64, 128, 256, 512
  // Small channels (where fixed-size optimization helps most)

  // 2 channels
  benchmark_conv1x1_fully_fixed<2, 32, 1, true>(rng);
  benchmark_conv1x1_fully_fixed<2, 64, 1, true>(rng);
  benchmark_conv1x1_fully_fixed<2, 128, 1, true>(rng);
  benchmark_conv1x1_fully_fixed<2, 256, 1, true>(rng);
  benchmark_conv1x1_fully_fixed<2, 512, 1, true>(rng);

  // 4 channels
  benchmark_conv1x1_fully_fixed<4, 32, 1, true>(rng);
  benchmark_conv1x1_fully_fixed<4, 64, 1, true>(rng);
  benchmark_conv1x1_fully_fixed<4, 128, 1, true>(rng);
  benchmark_conv1x1_fully_fixed<4, 256, 1, true>(rng);
  benchmark_conv1x1_fully_fixed<4, 512, 1, true>(rng);

  // 4 channels with 4 groups (grouped convolution)
  benchmark_conv1x1_fully_fixed<4, 32, 4, true>(rng);
  benchmark_conv1x1_fully_fixed<4, 64, 4, true>(rng);
  benchmark_conv1x1_fully_fixed<4, 128, 4, true>(rng);
  benchmark_conv1x1_fully_fixed<4, 256, 4, true>(rng);
  benchmark_conv1x1_fully_fixed<4, 512, 4, true>(rng);

  // 8 channels
  benchmark_conv1x1_fully_fixed<8, 32, 1, true>(rng);
  benchmark_conv1x1_fully_fixed<8, 64, 1, true>(rng);
  benchmark_conv1x1_fully_fixed<8, 128, 1, true>(rng);
  benchmark_conv1x1_fully_fixed<8, 256, 1, true>(rng);
  benchmark_conv1x1_fully_fixed<8, 512, 1, true>(rng);

  // 8 channels with 8 groups
  benchmark_conv1x1_fully_fixed<8, 32, 8, true>(rng);
  benchmark_conv1x1_fully_fixed<8, 64, 8, true>(rng);
  benchmark_conv1x1_fully_fixed<8, 128, 8, true>(rng);
  benchmark_conv1x1_fully_fixed<8, 256, 8, true>(rng);
  benchmark_conv1x1_fully_fixed<8, 512, 8, true>(rng);

  std::cout << "\n================================================================================\n";
  std::cout << "CONV1D: Fully Fixed (all dimensions) vs Dynamic\n";
  std::cout << "================================================================================\n";
  std::cout << "Type,Channels,Groups,KernelSize,Bias,Frames,Dynamic(ns),FullyFixed(ns),Speedup\n";

  // Conv1D with kernel size 3 (most common)
  // 4 channels
  benchmark_conv1d_fully_fixed<4, 3, 32, 1, true>(rng);
  benchmark_conv1d_fully_fixed<4, 3, 64, 1, true>(rng);
  benchmark_conv1d_fully_fixed<4, 3, 128, 1, true>(rng);
  benchmark_conv1d_fully_fixed<4, 3, 256, 1, true>(rng);
  benchmark_conv1d_fully_fixed<4, 3, 512, 1, true>(rng);

  // 4 channels with 4 groups
  benchmark_conv1d_fully_fixed<4, 3, 32, 4, true>(rng);
  benchmark_conv1d_fully_fixed<4, 3, 64, 4, true>(rng);
  benchmark_conv1d_fully_fixed<4, 3, 128, 4, true>(rng);
  benchmark_conv1d_fully_fixed<4, 3, 256, 4, true>(rng);
  benchmark_conv1d_fully_fixed<4, 3, 512, 4, true>(rng);

  // 8 channels
  benchmark_conv1d_fully_fixed<8, 3, 32, 1, true>(rng);
  benchmark_conv1d_fully_fixed<8, 3, 64, 1, true>(rng);
  benchmark_conv1d_fully_fixed<8, 3, 128, 1, true>(rng);
  benchmark_conv1d_fully_fixed<8, 3, 256, 1, true>(rng);
  benchmark_conv1d_fully_fixed<8, 3, 512, 1, true>(rng);

  // 8 channels with 8 groups
  benchmark_conv1d_fully_fixed<8, 3, 32, 8, true>(rng);
  benchmark_conv1d_fully_fixed<8, 3, 64, 8, true>(rng);
  benchmark_conv1d_fully_fixed<8, 3, 128, 8, true>(rng);
  benchmark_conv1d_fully_fixed<8, 3, 256, 8, true>(rng);
  benchmark_conv1d_fully_fixed<8, 3, 512, 8, true>(rng);

  return 0;
}
