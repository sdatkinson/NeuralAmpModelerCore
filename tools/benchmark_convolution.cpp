// Microbenchmark for Conv1x1 and Conv1D convolution operations
// Measures performance across various configurations of channels, groups, and frame sizes.
// Outputs CSV format for analysis.

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include "NAM/conv1d.h"
#include "NAM/dsp.h"

using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::nanoseconds;

// Number of iterations per benchmark configuration
constexpr int NUM_WARMUP_ITERATIONS = 10;
constexpr int NUM_BENCHMARK_ITERATIONS = 100;

// Benchmark configurations
constexpr int CHANNELS[] = {2, 3, 4, 5, 6, 7, 8};
constexpr int GROUPS[] = {1, 2, 3, 4, 5, 6, 7, 8};
constexpr int FRAMES[] = {64, 256, 1024};
constexpr int KERNEL_SIZES[] = {1, 3}; // For Conv1D

struct BenchmarkResult
{
  double mean_ns;
  double stddev_ns;
  double min_ns;
  double max_ns;
};

// Calculate statistics from timing samples
BenchmarkResult calculate_stats(const std::vector<double>& samples)
{
  BenchmarkResult result;
  double sum = 0.0;
  result.min_ns = samples[0];
  result.max_ns = samples[0];

  for (double s : samples)
  {
    sum += s;
    if (s < result.min_ns)
      result.min_ns = s;
    if (s > result.max_ns)
      result.max_ns = s;
  }

  result.mean_ns = sum / samples.size();

  double sq_sum = 0.0;
  for (double s : samples)
  {
    double diff = s - result.mean_ns;
    sq_sum += diff * diff;
  }
  result.stddev_ns = std::sqrt(sq_sum / samples.size());

  return result;
}

// Benchmark Conv1x1
void benchmark_conv1x1(int channels, int groups, int frames, std::mt19937& rng)
{
  if (channels % groups != 0)
    return; // Skip invalid configurations

  // Create Conv1x1 layer
  nam::Conv1x1 conv(channels, channels, false, groups);

  // Initialize with random weights
  const int num_weights = (channels / groups) * (channels / groups) * groups;
  std::vector<float> weights(num_weights);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto& w : weights)
    w = dist(rng);

  auto it = weights.begin();
  conv.set_weights_(it);
  conv.SetMaxBufferSize(frames);

  // Create random input
  Eigen::MatrixXf input(channels, frames);
  for (int i = 0; i < channels; i++)
    for (int j = 0; j < frames; j++)
      input(i, j) = dist(rng);

  // Warmup
  for (int i = 0; i < NUM_WARMUP_ITERATIONS; i++)
  {
    conv.process_(input, frames);
  }

  // Benchmark
  std::vector<double> samples;
  samples.reserve(NUM_BENCHMARK_ITERATIONS);

  for (int i = 0; i < NUM_BENCHMARK_ITERATIONS; i++)
  {
    auto t1 = high_resolution_clock::now();
    conv.process_(input, frames);
    auto t2 = high_resolution_clock::now();
    samples.push_back(static_cast<double>(duration_cast<nanoseconds>(t2 - t1).count()));
  }

  BenchmarkResult result = calculate_stats(samples);

  // Output CSV row
  std::cout << "Conv1x1," << channels << "," << groups << "," << frames << ",1," << std::fixed << std::setprecision(2)
            << result.mean_ns << "," << result.stddev_ns << "," << result.min_ns << "," << result.max_ns << "\n";
}

// Benchmark Conv1D
void benchmark_conv1d(int channels, int groups, int frames, int kernel_size, std::mt19937& rng)
{
  if (channels % groups != 0)
    return; // Skip invalid configurations

  // Create Conv1D layer
  nam::Conv1D conv;
  conv.set_size_(channels, channels, kernel_size, false, 1, groups);

  // Initialize with random weights
  const int num_weights = kernel_size * (channels / groups) * (channels / groups) * groups;
  std::vector<float> weights(num_weights);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto& w : weights)
    w = dist(rng);

  auto it = weights.begin();
  conv.set_weights_(it);
  conv.SetMaxBufferSize(frames);

  // Create random input
  Eigen::MatrixXf input(channels, frames);
  for (int i = 0; i < channels; i++)
    for (int j = 0; j < frames; j++)
      input(i, j) = dist(rng);

  // Warmup
  for (int i = 0; i < NUM_WARMUP_ITERATIONS; i++)
  {
    conv.Process(input, frames);
  }

  // Benchmark
  std::vector<double> samples;
  samples.reserve(NUM_BENCHMARK_ITERATIONS);

  for (int i = 0; i < NUM_BENCHMARK_ITERATIONS; i++)
  {
    auto t1 = high_resolution_clock::now();
    conv.Process(input, frames);
    auto t2 = high_resolution_clock::now();
    samples.push_back(static_cast<double>(duration_cast<nanoseconds>(t2 - t1).count()));
  }

  BenchmarkResult result = calculate_stats(samples);

  // Output CSV row
  std::cout << "Conv1D," << channels << "," << groups << "," << frames << "," << kernel_size << "," << std::fixed
            << std::setprecision(2) << result.mean_ns << "," << result.stddev_ns << "," << result.min_ns << ","
            << result.max_ns << "\n";
}

int main(int argc, char* argv[])
{
  // Print CSV header
  std::cout << "type,channels,groups,frames,kernel_size,mean_ns,stddev_ns,min_ns,max_ns\n";

  // Use fixed seed for reproducibility
  std::mt19937 rng(42);

  // Benchmark Conv1x1
  for (int channels : CHANNELS)
  {
    for (int groups : GROUPS)
    {
      for (int frames : FRAMES)
      {
        benchmark_conv1x1(channels, groups, frames, rng);
      }
    }
  }

  // Benchmark Conv1D
  for (int channels : CHANNELS)
  {
    for (int groups : GROUPS)
    {
      for (int frames : FRAMES)
      {
        for (int kernel_size : KERNEL_SIZES)
        {
          benchmark_conv1d(channels, groups, frames, kernel_size, rng);
        }
      }
    }
  }

  return 0;
}
