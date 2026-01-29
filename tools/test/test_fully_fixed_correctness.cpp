// Tests for Conv1x1FullyFixed and Conv1DFullyFixed correctness
// Compares outputs against dynamic implementations

#include <Eigen/Dense>
#include <cassert>
#include <cmath>
#include <random>
#include <vector>

#include "NAM/conv1d.h"
#include "NAM/conv1d_fixed.h"
#include "NAM/conv1x1_fixed.h"
#include "NAM/dsp.h"

namespace test_fully_fixed_correctness
{

constexpr float TOLERANCE = 1e-5f;

// Helper to check matrix equality
inline void assert_matrices_equal(const Eigen::MatrixXf& a, const Eigen::MatrixXf& b, int num_cols,
                                  float tol = TOLERANCE)
{
  assert(a.rows() == b.rows());
  for (int i = 0; i < a.rows(); i++)
  {
    for (int j = 0; j < num_cols; j++)
    {
      float diff = std::abs(a(i, j) - b(i, j));
      assert(diff < tol);
    }
  }
}

// ============================================================================
// Conv1x1FullyFixed Tests
// ============================================================================

void test_conv1x1_fully_fixed_2ch_32frames()
{
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  constexpr int Channels = 2;
  constexpr int MaxFrames = 32;
  constexpr int Groups = 1;
  constexpr bool HasBias = true;

  std::vector<float> weights;
  for (int i = 0; i < Channels * Channels; i++)
    weights.push_back(dist(rng));
  for (int i = 0; i < Channels; i++)
    weights.push_back(dist(rng));

  nam::Conv1x1FullyFixed<Channels, Channels, MaxFrames, Groups, HasBias> conv_fixed;
  nam::Conv1x1 conv_dynamic(Channels, Channels, HasBias, Groups);

  auto it1 = weights.begin();
  auto it2 = weights.begin();
  conv_fixed.set_weights_(it1);
  conv_dynamic.set_weights_(it2);

  conv_fixed.SetMaxBufferSize(MaxFrames);
  conv_dynamic.SetMaxBufferSize(MaxFrames);

  Eigen::MatrixXf input(Channels, MaxFrames);
  for (int i = 0; i < Channels; i++)
    for (int j = 0; j < MaxFrames; j++)
      input(i, j) = dist(rng);

  conv_fixed.process_(input, MaxFrames);
  conv_dynamic.process_(input, MaxFrames);

  assert_matrices_equal(conv_fixed.GetOutput(), conv_dynamic.GetOutput(), MaxFrames);
}

void test_conv1x1_fully_fixed_4ch_64frames()
{
  std::mt19937 rng(123);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  constexpr int Channels = 4;
  constexpr int MaxFrames = 64;
  constexpr int Groups = 1;
  constexpr bool HasBias = true;

  std::vector<float> weights;
  for (int i = 0; i < Channels * Channels; i++)
    weights.push_back(dist(rng));
  for (int i = 0; i < Channels; i++)
    weights.push_back(dist(rng));

  nam::Conv1x1FullyFixed<Channels, Channels, MaxFrames, Groups, HasBias> conv_fixed;
  nam::Conv1x1 conv_dynamic(Channels, Channels, HasBias, Groups);

  auto it1 = weights.begin();
  auto it2 = weights.begin();
  conv_fixed.set_weights_(it1);
  conv_dynamic.set_weights_(it2);

  conv_fixed.SetMaxBufferSize(MaxFrames);
  conv_dynamic.SetMaxBufferSize(MaxFrames);

  Eigen::MatrixXf input(Channels, MaxFrames);
  for (int i = 0; i < Channels; i++)
    for (int j = 0; j < MaxFrames; j++)
      input(i, j) = dist(rng);

  conv_fixed.process_(input, MaxFrames);
  conv_dynamic.process_(input, MaxFrames);

  assert_matrices_equal(conv_fixed.GetOutput(), conv_dynamic.GetOutput(), MaxFrames);
}

void test_conv1x1_fully_fixed_4ch_4groups()
{
  std::mt19937 rng(456);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  constexpr int Channels = 4;
  constexpr int MaxFrames = 64;
  constexpr int Groups = 4;
  constexpr bool HasBias = true;
  constexpr int PerGroup = Channels / Groups;

  std::vector<float> weights;
  for (int g = 0; g < Groups; g++)
    for (int i = 0; i < PerGroup; i++)
      for (int j = 0; j < PerGroup; j++)
        weights.push_back(dist(rng));
  for (int i = 0; i < Channels; i++)
    weights.push_back(dist(rng));

  nam::Conv1x1FullyFixed<Channels, Channels, MaxFrames, Groups, HasBias> conv_fixed;
  nam::Conv1x1 conv_dynamic(Channels, Channels, HasBias, Groups);

  auto it1 = weights.begin();
  auto it2 = weights.begin();
  conv_fixed.set_weights_(it1);
  conv_dynamic.set_weights_(it2);

  conv_fixed.SetMaxBufferSize(MaxFrames);
  conv_dynamic.SetMaxBufferSize(MaxFrames);

  Eigen::MatrixXf input(Channels, MaxFrames);
  for (int i = 0; i < Channels; i++)
    for (int j = 0; j < MaxFrames; j++)
      input(i, j) = dist(rng);

  conv_fixed.process_(input, MaxFrames);
  conv_dynamic.process_(input, MaxFrames);

  assert_matrices_equal(conv_fixed.GetOutput(), conv_dynamic.GetOutput(), MaxFrames);
}

void test_conv1x1_fully_fixed_8ch_8groups()
{
  std::mt19937 rng(789);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  constexpr int Channels = 8;
  constexpr int MaxFrames = 128;
  constexpr int Groups = 8;
  constexpr bool HasBias = true;
  constexpr int PerGroup = Channels / Groups;

  std::vector<float> weights;
  for (int g = 0; g < Groups; g++)
    for (int i = 0; i < PerGroup; i++)
      for (int j = 0; j < PerGroup; j++)
        weights.push_back(dist(rng));
  for (int i = 0; i < Channels; i++)
    weights.push_back(dist(rng));

  nam::Conv1x1FullyFixed<Channels, Channels, MaxFrames, Groups, HasBias> conv_fixed;
  nam::Conv1x1 conv_dynamic(Channels, Channels, HasBias, Groups);

  auto it1 = weights.begin();
  auto it2 = weights.begin();
  conv_fixed.set_weights_(it1);
  conv_dynamic.set_weights_(it2);

  conv_fixed.SetMaxBufferSize(MaxFrames);
  conv_dynamic.SetMaxBufferSize(MaxFrames);

  Eigen::MatrixXf input(Channels, MaxFrames);
  for (int i = 0; i < Channels; i++)
    for (int j = 0; j < MaxFrames; j++)
      input(i, j) = dist(rng);

  conv_fixed.process_(input, MaxFrames);
  conv_dynamic.process_(input, MaxFrames);

  assert_matrices_equal(conv_fixed.GetOutput(), conv_dynamic.GetOutput(), MaxFrames);
}

void test_conv1x1_fully_fixed_no_bias()
{
  std::mt19937 rng(111);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  constexpr int Channels = 4;
  constexpr int MaxFrames = 64;
  constexpr int Groups = 1;
  constexpr bool HasBias = false;

  std::vector<float> weights;
  for (int i = 0; i < Channels * Channels; i++)
    weights.push_back(dist(rng));

  nam::Conv1x1FullyFixed<Channels, Channels, MaxFrames, Groups, HasBias> conv_fixed;
  nam::Conv1x1 conv_dynamic(Channels, Channels, HasBias, Groups);

  auto it1 = weights.begin();
  auto it2 = weights.begin();
  conv_fixed.set_weights_(it1);
  conv_dynamic.set_weights_(it2);

  conv_fixed.SetMaxBufferSize(MaxFrames);
  conv_dynamic.SetMaxBufferSize(MaxFrames);

  Eigen::MatrixXf input(Channels, MaxFrames);
  for (int i = 0; i < Channels; i++)
    for (int j = 0; j < MaxFrames; j++)
      input(i, j) = dist(rng);

  conv_fixed.process_(input, MaxFrames);
  conv_dynamic.process_(input, MaxFrames);

  assert_matrices_equal(conv_fixed.GetOutput(), conv_dynamic.GetOutput(), MaxFrames);
}

void test_conv1x1_fully_fixed_partial_buffer()
{
  std::mt19937 rng(222);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  constexpr int Channels = 4;
  constexpr int MaxFrames = 64;
  constexpr int Groups = 1;
  constexpr bool HasBias = true;
  constexpr int NumFrames = 32; // Half buffer

  std::vector<float> weights;
  for (int i = 0; i < Channels * Channels; i++)
    weights.push_back(dist(rng));
  for (int i = 0; i < Channels; i++)
    weights.push_back(dist(rng));

  nam::Conv1x1FullyFixed<Channels, Channels, MaxFrames, Groups, HasBias> conv_fixed;
  nam::Conv1x1 conv_dynamic(Channels, Channels, HasBias, Groups);

  auto it1 = weights.begin();
  auto it2 = weights.begin();
  conv_fixed.set_weights_(it1);
  conv_dynamic.set_weights_(it2);

  conv_fixed.SetMaxBufferSize(MaxFrames);
  conv_dynamic.SetMaxBufferSize(MaxFrames);

  Eigen::MatrixXf input(Channels, MaxFrames);
  for (int i = 0; i < Channels; i++)
    for (int j = 0; j < MaxFrames; j++)
      input(i, j) = dist(rng);

  conv_fixed.process_(input, NumFrames);
  conv_dynamic.process_(input, NumFrames);

  assert_matrices_equal(conv_fixed.GetOutput(), conv_dynamic.GetOutput(), NumFrames);
}

void test_conv1x1_fully_fixed_multiple_calls()
{
  std::mt19937 rng(333);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  constexpr int Channels = 4;
  constexpr int MaxFrames = 64;
  constexpr int Groups = 1;
  constexpr bool HasBias = true;

  std::vector<float> weights;
  for (int i = 0; i < Channels * Channels; i++)
    weights.push_back(dist(rng));
  for (int i = 0; i < Channels; i++)
    weights.push_back(dist(rng));

  nam::Conv1x1FullyFixed<Channels, Channels, MaxFrames, Groups, HasBias> conv_fixed;
  nam::Conv1x1 conv_dynamic(Channels, Channels, HasBias, Groups);

  auto it1 = weights.begin();
  auto it2 = weights.begin();
  conv_fixed.set_weights_(it1);
  conv_dynamic.set_weights_(it2);

  conv_fixed.SetMaxBufferSize(MaxFrames);
  conv_dynamic.SetMaxBufferSize(MaxFrames);

  for (int call = 0; call < 5; call++)
  {
    Eigen::MatrixXf input(Channels, MaxFrames);
    for (int i = 0; i < Channels; i++)
      for (int j = 0; j < MaxFrames; j++)
        input(i, j) = dist(rng);

    conv_fixed.process_(input, MaxFrames);
    conv_dynamic.process_(input, MaxFrames);

    assert_matrices_equal(conv_fixed.GetOutput(), conv_dynamic.GetOutput(), MaxFrames);
  }
}

// ============================================================================
// Conv1DFullyFixed Tests
// ============================================================================

void test_conv1d_fully_fixed_4ch_k3_64frames()
{
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  constexpr int Channels = 4;
  constexpr int KernelSize = 3;
  constexpr int MaxFrames = 64;
  constexpr int Groups = 1;
  constexpr bool HasBias = true;
  const int dilation = 1;

  std::vector<float> weights;
  for (int i = 0; i < Channels; i++)
    for (int j = 0; j < Channels; j++)
      for (int k = 0; k < KernelSize; k++)
        weights.push_back(dist(rng));
  for (int i = 0; i < Channels; i++)
    weights.push_back(dist(rng));

  nam::Conv1DFullyFixed<Channels, Channels, KernelSize, MaxFrames, Groups, HasBias> conv_fixed(dilation);
  nam::Conv1D conv_dynamic;
  conv_dynamic.set_size_(Channels, Channels, KernelSize, HasBias, dilation, Groups);

  auto it1 = weights.begin();
  auto it2 = weights.begin();
  conv_fixed.set_weights_(it1);
  conv_dynamic.set_weights_(it2);

  conv_fixed.SetMaxBufferSize(MaxFrames);
  conv_dynamic.SetMaxBufferSize(MaxFrames);

  Eigen::MatrixXf input(Channels, MaxFrames);
  for (int i = 0; i < Channels; i++)
    for (int j = 0; j < MaxFrames; j++)
      input(i, j) = dist(rng);

  conv_fixed.Process(input, MaxFrames);
  conv_dynamic.Process(input, MaxFrames);

  assert_matrices_equal(conv_fixed.GetOutput(), conv_dynamic.GetOutput(), MaxFrames);
}

void test_conv1d_fully_fixed_4ch_4groups()
{
  std::mt19937 rng(123);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  constexpr int Channels = 4;
  constexpr int KernelSize = 3;
  constexpr int MaxFrames = 64;
  constexpr int Groups = 4;
  constexpr bool HasBias = true;
  constexpr int PerGroup = Channels / Groups;
  const int dilation = 1;

  std::vector<float> weights;
  for (int g = 0; g < Groups; g++)
    for (int i = 0; i < PerGroup; i++)
      for (int j = 0; j < PerGroup; j++)
        for (int k = 0; k < KernelSize; k++)
          weights.push_back(dist(rng));
  for (int i = 0; i < Channels; i++)
    weights.push_back(dist(rng));

  nam::Conv1DFullyFixed<Channels, Channels, KernelSize, MaxFrames, Groups, HasBias> conv_fixed(dilation);
  nam::Conv1D conv_dynamic;
  conv_dynamic.set_size_(Channels, Channels, KernelSize, HasBias, dilation, Groups);

  auto it1 = weights.begin();
  auto it2 = weights.begin();
  conv_fixed.set_weights_(it1);
  conv_dynamic.set_weights_(it2);

  conv_fixed.SetMaxBufferSize(MaxFrames);
  conv_dynamic.SetMaxBufferSize(MaxFrames);

  Eigen::MatrixXf input(Channels, MaxFrames);
  for (int i = 0; i < Channels; i++)
    for (int j = 0; j < MaxFrames; j++)
      input(i, j) = dist(rng);

  conv_fixed.Process(input, MaxFrames);
  conv_dynamic.Process(input, MaxFrames);

  assert_matrices_equal(conv_fixed.GetOutput(), conv_dynamic.GetOutput(), MaxFrames);
}

void test_conv1d_fully_fixed_8ch_8groups()
{
  std::mt19937 rng(456);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  constexpr int Channels = 8;
  constexpr int KernelSize = 3;
  constexpr int MaxFrames = 128;
  constexpr int Groups = 8;
  constexpr bool HasBias = true;
  constexpr int PerGroup = Channels / Groups;
  const int dilation = 1;

  std::vector<float> weights;
  for (int g = 0; g < Groups; g++)
    for (int i = 0; i < PerGroup; i++)
      for (int j = 0; j < PerGroup; j++)
        for (int k = 0; k < KernelSize; k++)
          weights.push_back(dist(rng));
  for (int i = 0; i < Channels; i++)
    weights.push_back(dist(rng));

  nam::Conv1DFullyFixed<Channels, Channels, KernelSize, MaxFrames, Groups, HasBias> conv_fixed(dilation);
  nam::Conv1D conv_dynamic;
  conv_dynamic.set_size_(Channels, Channels, KernelSize, HasBias, dilation, Groups);

  auto it1 = weights.begin();
  auto it2 = weights.begin();
  conv_fixed.set_weights_(it1);
  conv_dynamic.set_weights_(it2);

  conv_fixed.SetMaxBufferSize(MaxFrames);
  conv_dynamic.SetMaxBufferSize(MaxFrames);

  Eigen::MatrixXf input(Channels, MaxFrames);
  for (int i = 0; i < Channels; i++)
    for (int j = 0; j < MaxFrames; j++)
      input(i, j) = dist(rng);

  conv_fixed.Process(input, MaxFrames);
  conv_dynamic.Process(input, MaxFrames);

  assert_matrices_equal(conv_fixed.GetOutput(), conv_dynamic.GetOutput(), MaxFrames);
}

void test_conv1d_fully_fixed_dilation2()
{
  std::mt19937 rng(789);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  constexpr int Channels = 4;
  constexpr int KernelSize = 3;
  constexpr int MaxFrames = 64;
  constexpr int Groups = 1;
  constexpr bool HasBias = true;
  const int dilation = 2;

  std::vector<float> weights;
  for (int i = 0; i < Channels; i++)
    for (int j = 0; j < Channels; j++)
      for (int k = 0; k < KernelSize; k++)
        weights.push_back(dist(rng));
  for (int i = 0; i < Channels; i++)
    weights.push_back(dist(rng));

  nam::Conv1DFullyFixed<Channels, Channels, KernelSize, MaxFrames, Groups, HasBias> conv_fixed(dilation);
  nam::Conv1D conv_dynamic;
  conv_dynamic.set_size_(Channels, Channels, KernelSize, HasBias, dilation, Groups);

  auto it1 = weights.begin();
  auto it2 = weights.begin();
  conv_fixed.set_weights_(it1);
  conv_dynamic.set_weights_(it2);

  conv_fixed.SetMaxBufferSize(MaxFrames);
  conv_dynamic.SetMaxBufferSize(MaxFrames);

  Eigen::MatrixXf input(Channels, MaxFrames);
  for (int i = 0; i < Channels; i++)
    for (int j = 0; j < MaxFrames; j++)
      input(i, j) = dist(rng);

  conv_fixed.Process(input, MaxFrames);
  conv_dynamic.Process(input, MaxFrames);

  assert_matrices_equal(conv_fixed.GetOutput(), conv_dynamic.GetOutput(), MaxFrames);
}

void test_conv1d_fully_fixed_dilation8()
{
  std::mt19937 rng(111);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  constexpr int Channels = 4;
  constexpr int KernelSize = 3;
  constexpr int MaxFrames = 128;
  constexpr int Groups = 1;
  constexpr bool HasBias = true;
  const int dilation = 8;

  std::vector<float> weights;
  for (int i = 0; i < Channels; i++)
    for (int j = 0; j < Channels; j++)
      for (int k = 0; k < KernelSize; k++)
        weights.push_back(dist(rng));
  for (int i = 0; i < Channels; i++)
    weights.push_back(dist(rng));

  nam::Conv1DFullyFixed<Channels, Channels, KernelSize, MaxFrames, Groups, HasBias> conv_fixed(dilation);
  nam::Conv1D conv_dynamic;
  conv_dynamic.set_size_(Channels, Channels, KernelSize, HasBias, dilation, Groups);

  auto it1 = weights.begin();
  auto it2 = weights.begin();
  conv_fixed.set_weights_(it1);
  conv_dynamic.set_weights_(it2);

  conv_fixed.SetMaxBufferSize(MaxFrames);
  conv_dynamic.SetMaxBufferSize(MaxFrames);

  Eigen::MatrixXf input(Channels, MaxFrames);
  for (int i = 0; i < Channels; i++)
    for (int j = 0; j < MaxFrames; j++)
      input(i, j) = dist(rng);

  conv_fixed.Process(input, MaxFrames);
  conv_dynamic.Process(input, MaxFrames);

  assert_matrices_equal(conv_fixed.GetOutput(), conv_dynamic.GetOutput(), MaxFrames);
}

void test_conv1d_fully_fixed_no_bias()
{
  std::mt19937 rng(222);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  constexpr int Channels = 4;
  constexpr int KernelSize = 3;
  constexpr int MaxFrames = 64;
  constexpr int Groups = 1;
  constexpr bool HasBias = false;
  const int dilation = 1;

  std::vector<float> weights;
  for (int i = 0; i < Channels; i++)
    for (int j = 0; j < Channels; j++)
      for (int k = 0; k < KernelSize; k++)
        weights.push_back(dist(rng));

  nam::Conv1DFullyFixed<Channels, Channels, KernelSize, MaxFrames, Groups, HasBias> conv_fixed(dilation);
  nam::Conv1D conv_dynamic;
  conv_dynamic.set_size_(Channels, Channels, KernelSize, HasBias, dilation, Groups);

  auto it1 = weights.begin();
  auto it2 = weights.begin();
  conv_fixed.set_weights_(it1);
  conv_dynamic.set_weights_(it2);

  conv_fixed.SetMaxBufferSize(MaxFrames);
  conv_dynamic.SetMaxBufferSize(MaxFrames);

  Eigen::MatrixXf input(Channels, MaxFrames);
  for (int i = 0; i < Channels; i++)
    for (int j = 0; j < MaxFrames; j++)
      input(i, j) = dist(rng);

  conv_fixed.Process(input, MaxFrames);
  conv_dynamic.Process(input, MaxFrames);

  assert_matrices_equal(conv_fixed.GetOutput(), conv_dynamic.GetOutput(), MaxFrames);
}

void test_conv1d_fully_fixed_multiple_calls()
{
  std::mt19937 rng(333);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  constexpr int Channels = 4;
  constexpr int KernelSize = 3;
  constexpr int MaxFrames = 64;
  constexpr int Groups = 1;
  constexpr bool HasBias = true;
  const int dilation = 1;

  std::vector<float> weights;
  for (int i = 0; i < Channels; i++)
    for (int j = 0; j < Channels; j++)
      for (int k = 0; k < KernelSize; k++)
        weights.push_back(dist(rng));
  for (int i = 0; i < Channels; i++)
    weights.push_back(dist(rng));

  nam::Conv1DFullyFixed<Channels, Channels, KernelSize, MaxFrames, Groups, HasBias> conv_fixed(dilation);
  nam::Conv1D conv_dynamic;
  conv_dynamic.set_size_(Channels, Channels, KernelSize, HasBias, dilation, Groups);

  auto it1 = weights.begin();
  auto it2 = weights.begin();
  conv_fixed.set_weights_(it1);
  conv_dynamic.set_weights_(it2);

  conv_fixed.SetMaxBufferSize(MaxFrames);
  conv_dynamic.SetMaxBufferSize(MaxFrames);

  // Multiple calls - tests history management
  for (int call = 0; call < 10; call++)
  {
    Eigen::MatrixXf input(Channels, MaxFrames);
    for (int i = 0; i < Channels; i++)
      for (int j = 0; j < MaxFrames; j++)
        input(i, j) = dist(rng);

    conv_fixed.Process(input, MaxFrames);
    conv_dynamic.Process(input, MaxFrames);

    assert_matrices_equal(conv_fixed.GetOutput(), conv_dynamic.GetOutput(), MaxFrames);
  }
}

void test_conv1d_fully_fixed_multiple_calls_dilation4()
{
  std::mt19937 rng(444);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  constexpr int Channels = 4;
  constexpr int KernelSize = 3;
  constexpr int MaxFrames = 64;
  constexpr int Groups = 1;
  constexpr bool HasBias = true;
  const int dilation = 4;

  std::vector<float> weights;
  for (int i = 0; i < Channels; i++)
    for (int j = 0; j < Channels; j++)
      for (int k = 0; k < KernelSize; k++)
        weights.push_back(dist(rng));
  for (int i = 0; i < Channels; i++)
    weights.push_back(dist(rng));

  nam::Conv1DFullyFixed<Channels, Channels, KernelSize, MaxFrames, Groups, HasBias> conv_fixed(dilation);
  nam::Conv1D conv_dynamic;
  conv_dynamic.set_size_(Channels, Channels, KernelSize, HasBias, dilation, Groups);

  auto it1 = weights.begin();
  auto it2 = weights.begin();
  conv_fixed.set_weights_(it1);
  conv_dynamic.set_weights_(it2);

  conv_fixed.SetMaxBufferSize(MaxFrames);
  conv_dynamic.SetMaxBufferSize(MaxFrames);

  for (int call = 0; call < 10; call++)
  {
    Eigen::MatrixXf input(Channels, MaxFrames);
    for (int i = 0; i < Channels; i++)
      for (int j = 0; j < MaxFrames; j++)
        input(i, j) = dist(rng);

    conv_fixed.Process(input, MaxFrames);
    conv_dynamic.Process(input, MaxFrames);

    assert_matrices_equal(conv_fixed.GetOutput(), conv_dynamic.GetOutput(), MaxFrames);
  }
}

void test_conv1d_fully_fixed_varying_buffer_sizes()
{
  std::mt19937 rng(555);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  constexpr int Channels = 4;
  constexpr int KernelSize = 3;
  constexpr int MaxFrames = 64;
  constexpr int Groups = 1;
  constexpr bool HasBias = true;
  const int dilation = 2;

  std::vector<float> weights;
  for (int i = 0; i < Channels; i++)
    for (int j = 0; j < Channels; j++)
      for (int k = 0; k < KernelSize; k++)
        weights.push_back(dist(rng));
  for (int i = 0; i < Channels; i++)
    weights.push_back(dist(rng));

  nam::Conv1DFullyFixed<Channels, Channels, KernelSize, MaxFrames, Groups, HasBias> conv_fixed(dilation);
  nam::Conv1D conv_dynamic;
  conv_dynamic.set_size_(Channels, Channels, KernelSize, HasBias, dilation, Groups);

  auto it1 = weights.begin();
  auto it2 = weights.begin();
  conv_fixed.set_weights_(it1);
  conv_dynamic.set_weights_(it2);

  conv_fixed.SetMaxBufferSize(MaxFrames);
  conv_dynamic.SetMaxBufferSize(MaxFrames);

  // Varying sizes to stress test history management
  int sizes[] = {64, 32, 16, 64, 32, 8, 64};
  for (int num_frames : sizes)
  {
    Eigen::MatrixXf input(Channels, MaxFrames);
    for (int i = 0; i < Channels; i++)
      for (int j = 0; j < MaxFrames; j++)
        input(i, j) = dist(rng);

    conv_fixed.Process(input, num_frames);
    conv_dynamic.Process(input, num_frames);

    assert_matrices_equal(conv_fixed.GetOutput(), conv_dynamic.GetOutput(), num_frames);
  }
}

void test_conv1d_fully_fixed_kernel4()
{
  std::mt19937 rng(666);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  constexpr int Channels = 4;
  constexpr int KernelSize = 4;
  constexpr int MaxFrames = 64;
  constexpr int Groups = 1;
  constexpr bool HasBias = true;
  const int dilation = 1;

  std::vector<float> weights;
  for (int i = 0; i < Channels; i++)
    for (int j = 0; j < Channels; j++)
      for (int k = 0; k < KernelSize; k++)
        weights.push_back(dist(rng));
  for (int i = 0; i < Channels; i++)
    weights.push_back(dist(rng));

  nam::Conv1DFullyFixed<Channels, Channels, KernelSize, MaxFrames, Groups, HasBias> conv_fixed(dilation);
  nam::Conv1D conv_dynamic;
  conv_dynamic.set_size_(Channels, Channels, KernelSize, HasBias, dilation, Groups);

  auto it1 = weights.begin();
  auto it2 = weights.begin();
  conv_fixed.set_weights_(it1);
  conv_dynamic.set_weights_(it2);

  conv_fixed.SetMaxBufferSize(MaxFrames);
  conv_dynamic.SetMaxBufferSize(MaxFrames);

  Eigen::MatrixXf input(Channels, MaxFrames);
  for (int i = 0; i < Channels; i++)
    for (int j = 0; j < MaxFrames; j++)
      input(i, j) = dist(rng);

  conv_fixed.Process(input, MaxFrames);
  conv_dynamic.Process(input, MaxFrames);

  assert_matrices_equal(conv_fixed.GetOutput(), conv_dynamic.GetOutput(), MaxFrames);
}

} // namespace test_fully_fixed_correctness
