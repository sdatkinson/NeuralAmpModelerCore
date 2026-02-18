// Tests for correct handling of non-contiguous Eigen block expressions
// (e.g. topRows()) in inline GEMM paths.
//
// When a matrix expression like matrix.topRows(n) is passed to a function
// that accesses raw pointers via .data(), the outerStride() (distance between
// columns) may be larger than rows(). Code that assumes stride == rows will
// read/write wrong memory locations.

#include <Eigen/Dense>
#include <cassert>
#include <cmath>
#include <memory>
#include <vector>

#include "NAM/activations.h"
#include "NAM/dsp.h"
#include "NAM/film.h"
#include "NAM/gating_activations.h"

namespace test_noncontiguous_blocks
{

// Helper: create identity Conv1x1 weights (identity matrix + zero bias)
static std::vector<float> make_identity_weights(int channels, bool bias)
{
  std::vector<float> weights(channels * channels + (bias ? channels : 0), 0.0f);
  // Column-major identity matrix
  for (int i = 0; i < channels; i++)
    weights[i * channels + i] = 1.0f;
  return weights;
}

// Helper: create scaled Conv1x1 weights (diagonal scale matrix + zero bias)
static std::vector<float> make_scale_weights(int in_ch, int out_ch, float scale, bool bias)
{
  std::vector<float> weights(out_ch * in_ch + (bias ? out_ch : 0), 0.0f);
  // Set all weight elements to scale (for simple testing)
  for (int o = 0; o < out_ch; o++)
    for (int i = 0; i < in_ch; i++)
      weights[i * out_ch + o] = (i == o) ? scale : 0.0f;
  return weights;
}

// ============================================================
// Conv1x1::process_() with non-contiguous input (topRows)
// ============================================================

void test_conv1x1_process_toprows()
{
  // Create a matrix with 8 rows, but only pass topRows(4) to Conv1x1
  // This simulates the gated activation path in wavenet.cpp
  const int total_rows = 8;
  const int bottleneck = 4;
  const int num_frames = 3;

  // Create Conv1x1: 4 in -> 4 out, identity weights
  nam::Conv1x1 conv(bottleneck, bottleneck, /*bias=*/false);
  auto weights = make_identity_weights(bottleneck, false);
  auto it = weights.begin();
  conv.set_weights_(it);
  conv.SetMaxBufferSize(64);

  // Create full matrix with known values
  Eigen::MatrixXf full_matrix(total_rows, num_frames);
  for (int c = 0; c < total_rows; c++)
    for (int f = 0; f < num_frames; f++)
      full_matrix(c, f) = (float)(c * 10 + f);

  // Process only topRows(bottleneck) - this has outerStride = 8, rows = 4
  auto top_block = full_matrix.topRows(bottleneck);
  assert(top_block.outerStride() == total_rows); // Verify non-contiguous
  assert(top_block.rows() == bottleneck);

  conv.process_(top_block, num_frames);
  const auto& output = conv.GetOutput();

  // With identity weights, output should equal the top rows exactly
  for (int c = 0; c < bottleneck; c++)
  {
    for (int f = 0; f < num_frames; f++)
    {
      const float expected = full_matrix(c, f);
      const float actual = output(c, f);
      assert(std::abs(actual - expected) < 1e-6f);
    }
  }
}

void test_conv1x1_process_toprows_with_bias()
{
  const int total_rows = 6;
  const int bottleneck = 3;
  const int num_frames = 4;

  nam::Conv1x1 conv(bottleneck, bottleneck, /*bias=*/true);
  auto weights = make_identity_weights(bottleneck, false);
  // Add bias values
  weights.push_back(10.0f);
  weights.push_back(20.0f);
  weights.push_back(30.0f);
  auto it = weights.begin();
  conv.set_weights_(it);
  conv.SetMaxBufferSize(64);

  Eigen::MatrixXf full_matrix(total_rows, num_frames);
  for (int c = 0; c < total_rows; c++)
    for (int f = 0; f < num_frames; f++)
      full_matrix(c, f) = (float)(c + 1);

  conv.process_(full_matrix.topRows(bottleneck), num_frames);
  const auto& output = conv.GetOutput();

  const float biases[] = {10.0f, 20.0f, 30.0f};
  for (int c = 0; c < bottleneck; c++)
  {
    for (int f = 0; f < num_frames; f++)
    {
      const float expected = full_matrix(c, f) + biases[c];
      assert(std::abs(output(c, f) - expected) < 1e-6f);
    }
  }
}

void test_conv1x1_process_toprows_2x2()
{
  // Test specific 2x2 specialized path with non-contiguous input
  const int total_rows = 4; // doubled for gating
  const int bottleneck = 2;
  const int num_frames = 3;

  nam::Conv1x1 conv(bottleneck, bottleneck, /*bias=*/false);
  // Weight: [[2, 0], [0, 3]] (column-major: [2, 0, 0, 3])
  std::vector<float> weights = {2.0f, 0.0f, 0.0f, 3.0f};
  auto it = weights.begin();
  conv.set_weights_(it);
  conv.SetMaxBufferSize(64);

  Eigen::MatrixXf full_matrix(total_rows, num_frames);
  full_matrix << 1.0f, 2.0f, 3.0f,   // row 0 (top, used)
                 4.0f, 5.0f, 6.0f,   // row 1 (top, used)
                 99.0f, 99.0f, 99.0f, // row 2 (bottom, NOT used)
                 99.0f, 99.0f, 99.0f; // row 3 (bottom, NOT used)

  conv.process_(full_matrix.topRows(bottleneck), num_frames);
  const auto& output = conv.GetOutput();

  // output = [[2,0],[0,3]] * topRows(2)
  // Frame 0: [2*1, 3*4] = [2, 12]
  // Frame 1: [2*2, 3*5] = [4, 15]
  // Frame 2: [2*3, 3*6] = [6, 18]
  assert(std::abs(output(0, 0) - 2.0f) < 1e-6f);
  assert(std::abs(output(1, 0) - 12.0f) < 1e-6f);
  assert(std::abs(output(0, 1) - 4.0f) < 1e-6f);
  assert(std::abs(output(1, 1) - 15.0f) < 1e-6f);
  assert(std::abs(output(0, 2) - 6.0f) < 1e-6f);
  assert(std::abs(output(1, 2) - 18.0f) < 1e-6f);
}

void test_conv1x1_process_toprows_4x4()
{
  // Test specific 4x4 specialized path with non-contiguous input
  const int total_rows = 8;
  const int bottleneck = 4;
  const int num_frames = 2;

  nam::Conv1x1 conv(bottleneck, bottleneck, /*bias=*/false);
  auto weights = make_identity_weights(bottleneck, false);
  auto it = weights.begin();
  conv.set_weights_(it);
  conv.SetMaxBufferSize(64);

  Eigen::MatrixXf full_matrix(total_rows, num_frames);
  // Top 4 rows: known values; bottom 4 rows: poison values
  for (int c = 0; c < bottleneck; c++)
    for (int f = 0; f < num_frames; f++)
      full_matrix(c, f) = (float)(c * 10 + f + 1);
  for (int c = bottleneck; c < total_rows; c++)
    for (int f = 0; f < num_frames; f++)
      full_matrix(c, f) = -999.0f; // poison

  conv.process_(full_matrix.topRows(bottleneck), num_frames);
  const auto& output = conv.GetOutput();

  for (int c = 0; c < bottleneck; c++)
    for (int f = 0; f < num_frames; f++)
      assert(std::abs(output(c, f) - full_matrix(c, f)) < 1e-6f);
}

// Verify Conv1x1 with topRows matches result from contiguous copy
void test_conv1x1_toprows_matches_contiguous()
{
  const int total_rows = 8;
  const int bottleneck = 4;
  const int num_frames = 5;

  nam::Conv1x1 conv(bottleneck, 2, /*bias=*/true);
  // Random-ish weights for 4->2 with bias
  std::vector<float> weights = {
    1.0f, 0.5f, -1.0f, 0.5f, 2.0f, -0.5f, 0.0f, 1.5f, // 2x4 weights (column-major)
    3.0f, -2.0f // biases
  };
  auto it = weights.begin();
  conv.set_weights_(it);
  conv.SetMaxBufferSize(64);

  Eigen::MatrixXf full_matrix(total_rows, num_frames);
  full_matrix.setRandom();

  // Reference: copy topRows to contiguous matrix, process
  Eigen::MatrixXf contiguous_input = full_matrix.topRows(bottleneck).eval();
  conv.process_(contiguous_input, num_frames);
  Eigen::MatrixXf expected = conv.GetOutput().leftCols(num_frames).eval();

  // Test: process non-contiguous topRows directly
  conv.process_(full_matrix.topRows(bottleneck), num_frames);
  const auto& actual = conv.GetOutput();

  for (int c = 0; c < 2; c++)
    for (int f = 0; f < num_frames; f++)
      assert(std::abs(actual(c, f) - expected(c, f)) < 1e-5f);
}

// ============================================================
// FiLM::Process() with non-contiguous input (topRows)
// ============================================================

void test_film_process_toprows_with_shift()
{
  const int condition_dim = 2;
  const int input_dim = 3;
  const int total_rows = 6; // 2x input_dim, simulating gated _z matrix

  nam::FiLM film(condition_dim, input_dim, /*shift=*/true);
  film.SetMaxBufferSize(64);

  // Configure Conv1x1 with zero weights, fixed biases for scale/shift
  std::vector<float> weights((2 * input_dim) * condition_dim + (2 * input_dim), 0.0f);
  const int bias_offset = (2 * input_dim) * condition_dim;
  weights[bias_offset + 0] = 2.0f;  // scale[0]
  weights[bias_offset + 1] = -1.0f; // scale[1]
  weights[bias_offset + 2] = 0.5f;  // scale[2]
  weights[bias_offset + 3] = 10.0f; // shift[0]
  weights[bias_offset + 4] = -5.0f; // shift[1]
  weights[bias_offset + 5] = 3.0f;  // shift[2]
  auto it = weights.begin();
  film.set_weights_(it);

  const int num_frames = 4;

  // Create a wider matrix and pass topRows as input
  Eigen::MatrixXf full_matrix(total_rows, num_frames);
  for (int c = 0; c < total_rows; c++)
    for (int f = 0; f < num_frames; f++)
      full_matrix(c, f) = (float)(c + 1) * (f + 1);

  Eigen::MatrixXf condition(condition_dim, num_frames);
  condition.setRandom();

  // Process with non-contiguous topRows(input_dim)
  auto top_block = full_matrix.topRows(input_dim);
  assert(top_block.outerStride() == total_rows); // Verify non-contiguous

  film.Process(top_block, condition, num_frames);
  const auto& output = film.GetOutput();

  const float scales[] = {2.0f, -1.0f, 0.5f};
  const float shifts[] = {10.0f, -5.0f, 3.0f};
  for (int c = 0; c < input_dim; c++)
  {
    for (int f = 0; f < num_frames; f++)
    {
      const float expected = full_matrix(c, f) * scales[c] + shifts[c];
      assert(std::abs(output(c, f) - expected) < 1e-5f);
    }
  }
}

void test_film_process_toprows_scale_only()
{
  const int condition_dim = 2;
  const int input_dim = 4;
  const int total_rows = 8;

  nam::FiLM film(condition_dim, input_dim, /*shift=*/false);
  film.SetMaxBufferSize(64);

  std::vector<float> weights(input_dim * condition_dim + input_dim, 0.0f);
  const int bias_offset = input_dim * condition_dim;
  weights[bias_offset + 0] = 2.0f;
  weights[bias_offset + 1] = 3.0f;
  weights[bias_offset + 2] = -1.0f;
  weights[bias_offset + 3] = 0.5f;
  auto it = weights.begin();
  film.set_weights_(it);

  const int num_frames = 3;
  Eigen::MatrixXf full_matrix(total_rows, num_frames);
  full_matrix.setRandom();

  Eigen::MatrixXf condition(condition_dim, num_frames);
  condition.setRandom();

  film.Process(full_matrix.topRows(input_dim), condition, num_frames);
  const auto& output = film.GetOutput();

  const float scales[] = {2.0f, 3.0f, -1.0f, 0.5f};
  for (int c = 0; c < input_dim; c++)
    for (int f = 0; f < num_frames; f++)
    {
      const float expected = full_matrix(c, f) * scales[c];
      assert(std::abs(output(c, f) - expected) < 1e-5f);
    }
}

// Verify FiLM with topRows matches result from contiguous copy
void test_film_toprows_matches_contiguous()
{
  const int condition_dim = 2;
  const int input_dim = 3;
  const int total_rows = 6;
  const int num_frames = 4;

  nam::FiLM film(condition_dim, input_dim, /*shift=*/true);
  film.SetMaxBufferSize(64);

  std::vector<float> weights((2 * input_dim) * condition_dim + (2 * input_dim), 0.0f);
  const int bias_offset = (2 * input_dim) * condition_dim;
  weights[bias_offset + 0] = 2.0f;
  weights[bias_offset + 1] = -1.0f;
  weights[bias_offset + 2] = 0.5f;
  weights[bias_offset + 3] = 10.0f;
  weights[bias_offset + 4] = -5.0f;
  weights[bias_offset + 5] = 3.0f;
  auto it = weights.begin();
  film.set_weights_(it);

  Eigen::MatrixXf full_matrix(total_rows, num_frames);
  full_matrix.setRandom();
  Eigen::MatrixXf condition(condition_dim, num_frames);
  condition.setRandom();

  // Reference: contiguous copy
  Eigen::MatrixXf contiguous = full_matrix.topRows(input_dim).eval();
  film.Process(contiguous, condition, num_frames);
  Eigen::MatrixXf expected = film.GetOutput().leftCols(num_frames).eval();

  // Test: non-contiguous topRows
  film.Process(full_matrix.topRows(input_dim), condition, num_frames);
  const auto& actual = film.GetOutput();

  for (int c = 0; c < input_dim; c++)
    for (int f = 0; f < num_frames; f++)
      assert(std::abs(actual(c, f) - expected(c, f)) < 1e-6f);
}

// ============================================================
// FiLM::Process_() (in-place) with non-contiguous input (topRows)
// ============================================================

void test_film_process_inplace_toprows()
{
  const int condition_dim = 2;
  const int input_dim = 3;
  const int total_rows = 6;
  const int num_frames = 4;

  nam::FiLM film(condition_dim, input_dim, /*shift=*/true);
  film.SetMaxBufferSize(64);

  std::vector<float> weights((2 * input_dim) * condition_dim + (2 * input_dim), 0.0f);
  const int bias_offset = (2 * input_dim) * condition_dim;
  weights[bias_offset + 0] = 2.0f;
  weights[bias_offset + 1] = -1.0f;
  weights[bias_offset + 2] = 0.5f;
  weights[bias_offset + 3] = 10.0f;
  weights[bias_offset + 4] = -5.0f;
  weights[bias_offset + 5] = 3.0f;
  auto it = weights.begin();
  film.set_weights_(it);

  Eigen::MatrixXf full_matrix(total_rows, num_frames);
  for (int c = 0; c < total_rows; c++)
    for (int f = 0; f < num_frames; f++)
      full_matrix(c, f) = (float)(c + 1) * (f + 1);
  Eigen::MatrixXf original = full_matrix;

  Eigen::MatrixXf condition(condition_dim, num_frames);
  condition.setRandom();

  // In-place process on topRows block
  film.Process_(full_matrix.topRows(input_dim), condition, num_frames);

  const float scales[] = {2.0f, -1.0f, 0.5f};
  const float shifts[] = {10.0f, -5.0f, 3.0f};
  for (int c = 0; c < input_dim; c++)
    for (int f = 0; f < num_frames; f++)
    {
      const float expected = original(c, f) * scales[c] + shifts[c];
      assert(std::abs(full_matrix(c, f) - expected) < 1e-5f);
    }

  // Bottom rows should be unchanged
  for (int c = input_dim; c < total_rows; c++)
    for (int f = 0; f < num_frames; f++)
      assert(std::abs(full_matrix(c, f) - original(c, f)) < 1e-6f);
}

// ============================================================
// GatingActivation with non-contiguous output (topRows)
// ============================================================

// Helper to create a non-owning shared_ptr for stack-allocated activations in tests
template <typename T>
static nam::activations::Activation::Ptr make_test_ptr(T& activation)
{
  return nam::activations::Activation::Ptr(&activation, [](nam::activations::Activation*) {});
}

void test_gating_output_toprows()
{
  // Simulate wavenet.cpp pattern:
  // input_block = _z.leftCols(num_frames)     (contiguous, 2*bottleneck rows)
  // output_block = _z.topRows(bottleneck).leftCols(num_frames) (non-contiguous output)
  const int bottleneck = 3;
  const int total_rows = 2 * bottleneck;
  const int num_frames = 4;

  nam::activations::ActivationIdentity identity_act;
  nam::activations::ActivationSigmoid sigmoid_act;
  nam::gating_activations::GatingActivation gating(make_test_ptr(identity_act), make_test_ptr(sigmoid_act), bottleneck);

  // Input: contiguous (2*bottleneck x num_frames)
  Eigen::MatrixXf input(total_rows, num_frames);
  for (int c = 0; c < total_rows; c++)
    for (int f = 0; f < num_frames; f++)
      input(c, f) = (float)(c + 1) * 0.5f;

  // Output goes into topRows of a larger matrix
  Eigen::MatrixXf output_matrix(total_rows, num_frames);
  output_matrix.setConstant(-999.0f); // poison
  auto output_block = output_matrix.topRows(bottleneck).leftCols(num_frames);

  gating.apply(input, output_block);

  // Verify output values are correct (not reading poison from wrong stride)
  for (int f = 0; f < num_frames; f++)
  {
    for (int c = 0; c < bottleneck; c++)
    {
      const float input_val = input(c, f);               // identity activation
      const float gate_val = 1.0f / (1.0f + expf(-input(c + bottleneck, f))); // sigmoid
      const float expected = input_val * gate_val;
      assert(std::abs(output_matrix(c, f) - expected) < 1e-5f);
    }
  }

  // Bottom rows should still be poison (untouched)
  for (int c = bottleneck; c < total_rows; c++)
    for (int f = 0; f < num_frames; f++)
      assert(std::abs(output_matrix(c, f) - (-999.0f)) < 1e-6f);
}

void test_gating_toprows_matches_contiguous()
{
  const int bottleneck = 2;
  const int total_rows = 2 * bottleneck;
  const int num_frames = 3;

  nam::activations::ActivationReLU relu_act;
  nam::activations::ActivationSigmoid sigmoid_act;
  nam::gating_activations::GatingActivation gating(make_test_ptr(relu_act), make_test_ptr(sigmoid_act), bottleneck);

  Eigen::MatrixXf input(total_rows, num_frames);
  input.setRandom();

  // Reference: contiguous output
  Eigen::MatrixXf contiguous_output(bottleneck, num_frames);
  gating.apply(input, contiguous_output);

  // Test: non-contiguous output (topRows of larger matrix)
  Eigen::MatrixXf full_output(total_rows, num_frames);
  full_output.setZero();
  auto output_block = full_output.topRows(bottleneck).leftCols(num_frames);
  gating.apply(input, output_block);

  for (int c = 0; c < bottleneck; c++)
    for (int f = 0; f < num_frames; f++)
      assert(std::abs(full_output(c, f) - contiguous_output(c, f)) < 1e-6f);
}

// ============================================================
// BlendingActivation with non-contiguous output (topRows)
// ============================================================

void test_blending_output_toprows()
{
  const int bottleneck = 2;
  const int total_rows = 2 * bottleneck;
  const int num_frames = 3;

  nam::activations::ActivationIdentity identity_act;
  nam::activations::ActivationSigmoid sigmoid_act;
  nam::gating_activations::BlendingActivation blending(
    make_test_ptr(identity_act), make_test_ptr(sigmoid_act), bottleneck);

  Eigen::MatrixXf input(total_rows, num_frames);
  input.setRandom();

  // Reference: contiguous output
  Eigen::MatrixXf contiguous_output(bottleneck, num_frames);
  blending.apply(input, contiguous_output);

  // Test: non-contiguous output (topRows of larger matrix)
  Eigen::MatrixXf full_output(total_rows, num_frames);
  full_output.setConstant(-999.0f);
  auto output_block = full_output.topRows(bottleneck).leftCols(num_frames);
  blending.apply(input, output_block);

  for (int c = 0; c < bottleneck; c++)
    for (int f = 0; f < num_frames; f++)
      assert(std::abs(full_output(c, f) - contiguous_output(c, f)) < 1e-5f);

  // Bottom rows should be untouched
  for (int c = bottleneck; c < total_rows; c++)
    for (int f = 0; f < num_frames; f++)
      assert(std::abs(full_output(c, f) - (-999.0f)) < 1e-6f);
}

} // namespace test_noncontiguous_blocks
