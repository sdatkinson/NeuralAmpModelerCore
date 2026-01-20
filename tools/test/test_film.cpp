// Tests for FiLM

#include <Eigen/Dense>
#include <cassert>
#include <cmath>
#include <vector>

#include "NAM/film.h"

namespace test_film
{
void test_set_max_buffer_size()
{
  const int condition_dim = 2;
  const int input_dim = 3;
  nam::FiLM film(condition_dim, input_dim, /*shift=*/true);

  const int maxBufferSize = 128;
  film.SetMaxBufferSize(maxBufferSize);

  const auto out = film.GetOutput();
  assert(out.rows() == input_dim);
  assert(out.cols() == maxBufferSize);
}

void test_process_bias_only()
{
  const int condition_dim = 2;
  const int input_dim = 3;
  nam::FiLM film(condition_dim, input_dim, /*shift=*/true);

  const int maxBufferSize = 64;
  film.SetMaxBufferSize(maxBufferSize);

  // Configure the internal Conv1x1 (condition_dim -> 2*input_dim) to have:
  // - all-zero weights
  // - fixed biases so that scale/shift are constants
  //
  // Layout for Conv1x1 weights when groups=1:
  // - matrix weights: (2*input_dim * condition_dim)
  // - bias: (2*input_dim)
  std::vector<float> weights;
  weights.resize((2 * input_dim) * condition_dim + (2 * input_dim), 0.0f);

  // biases: [scale(0..input_dim-1), shift(0..input_dim-1)]
  const float scale0 = 2.0f;
  const float scale1 = -1.0f;
  const float scale2 = 0.5f;
  const float shift0 = 10.0f;
  const float shift1 = -20.0f;
  const float shift2 = 3.0f;

  const int bias_offset = (2 * input_dim) * condition_dim;
  weights[bias_offset + 0] = scale0;
  weights[bias_offset + 1] = scale1;
  weights[bias_offset + 2] = scale2;
  weights[bias_offset + 3] = shift0;
  weights[bias_offset + 4] = shift1;
  weights[bias_offset + 5] = shift2;

  auto it = weights.begin();
  film.set_weights_(it);
  assert(it == weights.end());

  const int num_frames = 4;
  Eigen::MatrixXf input(input_dim, num_frames);
  // Make each channel distinct, and vary over frames.
  input << 1.0f, 2.0f, 3.0f, 4.0f, //
    -1.0f, -2.0f, -3.0f, -4.0f, //
    0.25f, 0.5f, 0.75f, 1.0f;

  Eigen::MatrixXf condition(condition_dim, num_frames);
  condition.setRandom(); // doesn't matter because weights are zero

  film.Process(input, condition, num_frames);
  const auto out = film.GetOutput().leftCols(num_frames);

  // Expected: output = input * scale + shift (elementwise)
  const float scales[3] = {scale0, scale1, scale2};
  const float shifts[3] = {shift0, shift1, shift2};
  for (int c = 0; c < input_dim; c++)
  {
    for (int t = 0; t < num_frames; t++)
    {
      const float expected = input(c, t) * scales[c] + shifts[c];
      assert(std::abs(out(c, t) - expected) < 1e-6f);
    }
  }
}

void test_process_scale_only()
{
  const int condition_dim = 2;
  const int input_dim = 3;
  nam::FiLM film(condition_dim, input_dim, /*shift=*/false);

  const int maxBufferSize = 64;
  film.SetMaxBufferSize(maxBufferSize);

  // Internal Conv1x1 is (condition_dim -> input_dim) in scale-only mode.
  // We'll use all-zero weights and biases that define the scale.
  std::vector<float> weights;
  weights.resize(input_dim * condition_dim + input_dim, 0.0f);

  const float scale0 = 2.0f;
  const float scale1 = -1.0f;
  const float scale2 = 0.5f;

  const int bias_offset = input_dim * condition_dim;
  weights[bias_offset + 0] = scale0;
  weights[bias_offset + 1] = scale1;
  weights[bias_offset + 2] = scale2;

  auto it = weights.begin();
  film.set_weights_(it);
  assert(it == weights.end());

  const int num_frames = 4;
  Eigen::MatrixXf input(input_dim, num_frames);
  input << 1.0f, 2.0f, 3.0f, 4.0f, //
    -1.0f, -2.0f, -3.0f, -4.0f, //
    0.25f, 0.5f, 0.75f, 1.0f;

  Eigen::MatrixXf condition(condition_dim, num_frames);
  condition.setRandom(); // doesn't matter because weights are zero

  film.Process(input, condition, num_frames);
  const auto out = film.GetOutput().leftCols(num_frames);

  const float scales[3] = {scale0, scale1, scale2};
  for (int c = 0; c < input_dim; c++)
  {
    for (int t = 0; t < num_frames; t++)
    {
      const float expected = input(c, t) * scales[c];
      assert(std::abs(out(c, t) - expected) < 1e-6f);
    }
  }
}

void test_process_inplace_with_shift()
{
  const int condition_dim = 2;
  const int input_dim = 3;
  nam::FiLM film(condition_dim, input_dim, /*shift=*/true);

  const int maxBufferSize = 64;
  film.SetMaxBufferSize(maxBufferSize);

  // Configure the internal Conv1x1 with zero weights and fixed biases
  std::vector<float> weights;
  weights.resize((2 * input_dim) * condition_dim + (2 * input_dim), 0.0f);

  const float scale0 = 2.0f;
  const float scale1 = -1.0f;
  const float scale2 = 0.5f;
  const float shift0 = 10.0f;
  const float shift1 = -20.0f;
  const float shift2 = 3.0f;

  const int bias_offset = (2 * input_dim) * condition_dim;
  weights[bias_offset + 0] = scale0;
  weights[bias_offset + 1] = scale1;
  weights[bias_offset + 2] = scale2;
  weights[bias_offset + 3] = shift0;
  weights[bias_offset + 4] = shift1;
  weights[bias_offset + 5] = shift2;

  auto it = weights.begin();
  film.set_weights_(it);
  assert(it == weights.end());

  const int num_frames = 4;
  Eigen::MatrixXf input(input_dim, num_frames);
  input << 1.0f, 2.0f, 3.0f, 4.0f, //
    -1.0f, -2.0f, -3.0f, -4.0f, //
    0.25f, 0.5f, 0.75f, 1.0f;

  // Keep a copy of the original input for comparison
  const Eigen::MatrixXf input_original = input;

  Eigen::MatrixXf condition(condition_dim, num_frames);
  condition.setRandom(); // doesn't matter because weights are zero

  // Test in-place processing
  film.Process_(input, condition, num_frames);

  // Verify that input was modified in-place
  // Expected: output = input * scale + shift (elementwise)
  const float scales[3] = {scale0, scale1, scale2};
  const float shifts[3] = {shift0, shift1, shift2};
  for (int c = 0; c < input_dim; c++)
  {
    for (int t = 0; t < num_frames; t++)
    {
      const float expected = input_original(c, t) * scales[c] + shifts[c];
      assert(std::abs(input(c, t) - expected) < 1e-6f);
    }
  }

  // Verify that Process_ produces the same result as Process
  Eigen::MatrixXf input_for_process = input_original;
  film.Process(input_for_process, condition, num_frames);
  const auto out = film.GetOutput().leftCols(num_frames);

  for (int c = 0; c < input_dim; c++)
  {
    for (int t = 0; t < num_frames; t++)
    {
      assert(std::abs(input(c, t) - out(c, t)) < 1e-6f);
    }
  }
}

void test_process_inplace_scale_only()
{
  const int condition_dim = 2;
  const int input_dim = 3;
  nam::FiLM film(condition_dim, input_dim, /*shift=*/false);

  const int maxBufferSize = 64;
  film.SetMaxBufferSize(maxBufferSize);

  // Internal Conv1x1 is (condition_dim -> input_dim) in scale-only mode.
  std::vector<float> weights;
  weights.resize(input_dim * condition_dim + input_dim, 0.0f);

  const float scale0 = 2.0f;
  const float scale1 = -1.0f;
  const float scale2 = 0.5f;

  const int bias_offset = input_dim * condition_dim;
  weights[bias_offset + 0] = scale0;
  weights[bias_offset + 1] = scale1;
  weights[bias_offset + 2] = scale2;

  auto it = weights.begin();
  film.set_weights_(it);
  assert(it == weights.end());

  const int num_frames = 4;
  Eigen::MatrixXf input(input_dim, num_frames);
  input << 1.0f, 2.0f, 3.0f, 4.0f, //
    -1.0f, -2.0f, -3.0f, -4.0f, //
    0.25f, 0.5f, 0.75f, 1.0f;

  // Keep a copy of the original input for comparison
  const Eigen::MatrixXf input_original = input;

  Eigen::MatrixXf condition(condition_dim, num_frames);
  condition.setRandom(); // doesn't matter because weights are zero

  // Test in-place processing
  film.Process_(input, condition, num_frames);

  // Verify that input was modified in-place
  const float scales[3] = {scale0, scale1, scale2};
  for (int c = 0; c < input_dim; c++)
  {
    for (int t = 0; t < num_frames; t++)
    {
      const float expected = input_original(c, t) * scales[c];
      assert(std::abs(input(c, t) - expected) < 1e-6f);
    }
  }

  // Verify that Process_ produces the same result as Process
  Eigen::MatrixXf input_for_process = input_original;
  film.Process(input_for_process, condition, num_frames);
  const auto out = film.GetOutput().leftCols(num_frames);

  for (int c = 0; c < input_dim; c++)
  {
    for (int t = 0; t < num_frames; t++)
    {
      assert(std::abs(input(c, t) - out(c, t)) < 1e-6f);
    }
  }
}

void test_process_inplace_partial_frames()
{
  // Test that only the first num_frames columns are modified when input has more columns
  const int condition_dim = 2;
  const int input_dim = 3;
  nam::FiLM film(condition_dim, input_dim, /*shift=*/true);

  const int maxBufferSize = 64;
  film.SetMaxBufferSize(maxBufferSize);

  // Configure the internal Conv1x1 with zero weights and fixed biases
  std::vector<float> weights;
  weights.resize((2 * input_dim) * condition_dim + (2 * input_dim), 0.0f);

  const float scale0 = 2.0f;
  const float scale1 = -1.0f;
  const float scale2 = 0.5f;
  const float shift0 = 10.0f;
  const float shift1 = -20.0f;
  const float shift2 = 3.0f;

  const int bias_offset = (2 * input_dim) * condition_dim;
  weights[bias_offset + 0] = scale0;
  weights[bias_offset + 1] = scale1;
  weights[bias_offset + 2] = scale2;
  weights[bias_offset + 3] = shift0;
  weights[bias_offset + 4] = shift1;
  weights[bias_offset + 5] = shift2;

  auto it = weights.begin();
  film.set_weights_(it);
  assert(it == weights.end());

  // Create input with more columns than num_frames
  const int total_cols = 8;
  const int num_frames = 4;
  Eigen::MatrixXf input(input_dim, total_cols);
  input << 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, //
    -1.0f, -2.0f, -3.0f, -4.0f, -5.0f, -6.0f, -7.0f, -8.0f, //
    0.25f, 0.5f, 0.75f, 1.0f, 1.25f, 1.5f, 1.75f, 2.0f;

  // Keep a copy of the original input
  const Eigen::MatrixXf input_original = input;

  Eigen::MatrixXf condition(condition_dim, num_frames);
  condition.setRandom();

  // Test in-place processing with only num_frames frames
  film.Process_(input, condition, num_frames);

  // Verify that only the first num_frames columns were modified
  const float scales[3] = {scale0, scale1, scale2};
  const float shifts[3] = {shift0, shift1, shift2};
  for (int c = 0; c < input_dim; c++)
  {
    // First num_frames should be modified
    for (int t = 0; t < num_frames; t++)
    {
      const float expected = input_original(c, t) * scales[c] + shifts[c];
      assert(std::abs(input(c, t) - expected) < 1e-6f);
    }
    // Remaining columns should be unchanged
    for (int t = num_frames; t < total_cols; t++)
    {
      assert(std::abs(input(c, t) - input_original(c, t)) < 1e-6f);
    }
  }
}
} // namespace test_film
