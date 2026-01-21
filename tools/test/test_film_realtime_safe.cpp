// Test to verify FiLM::Process and FiLM::Process_ are real-time safe (no allocations/frees)

#include <Eigen/Dense>
#include <cassert>
#include <cmath>
#include <string>
#include <vector>

#include "NAM/film.h"
#include "allocation_tracking.h"

namespace test_film_realtime_safe
{
using namespace allocation_tracking;

// Test that pre-allocated Eigen operations with noalias() don't allocate
void test_allocation_tracking_pass()
{
  const int rows = 10;
  const int cols = 20;

  // Pre-allocate matrices for matrix product: c = a * b
  // a is rows x cols, b is cols x rows, so c is rows x rows
  Eigen::MatrixXf a(rows, cols);
  Eigen::MatrixXf b(cols, rows);
  Eigen::MatrixXf c(rows, rows);

  a.setConstant(1.0f);
  b.setConstant(2.0f);

  run_allocation_test_no_allocations(
    nullptr, // No setup needed
    [&]() {
      // Matrix product with noalias() - should not allocate (all matrices pre-allocated)
      c.noalias() = a * b;
    },
    nullptr, // No teardown needed
    "test_allocation_tracking_pass");

  // Verify result: c should be rows x rows with value 2*cols (each element is sum of cols elements of value 2)
  assert(c.rows() == rows && c.cols() == rows);
  assert(std::abs(c(0, 0) - 2.0f * cols) < 0.001f);
}

// Test that creating a new matrix causes allocations (should be caught)
void test_allocation_tracking_fail()
{
  run_allocation_test_expect_allocations(
    nullptr, // No setup needed
    [&]() {
      // This operation should allocate (creating new matrix)
      Eigen::MatrixXf a(10, 20);
      a.setConstant(1.0f);
    },
    nullptr, // No teardown needed
    "test_allocation_tracking_fail");
}

// Test that FiLM::Process() method with shift does not allocate or free memory
void test_film_process_with_shift_realtime_safe()
{
  // Setup: Create a FiLM with shift enabled
  const int condition_dim = 2;
  const int input_dim = 3;
  nam::FiLM film(condition_dim, input_dim, /*shift=*/true);

  const int maxBufferSize = 256;
  film.SetMaxBufferSize(maxBufferSize);

  // Set weights: all-zero weights with fixed biases
  std::vector<float> weights;
  weights.resize((2 * input_dim) * condition_dim + (2 * input_dim), 0.0f);

  // Set biases for scale and shift
  const int bias_offset = (2 * input_dim) * condition_dim;
  weights[bias_offset + 0] = 2.0f; // scale0
  weights[bias_offset + 1] = -1.0f; // scale1
  weights[bias_offset + 2] = 0.5f; // scale2
  weights[bias_offset + 3] = 10.0f; // shift0
  weights[bias_offset + 4] = -20.0f; // shift1
  weights[bias_offset + 5] = 3.0f; // shift2

  auto it = weights.begin();
  film.set_weights_(it);

  // Test with several different buffer sizes
  std::vector<int> buffer_sizes{1, 8, 16, 32, 64, 128, 256};

  for (int buffer_size : buffer_sizes)
  {
    // Prepare input/condition matrices (allocate before tracking)
    Eigen::MatrixXf input(input_dim, buffer_size);
    Eigen::MatrixXf condition(condition_dim, buffer_size);
    input.setConstant(0.5f);
    condition.setConstant(0.5f);

    std::string test_name = "FiLM Process (with shift) - Buffer size " + std::to_string(buffer_size);
    run_allocation_test_no_allocations(
      nullptr, // No setup needed
      [&]() {
        // Call Process() - this should not allocate or free
        film.Process(input, condition, buffer_size);
      },
      nullptr, // No teardown needed
      test_name.c_str());

    // Verify output is valid
    auto output = film.GetOutput().leftCols(buffer_size);
    assert(output.rows() == input_dim && output.cols() == buffer_size);
    assert(std::isfinite(output(0, 0)));
    assert(std::isfinite(output(input_dim - 1, buffer_size - 1)));
  }
}

// Test that FiLM::Process() method without shift does not allocate or free memory
void test_film_process_without_shift_realtime_safe()
{
  // Setup: Create a FiLM with shift disabled (scale-only mode)
  const int condition_dim = 2;
  const int input_dim = 3;
  nam::FiLM film(condition_dim, input_dim, /*shift=*/false);

  const int maxBufferSize = 256;
  film.SetMaxBufferSize(maxBufferSize);

  // Set weights: all-zero weights with fixed biases for scale
  std::vector<float> weights;
  weights.resize(input_dim * condition_dim + input_dim, 0.0f);

  // Set biases for scale
  const int bias_offset = input_dim * condition_dim;
  weights[bias_offset + 0] = 2.0f; // scale0
  weights[bias_offset + 1] = -1.0f; // scale1
  weights[bias_offset + 2] = 0.5f; // scale2

  auto it = weights.begin();
  film.set_weights_(it);

  // Test with several different buffer sizes
  std::vector<int> buffer_sizes{1, 8, 16, 32, 64, 128, 256};

  for (int buffer_size : buffer_sizes)
  {
    // Prepare input/condition matrices (allocate before tracking)
    Eigen::MatrixXf input(input_dim, buffer_size);
    Eigen::MatrixXf condition(condition_dim, buffer_size);
    input.setConstant(0.5f);
    condition.setConstant(0.5f);

    std::string test_name = "FiLM Process (without shift) - Buffer size " + std::to_string(buffer_size);
    run_allocation_test_no_allocations(
      nullptr, // No setup needed
      [&]() {
        // Call Process() - this should not allocate or free
        film.Process(input, condition, buffer_size);
      },
      nullptr, // No teardown needed
      test_name.c_str());

    // Verify output is valid
    auto output = film.GetOutput().leftCols(buffer_size);
    assert(output.rows() == input_dim && output.cols() == buffer_size);
    assert(std::isfinite(output(0, 0)));
    assert(std::isfinite(output(input_dim - 1, buffer_size - 1)));
  }
}

// Test that FiLM::Process_() in-place method with shift does not allocate or free memory
void test_film_process_inplace_with_shift_realtime_safe()
{
  // Setup: Create a FiLM with shift enabled
  const int condition_dim = 2;
  const int input_dim = 3;
  nam::FiLM film(condition_dim, input_dim, /*shift=*/true);

  const int maxBufferSize = 256;
  film.SetMaxBufferSize(maxBufferSize);

  // Set weights: all-zero weights with fixed biases
  std::vector<float> weights;
  weights.resize((2 * input_dim) * condition_dim + (2 * input_dim), 0.0f);

  // Set biases for scale and shift
  const int bias_offset = (2 * input_dim) * condition_dim;
  weights[bias_offset + 0] = 2.0f; // scale0
  weights[bias_offset + 1] = -1.0f; // scale1
  weights[bias_offset + 2] = 0.5f; // scale2
  weights[bias_offset + 3] = 10.0f; // shift0
  weights[bias_offset + 4] = -20.0f; // shift1
  weights[bias_offset + 5] = 3.0f; // shift2

  auto it = weights.begin();
  film.set_weights_(it);

  // Test with several different buffer sizes
  std::vector<int> buffer_sizes{1, 8, 16, 32, 64, 128, 256};

  for (int buffer_size : buffer_sizes)
  {
    // Prepare input/condition matrices (allocate before tracking)
    Eigen::MatrixXf input(input_dim, buffer_size);
    Eigen::MatrixXf condition(condition_dim, buffer_size);
    input.setConstant(0.5f);
    condition.setConstant(0.5f);

    std::string test_name = "FiLM Process_ (in-place with shift) - Buffer size " + std::to_string(buffer_size);
    run_allocation_test_no_allocations(
      nullptr, // No setup needed
      [&]() {
        // Call Process_() - this should not allocate or free
        film.Process_(input, condition, buffer_size);
      },
      nullptr, // No teardown needed
      test_name.c_str());

    // Verify output is valid (input should be modified in-place)
    assert(input.rows() == input_dim && input.cols() >= buffer_size);
    assert(std::isfinite(input(0, 0)));
    assert(std::isfinite(input(input_dim - 1, buffer_size - 1)));
  }
}

// Test that FiLM::Process_() in-place method without shift does not allocate or free memory
void test_film_process_inplace_without_shift_realtime_safe()
{
  // Setup: Create a FiLM with shift disabled (scale-only mode)
  const int condition_dim = 2;
  const int input_dim = 3;
  nam::FiLM film(condition_dim, input_dim, /*shift=*/false);

  const int maxBufferSize = 256;
  film.SetMaxBufferSize(maxBufferSize);

  // Set weights: all-zero weights with fixed biases for scale
  std::vector<float> weights;
  weights.resize(input_dim * condition_dim + input_dim, 0.0f);

  // Set biases for scale
  const int bias_offset = input_dim * condition_dim;
  weights[bias_offset + 0] = 2.0f; // scale0
  weights[bias_offset + 1] = -1.0f; // scale1
  weights[bias_offset + 2] = 0.5f; // scale2

  auto it = weights.begin();
  film.set_weights_(it);

  // Test with several different buffer sizes
  std::vector<int> buffer_sizes{1, 8, 16, 32, 64, 128, 256};

  for (int buffer_size : buffer_sizes)
  {
    // Prepare input/condition matrices (allocate before tracking)
    Eigen::MatrixXf input(input_dim, buffer_size);
    Eigen::MatrixXf condition(condition_dim, buffer_size);
    input.setConstant(0.5f);
    condition.setConstant(0.5f);

    std::string test_name = "FiLM Process_ (in-place without shift) - Buffer size " + std::to_string(buffer_size);
    run_allocation_test_no_allocations(
      nullptr, // No setup needed
      [&]() {
        // Call Process_() - this should not allocate or free
        film.Process_(input, condition, buffer_size);
      },
      nullptr, // No teardown needed
      test_name.c_str());

    // Verify output is valid (input should be modified in-place)
    assert(input.rows() == input_dim && input.cols() >= buffer_size);
    assert(std::isfinite(input(0, 0)));
    assert(std::isfinite(input(input_dim - 1, buffer_size - 1)));
  }
}

// Test that FiLM::Process() with larger dimensions does not allocate or free memory
void test_film_process_large_dimensions_realtime_safe()
{
  // Setup: Create a FiLM with larger dimensions
  const int condition_dim = 8;
  const int input_dim = 16;
  nam::FiLM film(condition_dim, input_dim, /*shift=*/true);

  const int maxBufferSize = 256;
  film.SetMaxBufferSize(maxBufferSize);

  // Set weights: all-zero weights with fixed biases
  std::vector<float> weights;
  weights.resize((2 * input_dim) * condition_dim + (2 * input_dim), 0.0f);

  // Set biases for scale and shift (use simple pattern)
  const int bias_offset = (2 * input_dim) * condition_dim;
  for (int i = 0; i < input_dim; i++)
  {
    weights[bias_offset + i] = 1.0f + 0.1f * i; // scale
    weights[bias_offset + input_dim + i] = 0.5f * i; // shift
  }

  auto it = weights.begin();
  film.set_weights_(it);

  // Test with several different buffer sizes
  std::vector<int> buffer_sizes{1, 8, 16, 32, 64, 128, 256};

  for (int buffer_size : buffer_sizes)
  {
    // Prepare input/condition matrices (allocate before tracking)
    Eigen::MatrixXf input(input_dim, buffer_size);
    Eigen::MatrixXf condition(condition_dim, buffer_size);
    input.setConstant(0.5f);
    condition.setConstant(0.5f);

    std::string test_name = "FiLM Process (large dimensions: condition_dim=" + std::to_string(condition_dim)
                            + ", input_dim=" + std::to_string(input_dim) + ") - Buffer size "
                            + std::to_string(buffer_size);
    run_allocation_test_no_allocations(
      nullptr, // No setup needed
      [&]() {
        // Call Process() - this should not allocate or free
        film.Process(input, condition, buffer_size);
      },
      nullptr, // No teardown needed
      test_name.c_str());

    // Verify output is valid
    auto output = film.GetOutput().leftCols(buffer_size);
    assert(output.rows() == input_dim && output.cols() == buffer_size);
    assert(std::isfinite(output(0, 0)));
    assert(std::isfinite(output(input_dim - 1, buffer_size - 1)));
  }
}

// Test that FiLM::Process() with partial frame processing does not allocate or free memory
void test_film_process_partial_frames_realtime_safe()
{
  // Setup: Create a FiLM
  const int condition_dim = 2;
  const int input_dim = 3;
  nam::FiLM film(condition_dim, input_dim, /*shift=*/true);

  const int maxBufferSize = 256;
  film.SetMaxBufferSize(maxBufferSize);

  // Set weights: all-zero weights with fixed biases
  std::vector<float> weights;
  weights.resize((2 * input_dim) * condition_dim + (2 * input_dim), 0.0f);

  // Set biases for scale and shift
  const int bias_offset = (2 * input_dim) * condition_dim;
  weights[bias_offset + 0] = 2.0f; // scale0
  weights[bias_offset + 1] = -1.0f; // scale1
  weights[bias_offset + 2] = 0.5f; // scale2
  weights[bias_offset + 3] = 10.0f; // shift0
  weights[bias_offset + 4] = -20.0f; // shift1
  weights[bias_offset + 5] = 3.0f; // shift2

  auto it = weights.begin();
  film.set_weights_(it);

  // Test with buffer smaller than maxBufferSize to verify partial frame processing
  const int full_buffer_size = 64;
  std::vector<int> partial_buffer_sizes{1, 8, 16, 32};

  for (int buffer_size : partial_buffer_sizes)
  {
    // Prepare input/condition matrices with full size (allocate before tracking)
    Eigen::MatrixXf input(input_dim, full_buffer_size);
    Eigen::MatrixXf condition(condition_dim, full_buffer_size);
    input.setConstant(0.5f);
    condition.setConstant(0.5f);

    std::string test_name = "FiLM Process (partial frames: " + std::to_string(buffer_size) + " of "
                            + std::to_string(full_buffer_size) + " frames)";
    run_allocation_test_no_allocations(
      nullptr, // No setup needed
      [&]() {
        // Call Process() with partial buffer_size - this should not allocate or free
        film.Process(input, condition, buffer_size);
      },
      nullptr, // No teardown needed
      test_name.c_str());

    // Verify output is valid
    auto output = film.GetOutput().leftCols(buffer_size);
    assert(output.rows() == input_dim && output.cols() >= buffer_size);
    assert(std::isfinite(output(0, 0)));
    assert(std::isfinite(output(input_dim - 1, buffer_size - 1)));
  }
}

// Test that FiLM::Process() with varying condition and input dimensions does not allocate or free memory
void test_film_process_varying_dimensions_realtime_safe()
{
  // Test various combinations of condition_dim and input_dim
  struct DimConfig
  {
    int condition_dim;
    int input_dim;
    bool shift;
  };

  std::vector<DimConfig> configs{
    {1, 1, true}, // Minimal dimensions with shift
    {1, 1, false}, // Minimal dimensions without shift
    {1, 4, true}, // Small condition, larger input
    {4, 1, false}, // Larger condition, small input
    {4, 4, true}, // Equal dimensions
    {3, 5, false}, // Non-power-of-2 dimensions
    {7, 11, true}, // Prime dimensions
  };

  const int maxBufferSize = 128;
  const int buffer_size = 64;

  for (const auto& config : configs)
  {
    // Setup: Create a FiLM with specific dimensions
    nam::FiLM film(config.condition_dim, config.input_dim, config.shift);
    film.SetMaxBufferSize(maxBufferSize);

    // Set weights: all-zero weights with fixed biases
    const int output_channels = config.shift ? (2 * config.input_dim) : config.input_dim;
    std::vector<float> weights;
    weights.resize(output_channels * config.condition_dim + output_channels, 0.0f);

    // Set biases
    const int bias_offset = output_channels * config.condition_dim;
    for (int i = 0; i < output_channels; i++)
    {
      weights[bias_offset + i] = 1.0f + 0.1f * i;
    }

    auto it = weights.begin();
    film.set_weights_(it);

    // Prepare input/condition matrices (allocate before tracking)
    Eigen::MatrixXf input(config.input_dim, buffer_size);
    Eigen::MatrixXf condition(config.condition_dim, buffer_size);
    input.setConstant(0.5f);
    condition.setConstant(0.5f);

    std::string shift_str = config.shift ? "true" : "false";
    std::string test_name = "FiLM Process (condition_dim=" + std::to_string(config.condition_dim)
                            + ", input_dim=" + std::to_string(config.input_dim) + ", shift=" + shift_str + ")";
    run_allocation_test_no_allocations(
      nullptr, // No setup needed
      [&]() {
        // Call Process() - this should not allocate or free
        film.Process(input, condition, buffer_size);
      },
      nullptr, // No teardown needed
      test_name.c_str());

    // Verify output is valid
    auto output = film.GetOutput().leftCols(buffer_size);
    assert(output.rows() == config.input_dim && output.cols() >= buffer_size);
    assert(std::isfinite(output(0, 0)));
    assert(std::isfinite(output(config.input_dim - 1, buffer_size - 1)));
  }
}

// Test that multiple consecutive calls to FiLM::Process() do not allocate or free memory
void test_film_process_consecutive_calls_realtime_safe()
{
  // Setup: Create a FiLM
  const int condition_dim = 2;
  const int input_dim = 3;
  nam::FiLM film(condition_dim, input_dim, /*shift=*/true);

  const int maxBufferSize = 256;
  film.SetMaxBufferSize(maxBufferSize);

  // Set weights
  std::vector<float> weights;
  weights.resize((2 * input_dim) * condition_dim + (2 * input_dim), 0.0f);

  const int bias_offset = (2 * input_dim) * condition_dim;
  weights[bias_offset + 0] = 2.0f;
  weights[bias_offset + 1] = -1.0f;
  weights[bias_offset + 2] = 0.5f;
  weights[bias_offset + 3] = 10.0f;
  weights[bias_offset + 4] = -20.0f;
  weights[bias_offset + 5] = 3.0f;

  auto it = weights.begin();
  film.set_weights_(it);

  const int buffer_size = 64;
  const int num_consecutive_calls = 10;

  // Prepare input/condition matrices (allocate before tracking)
  Eigen::MatrixXf input(input_dim, buffer_size);
  Eigen::MatrixXf condition(condition_dim, buffer_size);
  input.setConstant(0.5f);
  condition.setConstant(0.5f);

  std::string test_name = "FiLM Process (consecutive calls: " + std::to_string(num_consecutive_calls) + " calls)";
  run_allocation_test_no_allocations(
    nullptr, // No setup needed
    [&]() {
      // Call Process() multiple times consecutively - none should allocate or free
      for (int i = 0; i < num_consecutive_calls; i++)
      {
        film.Process(input, condition, buffer_size);
      }
    },
    nullptr, // No teardown needed
    test_name.c_str());

  // Verify output is valid after all calls
  auto output = film.GetOutput().leftCols(buffer_size);
  assert(output.rows() == input_dim && output.cols() >= buffer_size);
  assert(std::isfinite(output(0, 0)));
  assert(std::isfinite(output(input_dim - 1, buffer_size - 1)));
}
} // namespace test_film_realtime_safe
