// Test to verify WaveNet::process is real-time safe (no allocations/frees)

#include <Eigen/Dense>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <functional>
#include <iostream>
#include <new>
#include <string>
#include <vector>

#include "NAM/wavenet.h"
#include "NAM/conv1d.h"

// Allocation tracking
namespace
{
volatile int g_allocation_count = 0;
volatile int g_deallocation_count = 0;
volatile bool g_tracking_enabled = false;

// Original malloc/free functions
void* (*original_malloc)(size_t) = nullptr;
void (*original_free)(void*) = nullptr;
void* (*original_realloc)(void*, size_t) = nullptr;
} // namespace

// Override malloc/free to track Eigen allocations (Eigen uses malloc directly)
extern "C" {
void* malloc(size_t size)
{
  if (!original_malloc)
    original_malloc = reinterpret_cast<void* (*)(size_t)>(dlsym(RTLD_NEXT, "malloc"));
  void* ptr = original_malloc(size);
  if (g_tracking_enabled && ptr != nullptr)
    ++g_allocation_count;
  return ptr;
}

void free(void* ptr)
{
  if (!original_free)
    original_free = reinterpret_cast<void (*)(void*)>(dlsym(RTLD_NEXT, "free"));
  if (g_tracking_enabled && ptr != nullptr)
    ++g_deallocation_count;
  original_free(ptr);
}

void* realloc(void* ptr, size_t size)
{
  if (!original_realloc)
    original_realloc = reinterpret_cast<void* (*)(void*, size_t)>(dlsym(RTLD_NEXT, "realloc"));
  void* new_ptr = original_realloc(ptr, size);
  if (g_tracking_enabled)
  {
    if (ptr != nullptr && new_ptr != ptr)
      ++g_deallocation_count; // Old pointer was freed
    if (new_ptr != nullptr && new_ptr != ptr)
      ++g_allocation_count; // New allocation
  }
  return new_ptr;
}
}

// Overload global new/delete operators to track allocations
void* operator new(std::size_t size)
{
  void* ptr = std::malloc(size);
  if (!ptr)
    throw std::bad_alloc();
  if (g_tracking_enabled)
    ++g_allocation_count;
  return ptr;
}

void* operator new[](std::size_t size)
{
  void* ptr = std::malloc(size);
  if (!ptr)
    throw std::bad_alloc();
  if (g_tracking_enabled)
    ++g_allocation_count;
  return ptr;
}

void operator delete(void* ptr) noexcept
{
  if (g_tracking_enabled && ptr != nullptr)
    ++g_deallocation_count;
  std::free(ptr);
}

void operator delete[](void* ptr) noexcept
{
  if (g_tracking_enabled && ptr != nullptr)
    ++g_deallocation_count;
  std::free(ptr);
}

namespace test_wavenet
{
// Helper function to run allocation tracking tests
// setup: Function to run before tracking starts (can be nullptr)
// test: Function to run while tracking allocations (required)
// teardown: Function to run after tracking stops (can be nullptr)
// expected_allocations: Expected number of allocations (default 0)
// expected_deallocations: Expected number of deallocations (default 0)
// test_name: Name of the test for error messages
template <typename TestFunc>
void run_allocation_test(std::function<void()> setup, TestFunc test, std::function<void()> teardown,
                         int expected_allocations, int expected_deallocations, const char* test_name)
{
  // Run setup if provided
  if (setup)
    setup();

  // Reset allocation counters and enable tracking
  g_allocation_count = 0;
  g_deallocation_count = 0;
  g_tracking_enabled = true;

  // Run the test code
  test();

  // Disable tracking before any cleanup
  g_tracking_enabled = false;

  // Run teardown if provided
  if (teardown)
    teardown();

  // Assert expected allocations/deallocations
  if (g_allocation_count != expected_allocations || g_deallocation_count != expected_deallocations)
  {
    std::cerr << "ERROR: " << test_name << " - Expected " << expected_allocations << " allocations, "
              << expected_deallocations << " deallocations. Got " << g_allocation_count << " allocations, "
              << g_deallocation_count << " deallocations.\n";
    std::abort();
  }
}

// Convenience wrapper for tests that expect zero allocations (most common case)
template <typename TestFunc>
void run_allocation_test_no_allocations(std::function<void()> setup, TestFunc test, std::function<void()> teardown,
                                        const char* test_name)
{
  run_allocation_test(setup, test, teardown, 0, 0, test_name);
}

// Convenience wrapper for tests that expect allocations (for testing the tracking mechanism)
template <typename TestFunc>
void run_allocation_test_expect_allocations(std::function<void()> setup, TestFunc test, std::function<void()> teardown,
                                            const char* test_name)
{
  // Run setup if provided
  if (setup)
    setup();

  // Reset allocation counters and enable tracking
  g_allocation_count = 0;
  g_deallocation_count = 0;
  g_tracking_enabled = true;

  // Run the test code
  test();

  // Disable tracking before any cleanup
  g_tracking_enabled = false;

  // Run teardown if provided
  if (teardown)
    teardown();

  // Assert that allocations occurred (this test verifies our tracking works)
  if (g_allocation_count == 0 && g_deallocation_count == 0)
  {
    std::cerr << "ERROR: " << test_name
              << " - Expected allocations/deallocations but none occurred (tracking may not be working)\n";
    std::abort();
  }
}

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
      // Using noalias() is important for matrix products to avoid unnecessary temporaries
      // Note: Even without noalias(), Eigen may avoid temporaries when matrices are distinct,
      // but noalias() is best practice for real-time safety
      c.noalias() = a * b;
    },
    nullptr, // No teardown needed
    "test_allocation_tracking_pass");

  // Verify result: c should be rows x rows with value 2*cols (each element is sum of cols elements of value 2)
  assert(c.rows() == rows && c.cols() == rows);
  assert(std::abs(c(0, 0) - 2.0f * cols) < 0.001f);
}

// Test that resizing a matrix causes allocations (should be caught)
void test_allocation_tracking_fail()
{
  const int rows = 10;
  const int cols = 20;

  // Pre-allocate matrix
  Eigen::MatrixXf a(rows, cols);
  a.setConstant(1.0f);

  run_allocation_test_expect_allocations(
    nullptr, // No setup needed
    [&]() {
      // This operation should allocate (resizing requires reallocation)
      a.resize(rows * 2, cols * 2);
    },
    nullptr, // No teardown needed
    "test_allocation_tracking_fail");
}

// Test that Conv1D::Process() method does not allocate or free memory
void test_conv1d_process_realtime_safe()
{
  // Setup: Create a Conv1D
  const int in_channels = 1;
  const int out_channels = 1;
  const int kernel_size = 1;
  const bool do_bias = false;
  const int dilation = 1;

  nam::Conv1D conv;
  conv.set_size_(in_channels, out_channels, kernel_size, do_bias, dilation);

  // Set weights: simple identity
  std::vector<float> weights{1.0f};
  auto it = weights.begin();
  conv.set_weights_(it);

  const int maxBufferSize = 256;
  conv.SetMaxBufferSize(maxBufferSize);

  // Test with several different buffer sizes
  std::vector<int> buffer_sizes{1, 8, 16, 32, 64, 128, 256};

  for (int buffer_size : buffer_sizes)
  {
    // Prepare input matrix (allocate before tracking)
    Eigen::MatrixXf input(in_channels, buffer_size);
    input.setConstant(0.5f);

    std::string test_name = "Conv1D Process - Buffer size " + std::to_string(buffer_size);
    run_allocation_test_no_allocations(
      nullptr, // No setup needed
      [&]() {
        // Call Process() - this should not allocate or free
        conv.Process(input, buffer_size);
      },
      nullptr, // No teardown needed
      test_name.c_str());

    // Verify output is valid
    auto output = conv.GetOutput().leftCols(buffer_size);
    assert(output.rows() == out_channels && output.cols() == buffer_size);
    assert(std::isfinite(output(0, 0)));
  }
}

// Test that Conv1D::Process() with grouped convolution (groups > 1) does not allocate or free memory
void test_conv1d_grouped_process_realtime_safe()
{
  // Setup: Create a Conv1D with grouped convolution
  const int in_channels = 4;
  const int out_channels = 4;
  const int kernel_size = 2;
  const bool do_bias = true;
  const int dilation = 1;
  const int groups = 2;

  nam::Conv1D conv;
  conv.set_size_(in_channels, out_channels, kernel_size, do_bias, dilation, groups);

  // Set weights for grouped convolution
  // Each group: 2 in_channels, 2 out_channels, kernel_size=2
  // Weight layout: for each group g, for each (i,j), for each k
  std::vector<float> weights;
  // Group 0, kernel[0]: identity
  weights.push_back(1.0f);
  weights.push_back(0.0f);
  weights.push_back(0.0f);
  weights.push_back(1.0f);
  // Group 0, kernel[1]: identity
  weights.push_back(1.0f);
  weights.push_back(0.0f);
  weights.push_back(0.0f);
  weights.push_back(1.0f);
  // Group 1, kernel[0]: identity
  weights.push_back(1.0f);
  weights.push_back(0.0f);
  weights.push_back(0.0f);
  weights.push_back(1.0f);
  // Group 1, kernel[1]: identity
  weights.push_back(1.0f);
  weights.push_back(0.0f);
  weights.push_back(0.0f);
  weights.push_back(1.0f);
  // Bias: [0.1, 0.2, 0.3, 0.4]
  weights.push_back(0.1f);
  weights.push_back(0.2f);
  weights.push_back(0.3f);
  weights.push_back(0.4f);

  auto it = weights.begin();
  conv.set_weights_(it);

  const int maxBufferSize = 256;
  conv.SetMaxBufferSize(maxBufferSize);

  // Test with several different buffer sizes
  std::vector<int> buffer_sizes{1, 8, 16, 32, 64, 128, 256};

  for (int buffer_size : buffer_sizes)
  {
    // Prepare input matrix (allocate before tracking)
    Eigen::MatrixXf input(in_channels, buffer_size);
    input.setConstant(0.5f);

    std::string test_name = "Conv1D Grouped Process - Buffer size " + std::to_string(buffer_size);
    run_allocation_test_no_allocations(
      nullptr, // No setup needed
      [&]() {
        // Call Process() - this should not allocate or free
        conv.Process(input, buffer_size);
      },
      nullptr, // No teardown needed
      test_name.c_str());

    // Verify output is valid
    auto output = conv.GetOutput().leftCols(buffer_size);
    assert(output.rows() == out_channels && output.cols() == buffer_size);
    assert(std::isfinite(output(0, 0)));
    assert(std::isfinite(output(out_channels - 1, buffer_size - 1)));
  }
}

// Test that Conv1D::Process() with grouped convolution and dilation does not allocate or free memory
void test_conv1d_grouped_dilated_process_realtime_safe()
{
  // Setup: Create a Conv1D with grouped convolution and dilation
  const int in_channels = 6;
  const int out_channels = 6;
  const int kernel_size = 2;
  const bool do_bias = false;
  const int dilation = 2;
  const int groups = 3;

  nam::Conv1D conv;
  conv.set_size_(in_channels, out_channels, kernel_size, do_bias, dilation, groups);

  // Set weights for grouped convolution with 3 groups
  // Each group: 2 in_channels, 2 out_channels, kernel_size=2
  std::vector<float> weights;
  for (int g = 0; g < groups; g++)
  {
    // Group g, kernel[0]: identity
    weights.push_back(1.0f);
    weights.push_back(0.0f);
    weights.push_back(0.0f);
    weights.push_back(1.0f);
    // Group g, kernel[1]: identity
    weights.push_back(1.0f);
    weights.push_back(0.0f);
    weights.push_back(0.0f);
    weights.push_back(1.0f);
  }

  auto it = weights.begin();
  conv.set_weights_(it);

  const int maxBufferSize = 256;
  conv.SetMaxBufferSize(maxBufferSize);

  // Test with several different buffer sizes
  std::vector<int> buffer_sizes{1, 8, 16, 32, 64, 128, 256};

  for (int buffer_size : buffer_sizes)
  {
    // Prepare input matrix (allocate before tracking)
    Eigen::MatrixXf input(in_channels, buffer_size);
    input.setConstant(0.5f);

    std::string test_name = "Conv1D Grouped Dilated Process - Buffer size " + std::to_string(buffer_size);
    run_allocation_test_no_allocations(
      nullptr, // No setup needed
      [&]() {
        // Call Process() - this should not allocate or free
        conv.Process(input, buffer_size);
      },
      nullptr, // No teardown needed
      test_name.c_str());

    // Verify output is valid
    auto output = conv.GetOutput().leftCols(buffer_size);
    assert(output.rows() == out_channels && output.cols() == buffer_size);
    assert(std::isfinite(output(0, 0)));
    assert(std::isfinite(output(out_channels - 1, buffer_size - 1)));
  }
}

// Test that Layer::Process() method does not allocate or free memory
void test_layer_process_realtime_safe()
{
  // Setup: Create a Layer
  const int condition_size = 1;
  const int channels = 1;
  const int kernel_size = 1;
  const int dilation = 1;
  const std::string activation = "ReLU";
  const bool gated = false;

  auto layer = nam::wavenet::_Layer(condition_size, channels, kernel_size, dilation, activation, gated);

  // Set weights
  std::vector<float> weights{1.0f, 0.0f, // Conv (weight, bias)
                             1.0f, // Input mixin
                             1.0f, 0.0f}; // 1x1 (weight, bias)
  auto it = weights.begin();
  layer.set_weights_(it);

  const int maxBufferSize = 256;
  layer.SetMaxBufferSize(maxBufferSize);

  // Test with several different buffer sizes
  std::vector<int> buffer_sizes{1, 8, 16, 32, 64, 128, 256};

  for (int buffer_size : buffer_sizes)
  {
    // Prepare input/condition matrices (allocate before tracking)
    Eigen::MatrixXf input(channels, buffer_size);
    Eigen::MatrixXf condition(condition_size, buffer_size);
    input.setConstant(0.5f);
    condition.setConstant(0.5f);

    std::string test_name = "Layer Process - Buffer size " + std::to_string(buffer_size);
    run_allocation_test_no_allocations(
      nullptr, // No setup needed
      [&]() {
        // Call Process() - this should not allocate or free
        layer.Process(input, condition, buffer_size);
      },
      nullptr, // No teardown needed
      test_name.c_str());

    // Verify output is valid
    auto output = layer.GetOutputNextLayer().leftCols(buffer_size);
    assert(output.rows() == channels && output.cols() == buffer_size);
    assert(std::isfinite(output(0, 0)));
  }
}

// Test that LayerArray::Process() method does not allocate or free memory
void test_layer_array_process_realtime_safe()
{
  // Setup: Create LayerArray
  const int input_size = 1;
  const int condition_size = 1;
  const int head_size = 1;
  const int channels = 1;
  const int kernel_size = 1;
  std::vector<int> dilations{1};
  const std::string activation = "ReLU";
  const bool gated = false;
  const bool head_bias = false;
  const int groups = 1;

  auto layer_array = nam::wavenet::_LayerArray(
    input_size, condition_size, head_size, channels, kernel_size, dilations, activation, gated, head_bias, groups);

  // Set weights: rechannel(1), layer(conv:1+1, input_mixin:1, 1x1:1+1), head_rechannel(1)
  std::vector<float> weights{1.0f, // Rechannel
                             1.0f, 0.0f, // Layer: conv
                             1.0f, // Layer: input_mixin
                             1.0f, 0.0f, // Layer: 1x1
                             1.0f}; // Head rechannel
  auto it = weights.begin();
  layer_array.set_weights_(it);

  const int maxBufferSize = 256;
  layer_array.SetMaxBufferSize(maxBufferSize);

  // Test with several different buffer sizes
  std::vector<int> buffer_sizes{1, 8, 16, 32, 64, 128, 256};

  for (int buffer_size : buffer_sizes)
  {
    // Prepare input/condition matrices (allocate before tracking)
    Eigen::MatrixXf layer_inputs(input_size, buffer_size);
    Eigen::MatrixXf condition(condition_size, buffer_size);
    layer_inputs.setConstant(0.5f);
    condition.setConstant(0.5f);

    std::string test_name = "LayerArray Process - Buffer size " + std::to_string(buffer_size);
    run_allocation_test_no_allocations(
      nullptr, // No setup needed
      [&]() {
        // Call Process() - this should not allocate or free
        layer_array.Process(layer_inputs, condition, buffer_size);
      },
      nullptr, // No teardown needed
      test_name.c_str());

    // Verify output is valid
    auto layer_outputs = layer_array.GetLayerOutputs().leftCols(buffer_size);
    auto head_outputs = layer_array.GetHeadOutputs().leftCols(buffer_size);
    assert(layer_outputs.rows() == channels && layer_outputs.cols() == buffer_size);
    assert(head_outputs.rows() == head_size && head_outputs.cols() == buffer_size);
    assert(std::isfinite(layer_outputs(0, 0)));
    assert(std::isfinite(head_outputs(0, 0)));
  }
}

// Test that WaveNet::process() method does not allocate or free memory
void test_process_realtime_safe()
{
  // Setup: Create WaveNet with two layer arrays (simplified configuration)
  const int input_size = 1;
  const int condition_size = 1;
  const int head_size = 1;
  const int channels = 1;
  const int kernel_size = 1;
  std::vector<int> dilations{1};
  const std::string activation = "ReLU";
  const bool gated = false;
  const bool head_bias = false;
  const float head_scale = 1.0f;
  const bool with_head = false;
  const int groups = 1;

  std::vector<nam::wavenet::LayerArrayParams> layer_array_params;
  // First layer array
  std::vector<int> dilations1{1};
  layer_array_params.push_back(nam::wavenet::LayerArrayParams(input_size, condition_size, head_size, channels,
                                                              kernel_size, std::move(dilations1), activation, gated,
                                                              head_bias, groups));
  // Second layer array (head_size of first must match channels of second)
  std::vector<int> dilations2{1};
  layer_array_params.push_back(nam::wavenet::LayerArrayParams(head_size, condition_size, head_size, channels,
                                                              kernel_size, std::move(dilations2), activation, gated,
                                                              head_bias, groups));

  // Weights: Array 0: rechannel(1), layer(conv:1+1, input_mixin:1, 1x1:1+1), head_rechannel(1)
  //          Array 1: same structure
  //          Head scale: 1
  std::vector<float> weights;
  // Array 0: rechannel, layer, head_rechannel
  weights.insert(weights.end(), {1.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f});
  // Array 1: rechannel, layer, head_rechannel
  weights.insert(weights.end(), {1.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f});
  weights.push_back(head_scale);

  auto wavenet = std::make_unique<nam::wavenet::WaveNet>(layer_array_params, head_scale, with_head, weights, 48000.0);

  const int maxBufferSize = 256;
  wavenet->Reset(48000.0, maxBufferSize);

  // Test with several different buffer sizes
  std::vector<int> buffer_sizes{1, 8, 16, 32, 64, 128, 256};

  for (int buffer_size : buffer_sizes)
  {
    // Prepare input/output buffers (allocate before tracking)
    std::vector<NAM_SAMPLE> input(buffer_size, 0.5f);
    std::vector<NAM_SAMPLE> output(buffer_size, 0.0f);

    std::string test_name = "WaveNet process - Buffer size " + std::to_string(buffer_size);
    run_allocation_test_no_allocations(
      nullptr, // No setup needed
      [&]() {
        // Call process() - this should not allocate or free
        wavenet->process(input.data(), output.data(), buffer_size);
      },
      nullptr, // No teardown needed
      test_name.c_str());

    // Verify output is valid
    for (int i = 0; i < buffer_size; i++)
    {
      assert(std::isfinite(output[i]));
    }
  }
}
} // namespace test_wavenet
