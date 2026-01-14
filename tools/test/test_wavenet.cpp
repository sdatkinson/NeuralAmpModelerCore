// Test to verify WaveNet::process is real-time safe (no allocations/frees)

#include <Eigen/Dense>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <new>
#include <vector>

#include "NAM/wavenet.h"

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

  // Reset allocation counters
  g_allocation_count = 0;
  g_deallocation_count = 0;
  g_tracking_enabled = true;

  // Matrix product with noalias() - should not allocate (all matrices pre-allocated)
  // Using noalias() is important for matrix products to avoid unnecessary temporaries
  // Note: Even without noalias(), Eigen may avoid temporaries when matrices are distinct,
  // but noalias() is best practice for real-time safety
  c.noalias() = a * b;

  // Disable tracking before any cleanup
  g_tracking_enabled = false;

  // Assert no allocations or frees occurred
  assert(g_allocation_count == 0 && "Matrix product with noalias() allocated memory (unexpected)");
  assert(g_deallocation_count == 0 && "Matrix product with noalias() freed memory (unexpected)");

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

  // Reset allocation counters
  g_allocation_count = 0;
  g_deallocation_count = 0;
  g_tracking_enabled = true;

  // This operation should allocate (resizing requires reallocation)
  a.resize(rows * 2, cols * 2);

  // Disable tracking before any cleanup
  g_tracking_enabled = false;

  // Assert that allocations occurred (this test verifies our tracking works)
  // Note: This test is meant to verify the tracking mechanism works,
  // so we expect allocations/deallocations here
  assert((g_allocation_count > 0 || g_deallocation_count > 0)
         && "Matrix resize should have caused allocations (tracking may not be working)");
}

// Test that process() method does not allocate or free memory
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

  std::vector<nam::wavenet::LayerArrayParams> layer_array_params;
  // First layer array
  layer_array_params.emplace_back(
    input_size, condition_size, head_size, channels, kernel_size, std::vector<int>{1}, activation, gated, head_bias);
  // Second layer array (head_size of first must match channels of second)
  layer_array_params.emplace_back(
    head_size, condition_size, head_size, channels, kernel_size, std::vector<int>{1}, activation, gated, head_bias);

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

    // Reset allocation counters
    g_allocation_count = 0;
    g_deallocation_count = 0;
    g_tracking_enabled = true;

    // Call process() - this should not allocate or free
    wavenet->process(input.data(), output.data(), buffer_size);

    // Disable tracking before any cleanup
    g_tracking_enabled = false;

    // Debug output
    if (g_allocation_count > 0 || g_deallocation_count > 0)
    {
      std::cerr << "Buffer size " << buffer_size << ": allocations=" << g_allocation_count
                << ", deallocations=" << g_deallocation_count << "\n";
    }

    // Assert no allocations or frees occurred
    if (g_allocation_count != 0 || g_deallocation_count != 0)
    {
      std::cerr << "ERROR: Buffer size " << buffer_size << " - process() allocated " << g_allocation_count
                << " times, freed " << g_deallocation_count << " times (not real-time safe)\n";
      std::abort();
    }

    // Verify output is valid
    for (int i = 0; i < buffer_size; i++)
    {
      assert(std::isfinite(output[i]));
    }
  }
}
} // namespace test_wavenet
