// Allocation tracking infrastructure for real-time safety tests
// This header provides tools to detect memory allocations/deallocations
// during real-time critical code paths.

#pragma once

#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <functional>
#include <iostream>
#include <new>

// Allocation tracking globals
namespace allocation_tracking
{
extern volatile int g_allocation_count;
extern volatile int g_deallocation_count;
extern volatile bool g_tracking_enabled;

// Original malloc/free functions
extern void* (*original_malloc)(size_t);
extern void (*original_free)(void*);
extern void* (*original_realloc)(void*, size_t);

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
} // namespace allocation_tracking
