// Allocation tracking implementation
// This file contains the actual definitions of the global tracking variables
// and the overridden malloc/free/new/delete operators.

#include "allocation_tracking.h"

// Under a sanitizer (ASan, TSan, LSan) the runtime owns malloc, and replacing it here through dlsym(RTLD_NEXT) kills
// run_tests before main. There we count through the sanitizer's own allocation hooks instead, so the allocation tests
// still mean the same thing (only "zero" and "more than zero" are ever checked).
#if defined(__SANITIZE_ADDRESS__) || defined(__SANITIZE_THREAD__)
  #define NAM_TEST_UNDER_SANITIZER 1
#elif defined(__has_feature)
  #if __has_feature(address_sanitizer) || __has_feature(thread_sanitizer) || __has_feature(leak_sanitizer)             \
    || __has_feature(memory_sanitizer)
    #define NAM_TEST_UNDER_SANITIZER 1
  #endif
#endif

// Allocation tracking globals - definitions
namespace allocation_tracking
{
volatile int g_allocation_count = 0;
volatile int g_deallocation_count = 0;
volatile bool g_tracking_enabled = false;

// Original malloc/free functions
void* (*original_malloc)(size_t) = nullptr;
void (*original_free)(void*) = nullptr;
void* (*original_realloc)(void*, size_t) = nullptr;
} // namespace allocation_tracking

#ifdef NAM_TEST_UNDER_SANITIZER
  #include <sanitizer/allocator_interface.h>

namespace
{
void on_malloc(const volatile void* ptr, size_t)
{
  if (allocation_tracking::g_tracking_enabled && ptr != nullptr)
    ++allocation_tracking::g_allocation_count;
}

void on_free(const volatile void* ptr)
{
  if (allocation_tracking::g_tracking_enabled && ptr != nullptr)
    ++allocation_tracking::g_deallocation_count;
}

const int hooks_installed = __sanitizer_install_malloc_and_free_hooks(on_malloc, on_free);
} // namespace

#else
// Override malloc/free to track Eigen allocations (Eigen uses malloc directly)
extern "C" {
void* malloc(size_t size)
{
  if (!allocation_tracking::original_malloc)
    allocation_tracking::original_malloc = reinterpret_cast<void* (*)(size_t)>(dlsym(RTLD_NEXT, "malloc"));
  void* ptr = allocation_tracking::original_malloc(size);
  if (allocation_tracking::g_tracking_enabled && ptr != nullptr)
    ++allocation_tracking::g_allocation_count;
  return ptr;
}

void free(void* ptr)
{
  if (!allocation_tracking::original_free)
    allocation_tracking::original_free = reinterpret_cast<void (*)(void*)>(dlsym(RTLD_NEXT, "free"));
  if (allocation_tracking::g_tracking_enabled && ptr != nullptr)
    ++allocation_tracking::g_deallocation_count;
  allocation_tracking::original_free(ptr);
}

void* realloc(void* ptr, size_t size)
{
  if (!allocation_tracking::original_realloc)
    allocation_tracking::original_realloc = reinterpret_cast<void* (*)(void*, size_t)>(dlsym(RTLD_NEXT, "realloc"));
  void* new_ptr = allocation_tracking::original_realloc(ptr, size);
  if (allocation_tracking::g_tracking_enabled)
  {
    if (ptr != nullptr && new_ptr != ptr)
      ++allocation_tracking::g_deallocation_count; // Old pointer was freed
    if (new_ptr != nullptr && new_ptr != ptr)
      ++allocation_tracking::g_allocation_count; // New allocation
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
  if (allocation_tracking::g_tracking_enabled)
    ++allocation_tracking::g_allocation_count;
  return ptr;
}

void* operator new[](std::size_t size)
{
  void* ptr = std::malloc(size);
  if (!ptr)
    throw std::bad_alloc();
  if (allocation_tracking::g_tracking_enabled)
    ++allocation_tracking::g_allocation_count;
  return ptr;
}

void operator delete(void* ptr) noexcept
{
  if (allocation_tracking::g_tracking_enabled && ptr != nullptr)
    ++allocation_tracking::g_deallocation_count;
  std::free(ptr);
}

void operator delete[](void* ptr) noexcept
{
  if (allocation_tracking::g_tracking_enabled && ptr != nullptr)
    ++allocation_tracking::g_deallocation_count;
  std::free(ptr);
}

#endif // NAM_TEST_UNDER_SANITIZER
