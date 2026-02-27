// Allocation tracking implementation
// This file contains the actual definitions of the global tracking variables
// and the overridden malloc/free/new/delete operators.

#include "allocation_tracking.h"

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
