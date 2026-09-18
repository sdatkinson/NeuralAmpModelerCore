#pragma once
// Threading shim.
//
// NAM_SINGLE_THREADED: for bare-metal targets whose C library has no thread
// support (newlib without gthreads has no std::mutex). All engine locking
// collapses to no-ops; the embedder guarantees single-threaded use (or its
// own external synchronization) around model loading and slim switching.

// thread_local needs a TLS runtime (__aeabi_read_tp on ARM) that bare-metal
// newlib does not provide; single-threaded builds use plain statics.
#ifdef NAM_SINGLE_THREADED
  #define NAM_THREAD_LOCAL
#else
  #define NAM_THREAD_LOCAL thread_local
#endif

#ifdef NAM_SINGLE_THREADED

namespace nam
{
struct Mutex
{
};

template <typename T>
struct LockGuard
{
  explicit LockGuard(T&) {}
};
} // namespace nam

#else

  #include <mutex>

namespace nam
{
using Mutex = std::mutex;

template <typename T>
using LockGuard = std::lock_guard<T>;
} // namespace nam

#endif
