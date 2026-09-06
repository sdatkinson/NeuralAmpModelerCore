#pragma once

// Portability layer. Everything that differs between a desktop plugin build and
// a constrained embedded build is decided here, so the rest of the library can be
// written as plain C++ without conditional compilation at the call sites.
//
// Opt-in switches, defined on the compiler command line:
//   NAM_NO_EXCEPTIONS  target is built with -fno-exceptions
//   NAM_NO_MUTEX       target is single threaded
//   NAM_NO_JSON        drop the JSON (.nam) loader; binary (.namb) loading only
//   NAM_NO_FILESYSTEM  target has no std::filesystem; drop path-based loading

#include <cstdlib>

#if !defined(NAM_NO_EXCEPTIONS)
  #include <stdexcept>
#endif

#if !defined(NAM_NO_MUTEX)
  #include <mutex>
#endif

#if defined(_MSC_VER) && !defined(__llvm__)
  #define NAM_RESTRICT __restrict
#else
  #define NAM_RESTRICT __restrict__
#endif

#if defined(NAM_NO_JSON)
  #define NAM_HAS_JSON 0
#else
  #define NAM_HAS_JSON 1
#endif

#if defined(NAM_NO_FILESYSTEM)
  #define NAM_HAS_FILESYSTEM 0
#else
  #define NAM_HAS_FILESYSTEM 1
#endif

#ifndef NAM_SECTION_CODE_FAST
  // Intentionally empty by default; target builds may place hot code in fast memory.
  #define NAM_SECTION_CODE_FAST
#endif

namespace nam
{
namespace detail
{
/// \brief Terminal error handler used when exceptions are disabled
/// \param what Stringified exception expression; never evaluated, so failure paths
///             cost no code and perform no allocation
[[noreturn]] inline void fail(const char* what)
{
  (void)what;
  std::abort();
}
} // namespace detail

#if defined(NAM_NO_MUTEX)
class Mutex
{
public:
  void lock() {}
  void unlock() {}
};

class LockGuard
{
public:
  explicit LockGuard(Mutex&) {}
};
#else
using Mutex = std::mutex;
using LockGuard = std::lock_guard<std::mutex>;
#endif
} // namespace nam

#if defined(NAM_NO_EXCEPTIONS)
  #define NAM_THROW(...) ::nam::detail::fail(#__VA_ARGS__)
#else
  #define NAM_THROW(...) throw __VA_ARGS__
#endif