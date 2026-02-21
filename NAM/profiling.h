#pragma once

// Dynamic profiling registry for NAM building blocks
// Enable with -DNAM_PROFILING
//
// Usage:
//   1. Register profiling types at file scope (static init):
//        static int PROF_FOO = nam::profiling::register_type("Foo");
//   2. Call nam::profiling::reset() before benchmark
//   3. In hot path:
//        NAM_PROFILE_START();
//        // ... code ...
//        NAM_PROFILE_ADD(PROF_FOO);
//   4. Call nam::profiling::print_results() to display breakdown

#ifdef NAM_PROFILING

  #include <cstdint>
  #include <cstdio>

namespace nam
{
namespace profiling
{

constexpr int MAX_PROFILING_TYPES = 32;

struct ProfilingEntry
{
  const char* name;
  uint32_t accumulated_us;
};

extern ProfilingEntry g_entries[MAX_PROFILING_TYPES];
extern int g_num_entries;

// Register a named profiling type. Returns index for fast accumulation.
// Called at static-init time or during setup, NOT in the hot path.
int register_type(const char* name);

// Get current time in microseconds (platform-specific)
uint32_t get_time_us();

// Reset all profiling counters
void reset();

// Print profiling results to stdout
void print_results();

// Helper macros for timing sections
// Usage:
//   NAM_PROFILE_START();
//   // ... code to profile ...
//   NAM_PROFILE_ADD(PROF_FOO);  // Adds elapsed time to entry, resets timer

  #define NAM_PROFILE_START() uint32_t _prof_start = nam::profiling::get_time_us()
  #define NAM_PROFILE_ADD(idx)                                                                                         \
    do                                                                                                                 \
    {                                                                                                                  \
      uint32_t _prof_now = nam::profiling::get_time_us();                                                              \
      nam::profiling::g_entries[idx].accumulated_us += (_prof_now - _prof_start);                                      \
      _prof_start = _prof_now;                                                                                         \
    } while (0)

  // Variant that doesn't reset the timer (for one-shot measurements)
  #define NAM_PROFILE_ADD_NORESTART(idx)                                                                               \
    nam::profiling::g_entries[idx].accumulated_us += (nam::profiling::get_time_us() - _prof_start)

  // Reset the timer without recording (for re-syncing mid-function)
  #define NAM_PROFILE_RESTART() _prof_start = nam::profiling::get_time_us()

} // namespace profiling
} // namespace nam

#else // NAM_PROFILING not defined

  // No-op macros when profiling is disabled
  #define NAM_PROFILE_START() ((void)0)
  #define NAM_PROFILE_ADD(idx) ((void)0)
  #define NAM_PROFILE_ADD_NORESTART(idx) ((void)0)
  #define NAM_PROFILE_RESTART() ((void)0)

namespace nam
{
namespace profiling
{
inline void reset() {}
inline void print_results() {}
} // namespace profiling
} // namespace nam

#endif // NAM_PROFILING
