#include "profiling.h"

#ifdef NAM_PROFILING

#if defined(__ARM_ARCH_7EM__) || defined(ARM_MATH_CM7)
// ARM Cortex-M7: Use DWT cycle counter for precise timing
#include "stm32h7xx.h"

namespace nam {
namespace profiling {

Timings g_timings;

// CPU frequency in MHz (Daisy runs at 480 MHz)
static constexpr uint32_t CPU_FREQ_MHZ = 480;

uint32_t get_time_us() {
  // DWT->CYCCNT gives cycle count
  // Divide by CPU_FREQ_MHZ to get microseconds
  return DWT->CYCCNT / CPU_FREQ_MHZ;
}

} // namespace profiling
} // namespace nam

#else
// Non-ARM: Use std::chrono for timing (for testing on desktop)
#include <chrono>

namespace nam {
namespace profiling {

Timings g_timings;

uint32_t get_time_us() {
  using namespace std::chrono;
  static auto start = high_resolution_clock::now();
  auto now = high_resolution_clock::now();
  return (uint32_t)duration_cast<microseconds>(now - start).count();
}

} // namespace profiling
} // namespace nam

#endif // ARM check

#endif // NAM_PROFILING
