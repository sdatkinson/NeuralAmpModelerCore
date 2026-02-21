#include "profiling.h"

#ifdef NAM_PROFILING

  #if defined(__ARM_ARCH_7EM__) || defined(ARM_MATH_CM7)
    // ARM Cortex-M7: Use DWT cycle counter for precise timing
    #include "stm32h7xx.h"

namespace nam
{
namespace profiling
{

ProfilingEntry g_entries[MAX_PROFILING_TYPES] = {};
int g_num_entries = 0;

// CPU frequency in MHz (Daisy runs at 480 MHz)
static constexpr uint32_t CPU_FREQ_MHZ = 480;

uint32_t get_time_us()
{
  // DWT->CYCCNT gives cycle count
  // Divide by CPU_FREQ_MHZ to get microseconds
  return DWT->CYCCNT / CPU_FREQ_MHZ;
}

} // namespace profiling
} // namespace nam

  #else
    // Non-ARM: Use std::chrono for timing (for testing on desktop)
    #include <chrono>

namespace nam
{
namespace profiling
{

ProfilingEntry g_entries[MAX_PROFILING_TYPES] = {};
int g_num_entries = 0;

uint32_t get_time_us()
{
  using namespace std::chrono;
  static auto start = high_resolution_clock::now();
  auto now = high_resolution_clock::now();
  return (uint32_t)duration_cast<microseconds>(now - start).count();
}

} // namespace profiling
} // namespace nam

  #endif // ARM check

namespace nam
{
namespace profiling
{

int register_type(const char* name)
{
  int idx = g_num_entries++;
  g_entries[idx].name = name;
  g_entries[idx].accumulated_us = 0;
  return idx;
}

void reset()
{
  for (int i = 0; i < g_num_entries; i++)
    g_entries[i].accumulated_us = 0;
}

void print_results()
{
  uint32_t total = 0;
  for (int i = 0; i < g_num_entries; i++)
    total += g_entries[i].accumulated_us;

  printf("\nProfiling breakdown:\n");
  printf("%-12s %8s %6s\n", "Category", "Time(ms)", "%");
  printf("%-12s %8s %6s\n", "--------", "--------", "----");

  for (int i = 0; i < g_num_entries; i++)
  {
    uint32_t us = g_entries[i].accumulated_us;
    if (us > 0)
    {
      uint32_t pct = total > 0 ? (us * 100 / total) : 0;
      printf("%-12s %8.1f %5lu%%\n", g_entries[i].name, us / 1000.0f, (unsigned long)pct);
    }
  }

  printf("%-12s %8s %6s\n", "--------", "--------", "----");
  printf("%-12s %8.1f %5s\n", "Total", total / 1000.0f, "100%");
}

} // namespace profiling
} // namespace nam

#endif // NAM_PROFILING
