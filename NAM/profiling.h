#pragma once

// Comprehensive profiling for NAM building blocks
// Enable with -DNAM_PROFILING
//
// Usage:
//   1. Call nam::profiling::reset() before benchmark
//   2. Run model processing
//   3. Call nam::profiling::print_results() to display breakdown
//
// Categories cover all WaveNet operations including FiLM modulation.

#ifdef NAM_PROFILING

#include <cstdint>
#include <cstdio>

namespace nam {
namespace profiling {

// Timing accumulators (in microseconds)
struct Timings {
  // Dilated convolution (Conv1D)
  uint32_t conv1d = 0;

  // Pointwise convolutions (Conv1x1 variants)
  uint32_t input_mixin = 0;   // Input mixing Conv1x1
  uint32_t layer1x1 = 0;      // Layer 1x1 (residual projection)
  uint32_t head1x1 = 0;       // Head 1x1 (skip connection projection)
  uint32_t rechannel = 0;     // Rechannel Conv1x1 (input/output)
  uint32_t conv1x1 = 0;       // Other Conv1x1 (catch-all for non-WaveNet uses)

  // Activation
  uint32_t activation = 0;    // Activation functions (tanh, ReLU, Softsign, etc.)

  // FiLM modulation
  uint32_t film = 0;          // Feature-wise Linear Modulation (scale/shift)

  // Memory operations
  uint32_t copies = 0;        // Memory copies and additions
  uint32_t setzero = 0;       // setZero() calls
  uint32_t ringbuf = 0;       // Ring buffer operations (Write, Read, Advance)

  // Conditioning
  uint32_t condition = 0;     // Condition DSP processing

  // LSTM (for LSTM models)
  uint32_t lstm = 0;          // LSTM cell computations

  // Catch-all
  uint32_t other = 0;         // Everything else

  void reset() {
    conv1d = 0;
    input_mixin = 0;
    layer1x1 = 0;
    head1x1 = 0;
    rechannel = 0;
    conv1x1 = 0;
    activation = 0;
    film = 0;
    copies = 0;
    setzero = 0;
    ringbuf = 0;
    condition = 0;
    lstm = 0;
    other = 0;
  }

  uint32_t total() const {
    return conv1d + input_mixin + layer1x1 + head1x1 + rechannel + conv1x1 + activation + film + copies + setzero + ringbuf + condition + lstm + other;
  }
};

// Global timing accumulator
extern Timings g_timings;

// Get current time in microseconds (platform-specific)
uint32_t get_time_us();

// Reset profiling counters
inline void reset() { g_timings.reset(); }

// Print profiling results to stdout
inline void print_results() {
  const auto& t = g_timings;
  uint32_t total = t.total();

  printf("\nProfiling breakdown:\n");
  printf("%-12s %8s %6s\n", "Category", "Time(ms)", "%%");
  printf("%-12s %8s %6s\n", "--------", "--------", "----");

  auto print_row = [total](const char* name, uint32_t us) {
    if (us > 0 || total == 0) {
      uint32_t pct = total > 0 ? (us * 100 / total) : 0;
      printf("%-12s %8.1f %5lu%%\n", name, us / 1000.0f, (unsigned long)pct);
    }
  };

  print_row("Conv1D", t.conv1d);
  print_row("InputMixin", t.input_mixin);
  print_row("Layer1x1", t.layer1x1);
  print_row("Head1x1", t.head1x1);
  print_row("Rechannel", t.rechannel);
  print_row("Conv1x1", t.conv1x1);
  print_row("Activation", t.activation);
  print_row("FiLM", t.film);
  print_row("Copies", t.copies);
  print_row("SetZero", t.setzero);
  print_row("RingBuf", t.ringbuf);
  print_row("Condition", t.condition);
  print_row("LSTM", t.lstm);
  print_row("Other", t.other);

  printf("%-12s %8s %6s\n", "--------", "--------", "----");
  printf("%-12s %8.1f %5s\n", "Total", total / 1000.0f, "100%");
}

// Helper macros for timing sections
// Usage:
//   NAM_PROFILE_START();
//   // ... code to profile ...
//   NAM_PROFILE_ADD(conv1d);  // Adds elapsed time to conv1d, resets timer

#define NAM_PROFILE_START() uint32_t _prof_start = nam::profiling::get_time_us()
#define NAM_PROFILE_ADD(category) do { \
  uint32_t _prof_now = nam::profiling::get_time_us(); \
  nam::profiling::g_timings.category += (_prof_now - _prof_start); \
  _prof_start = _prof_now; \
} while(0)

// Variant that doesn't reset the timer (for one-shot measurements)
#define NAM_PROFILE_ADD_NORESTART(category) \
  nam::profiling::g_timings.category += (nam::profiling::get_time_us() - _prof_start)

} // namespace profiling
} // namespace nam

#else // NAM_PROFILING not defined

// No-op macros when profiling is disabled
#define NAM_PROFILE_START() ((void)0)
#define NAM_PROFILE_ADD(category) ((void)0)
#define NAM_PROFILE_ADD_NORESTART(category) ((void)0)

namespace nam {
namespace profiling {
  inline void reset() {}
  inline void print_results() {}
} // namespace profiling
} // namespace nam

#endif // NAM_PROFILING
