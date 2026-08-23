#include "NAM/linear.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

namespace
{
using Clock = std::chrono::steady_clock;

std::vector<float> make_weights(const int taps)
{
  std::vector<float> weights(taps);
  for (int i = 0; i < taps; ++i)
    weights[i] = 0.001f * std::exp(-6.0f * (float)i / (float)taps) * std::sin(0.031f * (float)(i + 1));
  return weights;
}

double percentile(const std::vector<double>& sorted, const double fraction)
{
  const size_t index = std::min(sorted.size() - 1, (size_t)std::floor(fraction * (double)(sorted.size() - 1)));
  return sorted[index];
}

bool verify_impulse_response(const int taps, const int callback_size, const int sample_rate)
{
  auto weights = make_weights(taps);
  nam::Linear model(1, 1, taps, false, weights, sample_rate, nam::LinearImplementation::FFT);
  model.Reset(sample_rate, callback_size);

  std::vector<NAM_SAMPLE> input(callback_size, 0.0);
  std::vector<NAM_SAMPLE> output(callback_size, 0.0);
  NAM_SAMPLE* inputs[] = {input.data()};
  NAM_SAMPLE* outputs[] = {output.data()};
  double max_abs_error = 0.0;
  int sample = 0;
  const int samples_to_process = taps + 2 * callback_size;
  while (sample < samples_to_process)
  {
    std::fill(input.begin(), input.end(), 0.0);
    if (sample == 0)
      input[0] = 1.0;
    const int frames = std::min(callback_size, samples_to_process - sample);
    model.process(inputs, outputs, frames);
    for (int i = 0; i < frames; ++i)
    {
      const double expected = sample + i < taps ? weights[sample + i] : 0.0;
      max_abs_error = std::max(max_abs_error, std::abs((double)output[i] - expected));
    }
    sample += frames;
  }
  std::cout << "verified_taps=" << taps << ",max_abs_error=" << std::scientific << max_abs_error << '\n';
  return max_abs_error < 5.0e-5;
}

void run(const int taps, const int callback_size, const int sample_rate, const double seconds,
         const nam::LinearImplementation implementation, const bool cold_cache)
{
  constexpr int max_callback_size = 512;
  auto weights = make_weights(taps);
  nam::Linear model(1, 1, taps, false, weights, sample_rate, implementation);
  model.Reset(sample_rate, max_callback_size);

  std::vector<NAM_SAMPLE> input(max_callback_size);
  std::vector<NAM_SAMPLE> output(max_callback_size);
  for (int i = 0; i < max_callback_size; ++i)
    input[i] = (NAM_SAMPLE)(0.1 * std::sin(0.017 * i) + 0.03 * std::cos(0.043 * i));
  NAM_SAMPLE* inputs[] = {input.data()};
  NAM_SAMPLE* outputs[] = {output.data()};

  for (int i = 0; i < 4096 / callback_size; ++i)
    model.process(inputs, outputs, callback_size);

  const int callbacks = std::max(1, (int)std::ceil(seconds * sample_rate / callback_size));
  std::vector<double> durations;
  durations.reserve(callbacks);
  std::vector<unsigned char> cache_polluter(cold_cache ? 64 * 1024 * 1024 : 0, 1);
  volatile NAM_SAMPLE sink = 0.0;
  volatile unsigned int cache_sink = 0;
  for (int i = 0; i < callbacks; ++i)
  {
    for (size_t byte = 0; byte < cache_polluter.size(); byte += 64)
      cache_sink = cache_sink + cache_polluter[byte];
    const auto start = Clock::now();
    model.process(inputs, outputs, callback_size);
    const auto end = Clock::now();
    sink = sink + output[i % callback_size];
    durations.push_back(std::chrono::duration<double, std::micro>(end - start).count());
  }

  std::sort(durations.begin(), durations.end());
  const double deadline = 1.0e6 * callback_size / sample_rate;
  const auto overruns =
    std::count_if(durations.begin(), durations.end(), [deadline](double us) { return us > deadline; });
  const double mean = std::accumulate(durations.begin(), durations.end(), 0.0) / durations.size();
  std::cout << taps << ',' << callback_size << ',' << std::fixed << std::setprecision(2) << mean << ','
            << percentile(durations, 0.50) << ',' << percentile(durations, 0.95) << ',' << percentile(durations, 0.99)
            << ',' << durations.back() << ',' << deadline << ',' << overruns << ',' << sink + cache_sink * 0.0f << '\n';
}
} // namespace

int main(int argc, char** argv)
{
  const int sample_rate = 48000;
  const double seconds = argc > 3 ? std::atof(argv[3]) : 2.0;
  const nam::LinearImplementation implementation =
    argc > 4 && std::string(argv[4]) == "direct" ? nam::LinearImplementation::Direct : nam::LinearImplementation::FFT;
  const bool cold_cache = argc > 5 && std::string(argv[5]) == "cold";
  const bool verify = argc > 5 && std::string(argv[5]) == "verify";
  const std::vector<int> taps = argc > 1 ? std::vector<int>{std::atoi(argv[1])}
                                         : std::vector<int>{1024, 2048, 4096, 8192, 48000, 240000, 1200000, 2880000};
  const std::vector<int> callbacks = argc > 2 ? std::vector<int>{std::atoi(argv[2])} : std::vector<int>{32, 64, 128};

  if (verify)
    return verify_impulse_response(taps.front(), callbacks.front(), sample_rate) ? 0 : 1;

  std::cout << "taps,callback,mean_us,p50_us,p95_us,p99_us,max_us,deadline_us,overruns,sink\n";
  for (const int tap_count : taps)
    for (const int callback_size : callbacks)
      run(tap_count, callback_size, sample_rate, seconds, implementation, cold_cache);
  return 0;
}
