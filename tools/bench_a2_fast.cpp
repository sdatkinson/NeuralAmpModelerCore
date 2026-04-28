// Side-by-side benchmark for the A2 fast-path WaveNet vs the generic WaveNet.
//
// Loads a .nam file once, constructs two DSPs from the same config + weights:
//   - fast: via a2_fast::create_a2_fast_config (explicit A2 factory)
//   - generic: via wavenet::parse_config_json (bypassing the dispatcher's A2 shortcut)
// Runs both over the same audio, reports median wall time + real-time factor.
//
// Usage:
//   bench_a2_fast [--buffer N] [--seconds S] [--iters I] <model.nam> [<model.nam> ...]
//
// Only compiled when NAM_ENABLE_A2_FAST is defined.

#if defined(NAM_ENABLE_A2_FAST)

  #include <algorithm>
  #include <chrono>
  #include <cmath>
  #include <cstdlib>
  #include <cstring>
  #include <filesystem>
  #include <fstream>
  #include <iomanip>
  #include <iostream>
  #include <memory>
  #include <sstream>
  #include <string>
  #include <utility>
  #include <vector>

  #include "json.hpp"

  #include "NAM/dsp.h"
  #include "NAM/wavenet/a2_fast.h"
  #include "NAM/wavenet/model.h"

using clock_t_hr = std::chrono::high_resolution_clock;
using ns = std::chrono::nanoseconds;

namespace
{

struct Options
{
  int buffer_size = 64;
  double seconds = 2.0;
  int iterations = 10;
  std::vector<std::string> model_paths;
};

Options parse_args(int argc, char** argv)
{
  Options o;
  for (int i = 1; i < argc; i++)
  {
    std::string a = argv[i];
    if (a == "--buffer" && i + 1 < argc)
      o.buffer_size = std::atoi(argv[++i]);
    else if (a == "--seconds" && i + 1 < argc)
      o.seconds = std::atof(argv[++i]);
    else if (a == "--iters" && i + 1 < argc)
      o.iterations = std::atoi(argv[++i]);
    else if (a == "-h" || a == "--help")
    {
      std::cerr << "Usage: bench_a2_fast [--buffer N] [--seconds S] [--iters I] <model.nam> ...\n";
      std::exit(0);
    }
    else
      o.model_paths.push_back(std::move(a));
  }
  return o;
}

struct LoadedModel
{
  nlohmann::json config;
  std::vector<float> weights;
  double sample_rate = 48000.0;
  std::string path;
};

LoadedModel load_nam(const std::string& path)
{
  LoadedModel m;
  m.path = path;
  std::ifstream is(path);
  if (!is)
    throw std::runtime_error("Could not open " + path);
  nlohmann::json j;
  is >> j;
  if (j.value("architecture", std::string()) != "WaveNet")
    throw std::runtime_error(path + ": not a WaveNet model");
  m.config = j["config"];
  m.weights = j["weights"].get<std::vector<float>>();
  if (j.contains("sample_rate") && !j["sample_rate"].is_null())
    m.sample_rate = j["sample_rate"].get<double>();
  return m;
}

struct Stats
{
  double min;
  double p50;
  double p99;
  double p999;
  double max;
  double mean;
  size_t n;
};

double percentile(const std::vector<double>& sorted_xs, double pct)
{
  if (sorted_xs.empty())
    return 0.0;
  const double idx = pct * (sorted_xs.size() - 1);
  const size_t i = static_cast<size_t>(idx);
  const double frac = idx - i;
  if (i + 1 >= sorted_xs.size())
    return sorted_xs[i];
  return sorted_xs[i] * (1.0 - frac) + sorted_xs[i + 1] * frac;
}

Stats compute_stats(std::vector<double> xs)
{
  Stats s{};
  if (xs.empty())
    return s;
  std::sort(xs.begin(), xs.end());
  double sum = 0.0;
  for (double x : xs)
    sum += x;
  s.n = xs.size();
  s.min = xs.front();
  s.max = xs.back();
  s.mean = sum / xs.size();
  s.p50 = percentile(xs, 0.50);
  s.p99 = percentile(xs, 0.99);
  s.p999 = percentile(xs, 0.999);
  return s;
}

// Run one full iteration, timing each block separately. Returns per-block times
// in milliseconds, appended to `out_times`.
void time_iteration_per_block(nam::DSP& dsp, std::vector<double>& input, std::vector<double>& output, int total,
                              int buffer_size, std::vector<double>& out_times)
{
  const double* in_base = input.data();
  double* out_base = output.data();
  int pos = 0;
  while (pos < total)
  {
    const int n = std::min(buffer_size, total - pos);
    const double* in_ptr = in_base + pos;
    double* out_ptr = out_base + pos;
    const double* in_arr[] = {in_ptr};
    double* out_arr[] = {out_ptr};
    auto t0 = clock_t_hr::now();
    dsp.process(const_cast<double**>(in_arr), out_arr, n);
    auto t1 = clock_t_hr::now();
    out_times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    pos += n;
  }
}

void bench_model(const LoadedModel& m, const Options& o)
{
  int channels = 0;
  if (!nam::wavenet::a2_fast::is_a2_shape(m.config, &channels))
  {
    std::cerr << "[skip] " << m.path << ": not an A2-shaped WaveNet\n";
    return;
  }

  // Build fast path.
  auto fast_cfg = nam::wavenet::a2_fast::create_a2_fast_config(m.config, m.sample_rate);
  std::vector<float> w_fast = m.weights;
  auto fast_dsp = fast_cfg->create(std::move(w_fast), m.sample_rate);

  // Build generic path (bypass dispatcher).
  auto generic_cfg = nam::wavenet::parse_config_json(m.config, m.sample_rate);
  std::vector<float> w_gen = m.weights;
  auto generic_dsp = generic_cfg.create(std::move(w_gen), m.sample_rate);

  const int total = static_cast<int>(o.seconds * m.sample_rate);
  std::vector<double> input(total);
  std::vector<double> output(total, 0.0);
  for (int i = 0; i < total; i++)
  {
    const double t = static_cast<double>(i) / m.sample_rate;
    input[i] = 0.25 * std::sin(2.0 * M_PI * 220.0 * t) + 0.10 * std::sin(2.0 * M_PI * 1230.0 * t);
  }

  fast_dsp->Reset(m.sample_rate, o.buffer_size);
  generic_dsp->Reset(m.sample_rate, o.buffer_size);

  // Warm up caches with one untimed iteration each.
  {
    std::vector<double> discard;
    time_iteration_per_block(*fast_dsp, input, output, total, o.buffer_size, discard);
    discard.clear();
    time_iteration_per_block(*generic_dsp, input, output, total, o.buffer_size, discard);
  }

  std::vector<double> fast_block_times;
  std::vector<double> gen_block_times;
  const int blocks_per_iter = (total + o.buffer_size - 1) / o.buffer_size;
  fast_block_times.reserve(o.iterations * blocks_per_iter);
  gen_block_times.reserve(o.iterations * blocks_per_iter);
  for (int it = 0; it < o.iterations; it++)
  {
    time_iteration_per_block(*fast_dsp, input, output, total, o.buffer_size, fast_block_times);
    time_iteration_per_block(*generic_dsp, input, output, total, o.buffer_size, gen_block_times);
  }

  const Stats fast_s = compute_stats(fast_block_times);
  const Stats gen_s = compute_stats(gen_block_times);
  const double block_audio_us = 1e6 * o.buffer_size / m.sample_rate;

  const std::string arch = (channels == 3) ? "A2 nano" : (channels == 8 ? "A2 standard" : "A2 unknown");

  auto fmt_us = [](double ms) { return ms * 1000.0; };
  std::cout << "\n== " << m.path << "  (" << arch << ", Channels=" << channels << ") ==\n";
  std::cout << "   audio=" << std::fixed << std::setprecision(0) << (1000.0 * total / m.sample_rate) << "ms @ "
            << static_cast<int>(m.sample_rate) << "Hz, block=" << o.buffer_size << ", iters=" << o.iterations << " ("
            << fast_s.n << " blocks timed each)\n";
  std::cout << "   per-block audio deadline: " << std::fixed << std::setprecision(1) << block_audio_us << " us\n";
  std::cout << std::fixed << std::setprecision(2);
  std::cout << "                  min     p50     p99     p99.9   max    mean\n";
  std::cout << "   fast    (us): " << std::setw(7) << fmt_us(fast_s.min) << " " << std::setw(7) << fmt_us(fast_s.p50)
            << " " << std::setw(7) << fmt_us(fast_s.p99) << " " << std::setw(7) << fmt_us(fast_s.p999) << " "
            << std::setw(7) << fmt_us(fast_s.max) << " " << std::setw(7) << fmt_us(fast_s.mean) << "\n";
  std::cout << "   generic (us): " << std::setw(7) << fmt_us(gen_s.min) << " " << std::setw(7) << fmt_us(gen_s.p50)
            << " " << std::setw(7) << fmt_us(gen_s.p99) << " " << std::setw(7) << fmt_us(gen_s.p999) << " "
            << std::setw(7) << fmt_us(gen_s.max) << " " << std::setw(7) << fmt_us(gen_s.mean) << "\n";
  const double fast_rtf = (block_audio_us / 1000.0) / fast_s.p50;
  const double gen_rtf = (block_audio_us / 1000.0) / gen_s.p50;
  const double speedup = gen_s.p50 / fast_s.p50;
  std::cout << "   RTF (p50):  fast=" << fast_rtf << "x  generic=" << gen_rtf << "x  speedup=" << speedup << "x\n";
}

} // namespace

int main(int argc, char** argv)
{
  Options o = parse_args(argc, argv);
  if (o.model_paths.empty())
  {
    std::cerr << "Usage: bench_a2_fast [--buffer N] [--seconds S] [--iters I] <model.nam> ...\n";
    return 1;
  }
  for (const auto& p : o.model_paths)
  {
    try
    {
      bench_model(load_nam(p), o);
    }
    catch (const std::exception& e)
    {
      std::cerr << "[error] " << p << ": " << e.what() << "\n";
    }
  }
  return 0;
}

#else
int main()
{
  std::cerr << "bench_a2_fast: rebuild with NAM_ENABLE_A2_FAST=ON.\n";
  return 1;
}
#endif
