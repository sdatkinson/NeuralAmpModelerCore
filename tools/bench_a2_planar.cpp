// Head-to-head for the planar NEON A2 kernels against the reference A2 fast
// path: same model, same weights, same input, same process, in one binary.
//
// It checks before it times. Every run first renders the whole signal through
// both engines and compares the output bit for bit; if they differ it says so
// and reports no speed at all, because a speed number for a kernel that is not
// reproducing the reference is not worth having.
//
// Timing follows the shape that survived being wrong in earlier attempts:
// interference on a desktop machine is one-sided -- it can only make a pass
// slower -- so the estimate is the mean of the fastest 70% of passes rather
// than the mean or the median of all of them, and the fastest single pass is
// printed next to it so the spread is visible.
//
// Usage:
//   bench_a2_planar [--buffer N] [--seconds S] [--warmup W] [--passes P]
//                   [--submodel widest|narrowest|<index>] <model.nam> ...
//
// A .nam holding a SlimmableContainer is unwrapped and one submodel is
// measured; --submodel picks it by width, not by position, so a reordering in
// the trainer cannot silently change what is being measured.
//
// There is only something to measure where the planar kernels exist, so on any
// other target this builds to a main() that says so and exits. The target is
// still built everywhere, deliberately: a tool that silently vanishes from some
// configurations is a tool nobody notices has stopped compiling.

#include <iostream>

#if defined(NAM_ENABLE_A2_FAST)
  #include "NAM/wavenet/a2_planar.h" // defines NAM_A2_PLANAR where it applies
#endif

#if defined(NAM_A2_PLANAR)

  #include <algorithm>
  #include <chrono>
  #include <cmath>
  #include <cstdlib>
  #include <cstring>
  #include <fstream>
  #include <iomanip>
  #include <memory>
  #include <string>
  #include <utility>
  #include <vector>

  #include "json.hpp"

  #include "NAM/dsp.h"
  #include "NAM/wavenet/a2_fast.h"

using hr_clock = std::chrono::high_resolution_clock;

namespace
{

struct Options
{
  int buffer_size = 64;
  double seconds = 10.9; // one pass of audio
  double warmup_seconds = 5.0; // discarded
  int passes = 12; // timed
  double accept_fraction = 0.7;
  std::string submodel = "widest";
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
    else if (a == "--warmup" && i + 1 < argc)
      o.warmup_seconds = std::atof(argv[++i]);
    else if (a == "--passes" && i + 1 < argc)
      o.passes = std::atoi(argv[++i]);
    else if (a == "--submodel" && i + 1 < argc)
      o.submodel = argv[++i];
    else if (a == "-h" || a == "--help")
    {
      std::cerr << "Usage: bench_a2_planar [--buffer N] [--seconds S] [--warmup W] [--passes P]\n"
                << "                       [--submodel widest|narrowest|<index>] <model.nam> ...\n";
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
  std::string note; // which submodel, when unwrapped
};

/// Pull one WaveNet out of a .nam, unwrapping a SlimmableContainer if that is
/// what it holds. Selection is by channel count, so it does not depend on the
/// order the submodels happen to be stored in.
LoadedModel load_nam(const std::string& path, const std::string& submodel)
{
  std::ifstream is(path);
  if (!is)
    throw std::runtime_error("Could not open " + path);
  nlohmann::json j;
  is >> j;

  const std::string arch = j.value("architecture", std::string());
  nlohmann::json wavenet = j;
  std::string note;

  if (arch == "SlimmableContainer")
  {
    const auto& subs = j.at("config").at("submodels");
    if (!subs.is_array() || subs.empty())
      throw std::runtime_error(path + ": SlimmableContainer has no submodels");

    int chosen = -1;
    if (submodel == "widest" || submodel == "narrowest")
    {
      int best = -1;
      for (size_t i = 0; i < subs.size(); i++)
      {
        const auto& m = subs[i].at("model");
        const int ch = m.at("config").at("layers")[0].value("channels", 0);
        const bool better = (chosen < 0) || (submodel == "widest" ? ch > best : ch < best);
        if (better)
        {
          best = ch;
          chosen = static_cast<int>(i);
        }
      }
    }
    else
    {
      chosen = std::atoi(submodel.c_str());
      if (chosen < 0 || chosen >= static_cast<int>(subs.size()))
        throw std::runtime_error(path + ": no submodel " + submodel);
    }

    wavenet = subs[chosen].at("model");
    note = "submodel " + std::to_string(chosen) + " of " + std::to_string(subs.size());
  }
  else if (arch != "WaveNet")
  {
    throw std::runtime_error(path + ": not a WaveNet or SlimmableContainer model");
  }

  LoadedModel m;
  m.path = path;
  m.note = note;
  m.config = wavenet.at("config");
  m.weights = wavenet.at("weights").get<std::vector<float>>();
  if (wavenet.contains("sample_rate") && !wavenet["sample_rate"].is_null())
    m.sample_rate = wavenet["sample_rate"].get<double>();
  return m;
}

/// One full pass over the signal, in blocks. Returns wall time in milliseconds.
double run_pass(nam::DSP& dsp, const std::vector<NAM_SAMPLE>& input, std::vector<NAM_SAMPLE>& output, int buffer_size)
{
  const int total = static_cast<int>(input.size());
  const auto t0 = hr_clock::now();
  int pos = 0;
  while (pos < total)
  {
    const int n = std::min(buffer_size, total - pos);
    const NAM_SAMPLE* in_ptr = input.data() + pos;
    NAM_SAMPLE* out_ptr = output.data() + pos;
    const NAM_SAMPLE* in_arr[] = {in_ptr};
    NAM_SAMPLE* out_arr[] = {out_ptr};
    dsp.process(const_cast<NAM_SAMPLE**>(in_arr), out_arr, n);
    pos += n;
  }
  const auto t1 = hr_clock::now();
  return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

struct Timing
{
  double mean = 0.0; // of the fastest `accept_fraction` of passes
  double fastest = 0.0;
  double slowest = 0.0;
  int kept = 0;
};

Timing summarise(std::vector<double> times, double accept_fraction)
{
  Timing t;
  if (times.empty())
    return t;
  std::sort(times.begin(), times.end());
  t.fastest = times.front();
  t.slowest = times.back();
  t.kept = std::max(1, static_cast<int>(times.size() * accept_fraction));
  double sum = 0.0;
  for (int i = 0; i < t.kept; i++)
    sum += times[i];
  t.mean = sum / t.kept;
  return t;
}

/// Bit-for-bit, not within a tolerance. Returns the index of the first
/// difference, or -1.
long long first_difference(const std::vector<NAM_SAMPLE>& a, const std::vector<NAM_SAMPLE>& b)
{
  if (std::memcmp(a.data(), b.data(), a.size() * sizeof(NAM_SAMPLE)) == 0)
    return -1;
  for (size_t i = 0; i < a.size(); i++)
    if (a[i] != b[i])
      return static_cast<long long>(i);
  return 0; // differing bits in equal values (a signed zero); still a difference
}

bool bench_model(const LoadedModel& m, const Options& o)
{
  int channels = 0;
  if (!nam::wavenet::a2_fast::is_a2_shape(m.config, &channels))
  {
    std::cerr << "[skip] " << m.path << ": not an A2-shaped WaveNet\n";
    return true;
  }

  auto reference = nam::wavenet::a2_fast::create_a2_fast_reference_model(channels, m.weights, m.sample_rate);
  auto planar = nam::wavenet::a2_fast::create_a2_planar_model(channels, m.weights, m.sample_rate);
  if (planar == nullptr)
  {
    std::cerr << "[skip] " << m.path << ": no planar kernel for " << channels << " channels on this target\n";
    return true;
  }

  const int total = static_cast<int>(o.seconds * m.sample_rate);
  std::vector<NAM_SAMPLE> input(total);
  for (int i = 0; i < total; i++)
  {
    const double t = static_cast<double>(i) / m.sample_rate;
    input[i] =
      static_cast<NAM_SAMPLE>(0.25 * std::sin(2.0 * M_PI * 220.0 * t) + 0.10 * std::sin(2.0 * M_PI * 1230.0 * t)
                              + 0.05 * std::sin(2.0 * M_PI * 3170.0 * t));
  }
  std::vector<NAM_SAMPLE> out_reference(total, static_cast<NAM_SAMPLE>(0));
  std::vector<NAM_SAMPLE> out_planar(total, static_cast<NAM_SAMPLE>(0));

  reference->Reset(m.sample_rate, o.buffer_size);
  planar->Reset(m.sample_rate, o.buffer_size);

  const std::string arch = (channels == 3) ? "A2 nano" : "A2 standard";
  std::cout << "\n== " << m.path << (m.note.empty() ? "" : ("  [" + m.note + "]")) << "\n"
            << "   " << arch << ", " << channels << " channels, " << m.weights.size() << " weights; " << std::fixed
            << std::setprecision(2) << (total / m.sample_rate) << " s of audio per pass at "
            << static_cast<int>(m.sample_rate) << " Hz, " << o.buffer_size << "-frame blocks\n";

  // --- Parity, before anything is timed ---------------------------------------
  run_pass(*reference, input, out_reference, o.buffer_size);
  run_pass(*planar, input, out_planar, o.buffer_size);
  const long long diff = first_difference(out_reference, out_planar);
  if (diff >= 0)
  {
    std::cerr << "   PARITY FAILED: first difference at sample " << diff << " of " << total
              << " (reference=" << out_reference[static_cast<size_t>(diff)]
              << ", planar=" << out_planar[static_cast<size_t>(diff)] << "). Not timing.\n";
    return false;
  }
  std::cout << "   parity: bit-identical over all " << total << " frames\n";

  // --- Warm up, then time -----------------------------------------------------
  const int warmup_passes = std::max(1, static_cast<int>(o.warmup_seconds / o.seconds + 0.5));
  for (int i = 0; i < warmup_passes; i++)
  {
    run_pass(*reference, input, out_reference, o.buffer_size);
    run_pass(*planar, input, out_planar, o.buffer_size);
  }

  std::vector<double> t_reference, t_planar;
  t_reference.reserve(o.passes);
  t_planar.reserve(o.passes);
  for (int i = 0; i < o.passes; i++)
  {
    // Interleaved, so a slow patch on the machine lands on both.
    t_reference.push_back(run_pass(*reference, input, out_reference, o.buffer_size));
    t_planar.push_back(run_pass(*planar, input, out_planar, o.buffer_size));
  }

  const Timing r = summarise(t_reference, o.accept_fraction);
  const Timing p = summarise(t_planar, o.accept_fraction);
  const double audio_ms = 1000.0 * total / m.sample_rate;

  std::cout << std::fixed << std::setprecision(2);
  std::cout << "   " << o.passes << " passes, mean of the fastest " << r.kept << "\n";
  std::cout << "                   mean/pass   fastest   slowest   x real time\n";
  std::cout << "   a2_fast         " << std::setw(8) << r.mean << " ms " << std::setw(9) << r.fastest << " "
            << std::setw(9) << r.slowest << "   " << std::setw(8) << (audio_ms / r.mean) << "\n";
  std::cout << "   planar NEON     " << std::setw(8) << p.mean << " ms " << std::setw(9) << p.fastest << " "
            << std::setw(9) << p.slowest << "   " << std::setw(8) << (audio_ms / p.mean) << "\n";
  std::cout << std::setprecision(3) << "   speedup: " << (r.mean / p.mean) << "x on the mean, "
            << (r.fastest / p.fastest) << "x on the fastest pass\n";
  return true;
}

} // namespace

int main(int argc, char** argv)
{
  const Options o = parse_args(argc, argv);
  if (o.model_paths.empty())
  {
    std::cerr << "Usage: bench_a2_planar [options] <model.nam> ...  (--help for options)\n";
    return 2;
  }

  bool ok = true;
  for (const auto& path : o.model_paths)
  {
    try
    {
      ok = bench_model(load_nam(path, o.submodel), o) && ok;
    }
    catch (const std::exception& e)
    {
      std::cerr << "[error] " << path << ": " << e.what() << "\n";
      ok = false;
    }
  }
  return ok ? 0 : 1;
}

#else // NAM_A2_PLANAR

int main()
{
  // Not an error: there is simply no planar kernel in this build to compare
  // against, either because NAM_ENABLE_A2_FAST is off, because the target is
  // not Apple Silicon, or because NAM_DISABLE_A2_PLANAR was set.
  std::cout << "bench_a2_planar: this build has no planar A2 kernel; nothing to measure.\n";
  return 0;
}

#endif // NAM_A2_PLANAR
