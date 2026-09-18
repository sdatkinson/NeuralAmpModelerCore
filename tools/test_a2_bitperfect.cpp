// Compare the current A2 implementation with an untouched checkout, compiled
// into a separate translation unit under the a2_reference namespace.
//
// Configure with -DNAM_A2_REFERENCE_DIR=/absolute/path/to/reference/checkout.
// Run from the repository root: test_a2_bitperfect [--bench | --bench-small | --latency | --negative-control]
// Assertions and floating-point tolerances are deliberately not used.

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "NAM/dsp.h"
#include "NAM/wavenet/a2_fast.h"

namespace nam::wavenet::a2_reference
{
std::unique_ptr<ModelConfig> create_a2_fast_config(const nlohmann::json& config, double sampleRate);
}

namespace
{
using Bits = std::conditional_t<sizeof(NAM_SAMPLE) == 4, uint32_t, uint64_t>;
constexpr Bits kExponent = sizeof(NAM_SAMPLE) == 4 ? Bits(0x7f800000U) : Bits(0x7ff0000000000000ULL);
uint64_t compared_samples = 0;
uint64_t checksum = 14695981039346656037ULL;
int cases_passed = 0;

struct Model
{
  nlohmann::json config;
  std::vector<float> weights;
  std::string label;
};

struct Pair
{
  std::unique_ptr<nam::DSP> reference;
  std::unique_ptr<nam::DSP> candidate;

  explicit Pair(const Model& model)
  {
    auto r = nam::wavenet::a2_reference::create_a2_fast_config(model.config, 48000.0);
    auto c = nam::wavenet::a2_fast::create_a2_fast_config(model.config, 48000.0);
    reference = r->create(model.weights, 48000.0);
    candidate = c->create(model.weights, 48000.0);
  }

  void reset(int maximum)
  {
    reference->Reset(48000.0, maximum);
    candidate->Reset(48000.0, maximum);
  }
};

float noise(uint32_t& state)
{
  state ^= state << 13;
  state ^= state >> 17;
  state ^= state << 5;
  return static_cast<float>(static_cast<int>(state & 65535U) - 32768) / 32768.0f;
}

std::vector<Model> load_models(bool with_random_weights)
{
  std::ifstream stream("example_models/A2.nam");
  if (!stream)
    throw std::runtime_error("Run from the repository root; cannot open example_models/A2.nam");
  nlohmann::json j;
  stream >> j;
  std::vector<Model> models;
  for (const auto& submodel : j.at("config").at("submodels"))
  {
    const auto& spec = submodel.at("model");
    Model m{spec.at("config"), spec.at("weights").get<std::vector<float>>(), ""};
    int channels = 0;
    if (!nam::wavenet::a2_fast::is_a2_shape(m.config, &channels))
      throw std::runtime_error("Fixture no longer has the expected A2 shape");
    m.label = channels == 3 ? "A2-Lite" : "A2-Full";
    models.push_back(m);
    if (with_random_weights)
    {
      for (uint32_t seed : {0x1a2b3c4dU, 0x98765432U})
      {
        Model random = m;
        uint32_t state = seed;
        for (float& weight : random.weights)
          weight = 0.125f * noise(state);
        random.weights.back() = 0.125f;
        random.label += "-seed-" + std::to_string(seed);
        models.push_back(std::move(random));
      }
    }
  }
  return models;
}

std::vector<NAM_SAMPLE> make_input(int count)
{
  std::vector<NAM_SAMPLE> input(count);
  uint32_t state = 0xa2219e37U;
  for (int i = 0; i < count; ++i)
  {
    const float n = noise(state);
    const int offset = i % 1024;
    const double t = i / 48000.0;
    switch ((i / 1024) % 8)
    {
      case 0: input[i] = (i & 1) ? NAM_SAMPLE(-0.0) : NAM_SAMPLE(0.0); break;
      case 1: input[i] = offset == 0 ? NAM_SAMPLE(1.0) : NAM_SAMPLE(0.0); break;
      case 2: input[i] = (i & 1) ? NAM_SAMPLE(-1.0) : NAM_SAMPLE(1.0); break;
      case 3: input[i] = NAM_SAMPLE(0.3); break;
      case 4: input[i] = NAM_SAMPLE(0.25 * std::sin(1382.3 * t) + 0.1 * std::sin(7738.9 * t)); break;
      case 5: input[i] = NAM_SAMPLE(4.0f * n); break;
      case 6: input[i] = NAM_SAMPLE(1.0e-6f * n); break;
      case 7: input[i] = NAM_SAMPLE((i & 1 ? -1.0 : 1.0) * std::numeric_limits<float>::denorm_min()); break;
    }
  }
  return input;
}

void compare(NAM_SAMPLE expected, NAM_SAMPLE actual, const std::string& label, size_t position)
{
  const Bits eb = std::bit_cast<Bits>(expected);
  const Bits ab = std::bit_cast<Bits>(actual);
  if ((eb & kExponent) == kExponent || (ab & kExponent) == kExponent)
    throw std::runtime_error(label + ": non-finite output at sample " + std::to_string(position));
  if (std::memcmp(&expected, &actual, sizeof(NAM_SAMPLE)) != 0)
  {
    std::cerr << label << ": sample " << position << " expected bits 0x" << std::hex << eb
              << ", actual bits 0x" << ab << std::dec << '\n';
    throw std::runtime_error("Bit-perfect comparison FAILED");
  }
  ++compared_samples;
  checksum = (checksum ^ static_cast<uint64_t>(ab)) * 1099511628211ULL;
}

void compare_stream(Pair& dsp, const std::vector<NAM_SAMPLE>& input, const std::vector<int>& schedule,
                    const std::string& label, bool in_place)
{
  const int maximum = *std::max_element(schedule.begin(), schedule.end());
  std::vector<NAM_SAMPLE> in(maximum), expected(maximum), actual(maximum);
  size_t position = 0, block = 0;
  while (position < input.size())
  {
    const int n = std::min<size_t>(schedule[block++ % schedule.size()], input.size() - position);
    std::copy_n(input.data() + position, n, in.data());
    std::fill(expected.begin(), expected.end(), std::numeric_limits<NAM_SAMPLE>::quiet_NaN());
    std::fill(actual.begin(), actual.end(), std::numeric_limits<NAM_SAMPLE>::quiet_NaN());
    if (in_place)
    {
      std::copy_n(in.data(), n, expected.data());
      std::copy_n(in.data(), n, actual.data());
    }
    NAM_SAMPLE* ri[] = {in_place ? expected.data() : in.data()};
    NAM_SAMPLE* ci[] = {in_place ? actual.data() : in.data()};
    NAM_SAMPLE* ro[] = {expected.data()};
    NAM_SAMPLE* co[] = {actual.data()};
    dsp.reference->process(ri, ro, n);
    dsp.candidate->process(ci, co, n);
    for (int i = 0; i < n; ++i)
      compare(expected[i], actual[i], label, position + i);
    position += n;
  }
  ++cases_passed;
}

void regression()
{
  struct Case { int maximum; std::vector<int> schedule; };
  const std::vector<Case> cases = {
    {1, {1}}, {3, {3}}, {7, {7}}, {16, {16}}, {32, {32}}, {64, {64}},
    {128, {128}}, {256, {256}}, {1024, {1024}}, {4096, {4096}},
    {32, {32, 31, 1, 32, 3}}, {65, {31, 32, 33, 63, 64, 65}},
    {64, {1, 3, 7, 31, 64, 2, 63}}, {257, {257, 1, 128, 3, 255, 7}},
    {4096, {1}}, {4096, {64}}, {4096, {127}},
    {4096, {4096, 1, 3, 127, 4095, 64, 257, 2, 1023}}};
  const auto input = make_input(32771);
  for (const auto& model : load_models(true))
  {
    for (const auto& c : cases)
    {
      Pair dsp(model);
      dsp.reset(c.maximum);
      const std::string label = model.label + " max=" + std::to_string(c.maximum)
                                + " first-block=" + std::to_string(c.schedule.front());
      compare_stream(dsp, input, c.schedule, label + " initial", false);
      dsp.reference->prewarm();
      dsp.candidate->prewarm();
      compare_stream(dsp, input, c.schedule, label + " cached/in-place", true);
      dsp.reset(c.maximum == 4096 ? 257 : 4096);
      compare_stream(dsp, input, {1, 17, 64, 257, 3}, label + " resized", false);
    }
    Pair cold(model);
    cold.reference->SetPrewarmOnReset(false);
    cold.candidate->SetPrewarmOnReset(false);
    cold.reset(32);
    compare_stream(cold, input, {1, 32, 257, 7, 4096, 3}, model.label + " cold/growth", false);
    std::cout << "PASS " << model.label << std::endl;
  }
  std::cout << "PASS: " << cases_passed << " streams, " << compared_samples
            << " samples, zero differing bits; checksum=0x" << std::hex << checksum << std::dec << '\n';
}

void process_stream(nam::DSP& dsp, std::vector<NAM_SAMPLE>& input, std::vector<NAM_SAMPLE>& output, int block)
{
  for (size_t pos = 0; pos < input.size(); pos += block)
  {
    const int n = std::min<size_t>(block, input.size() - pos);
    NAM_SAMPLE* in[] = {input.data() + pos};
    NAM_SAMPLE* out[] = {output.data() + pos};
    dsp.process(in, out, n);
  }
}

void latency_regression()
{
  constexpr int frames = 2051;
  int checked = 0;
  int max_reference_delay = 0;
  int max_candidate_delay = 0;
  for (const auto& model : load_models(true))
  {
    for (int block : {1, 32, 64, 257})
    {
      for (int maximum : {block, 4096})
      {
        Pair dsp(model);
        if (dsp.reference->GetPrewarmSamples() != dsp.candidate->GetPrewarmSamples())
          throw std::runtime_error("A2 receptive field / prewarm count changed");
        std::vector<NAM_SAMPLE> input(frames, NAM_SAMPLE(0.0));
        std::vector<NAM_SAMPLE> r_silence(frames), c_silence(frames), r_impulse(frames), c_impulse(frames);
        dsp.reset(maximum);
        dsp.reset(maximum);
        process_stream(*dsp.reference, input, r_silence, block);
        process_stream(*dsp.candidate, input, c_silence, block);
        for (int i = 0; i < frames; ++i)
          compare(r_silence[i], c_silence[i], model.label + " latency/silence", i);
        std::vector<int> positions = {0, block - 1, block, block + 1};
        positions.erase(std::unique(positions.begin(), positions.end()), positions.end());
        for (int impulse : positions)
        {
          std::fill(input.begin(), input.end(), NAM_SAMPLE(0.0));
          input[impulse] = NAM_SAMPLE(1.0);
          dsp.reset(maximum);
          process_stream(*dsp.reference, input, r_impulse, block);
          process_stream(*dsp.candidate, input, c_impulse, block);
          int first_r = -1, first_c = -1;
          for (int i = 0; i < frames; ++i)
          {
            compare(r_impulse[i], c_impulse[i], model.label + " latency/impulse", i);
            if (first_r < 0 && std::bit_cast<Bits>(r_impulse[i]) != std::bit_cast<Bits>(r_silence[i])) first_r = i;
            if (first_c < 0 && std::bit_cast<Bits>(c_impulse[i]) != std::bit_cast<Bits>(c_silence[i])) first_c = i;
          }
          if (first_r < impulse || first_c < impulse || first_c != first_r)
            throw std::runtime_error(model.label + ": impulse response onset changed or is missing");
          max_reference_delay = std::max(max_reference_delay, first_r - impulse);
          max_candidate_delay = std::max(max_candidate_delay, first_c - impulse);
          ++checked;
        }
      }
    }
  }
  std::cout << "PASS latency: " << checked << " impulse cases; maximum response onset delay: reference="
            << max_reference_delay << " samples, candidate=" << max_candidate_delay
            << " samples; added audio latency=0 samples\n";
}

double median(std::vector<double> values)
{
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

void benchmark(bool small_blocks_only = false)
{
  std::vector<NAM_SAMPLE> input(240000);
  uint32_t state = 0xa2b34c56U;
  for (size_t i = 0; i < input.size(); ++i)
  {
    const double t = static_cast<double>(i) / 48000.0;
    input[i] = NAM_SAMPLE(0.25 * std::sin(1382.3 * t) + 0.1 * std::sin(7738.9 * t) + 0.01 * noise(state));
  }
  std::vector<NAM_SAMPLE> output(input.size());
  std::cout << "model,block,max_buffer,reference_ms,candidate_ms,cpu_reduction_pct,paired_min_pct,paired_max_pct\n";
  for (const auto& model : load_models(false))
  {
    for (int block : small_blocks_only ? std::vector<int>{32, 64} : std::vector<int>{32, 64, 128, 256})
    {
      for (int maximum : small_blocks_only ? std::vector<int>{block} : std::vector<int>{block, 4096})
      {
        Pair dsp(model);
        dsp.reset(maximum);
        compare_stream(dsp, input, {block}, model.label + " benchmark", false);
        process_stream(*dsp.reference, input, output, block);
        process_stream(*dsp.candidate, input, output, block);
        std::vector<double> before, after, reductions;
        auto timed = [&](nam::DSP& engine) {
          engine.prewarm();
          const auto start = std::clock();
          process_stream(engine, input, output, block);
          const double ms = 1000.0 * (std::clock() - start) / CLOCKS_PER_SEC;
          if (!(ms > 0)) throw std::runtime_error("CPU timer has insufficient resolution");
          return ms;
        };
        for (int iteration = 0; iteration < 9; ++iteration)
        {
          double r, c;
          if (iteration % 2 == 0) { r = timed(*dsp.reference); c = timed(*dsp.candidate); }
          else { c = timed(*dsp.candidate); r = timed(*dsp.reference); }
          before.push_back(r);
          after.push_back(c);
          reductions.push_back(100.0 * (1.0 - c / r));
        }
        std::cout << model.label << ',' << block << ',' << maximum << ',' << std::fixed << std::setprecision(3)
                  << median(before) << ',' << median(after) << ',' << median(reductions) << ','
                  << *std::min_element(reductions.begin(), reductions.end()) << ','
                  << *std::max_element(reductions.begin(), reductions.end()) << std::endl;
      }
    }
  }
}
} // namespace

int main(int argc, char** argv)
{
  try
  {
    const std::string mode = argc > 1 ? argv[1] : "";
    if (mode == "--negative-control")
    {
      const NAM_SAMPLE expected = NAM_SAMPLE(0.125);
      const auto actual = std::bit_cast<NAM_SAMPLE>(std::bit_cast<Bits>(expected) ^ Bits(1));
      compare(expected, actual, "injected one-bit error", 0);
      return 0;
    }
    if (mode == "--bench") benchmark();
    else if (mode == "--bench-small") benchmark(true);
    else if (mode == "--latency") latency_regression();
    else if (mode.empty()) { regression(); latency_regression(); }
    else throw std::runtime_error("Unknown argument: " + mode);
    return 0;
  }
  catch (const std::exception& e)
  {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
