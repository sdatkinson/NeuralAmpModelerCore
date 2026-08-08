// Bit-identity verification for the planar NEON A2 kernels.
//
// The claim these kernels make is stronger than "close enough": for the A2 nano
// and A2 standard shapes they produce the *same float32 bits* as the reference
// A2 fast path, sample for sample. This asserts exactly that -- memcmp, not a
// tolerance -- across a spread of block sizes, including ones that exercise the
// partial-tile and single-frame tails.
//
// Built only where the planar kernels exist (AArch64 with the A2 fast path on).
// Everywhere else the test bodies compile to nothing.

#if defined(NAM_ENABLE_A2_FAST)

  #include <algorithm>
  #include <cassert>
  #include <cmath>
  #include <cstdint>
  #include <cstring>
  #include <iostream>
  #include <memory>
  #include <random>
  #include <string>
  #include <vector>

  #include "json.hpp"

  #include "NAM/dsp.h"
  #include "NAM/wavenet/a2_fast.h"
  #include "NAM/wavenet/a2_planar.h"

namespace test_a2_planar
{

  #if defined(NAM_A2_PLANAR)

namespace
{

nlohmann::json build_a2_config(int channels)
{
  using nlohmann::json;

  json activation = json::array();
  json gating_mode = json::array();
  json secondary = json::array();
  json kernel_sizes = json::array();
  json dilations = json::array();
  for (int i = 0; i < nam::wavenet::a2_fast::kNumLayers; i++)
  {
    activation.push_back({{"type", "LeakyReLU"}, {"negative_slope", nam::wavenet::a2_fast::kLeakySlope}});
    gating_mode.push_back("none");
    secondary.push_back(nullptr);
    kernel_sizes.push_back(nam::wavenet::a2_fast::kKernelSizes[i]);
    dilations.push_back(nam::wavenet::a2_fast::kDilations[i]);
  }

  json film_inactive = {{"active", false}, {"shift", true}, {"groups", 1}};

  json layer;
  layer["input_size"] = 1;
  layer["condition_size"] = 1;
  layer["channels"] = channels;
  layer["bottleneck"] = channels;
  layer["kernel_sizes"] = kernel_sizes;
  layer["dilations"] = dilations;
  layer["activation"] = activation;
  layer["gating_mode"] = gating_mode;
  layer["secondary_activation"] = secondary;
  layer["head"] = {{"out_channels", 1}, {"kernel_size", nam::wavenet::a2_fast::kHeadKernelSize}, {"bias", true}};
  layer["head1x1"] = {{"active", false}, {"out_channels", 1}, {"groups", 1}};
  layer["layer1x1"] = {{"active", true}, {"groups", 1}};
  layer["conv_pre_film"] = film_inactive;
  layer["conv_post_film"] = film_inactive;
  layer["input_mixin_pre_film"] = film_inactive;
  layer["input_mixin_post_film"] = film_inactive;
  layer["activation_pre_film"] = film_inactive;
  layer["activation_post_film"] = film_inactive;
  layer["layer1x1_post_film"] = film_inactive;
  layer["head1x1_post_film"] = film_inactive;
  layer["groups_input"] = 1;
  layer["groups_input_mixin"] = 1;

  json config;
  config["layers"] = json::array({layer});
  config["head_scale"] = 0.01f;
  return config;
}

int a2_weight_count(int channels)
{
  const int bn = channels;
  int total = /*rechannel*/ channels;
  for (int i = 0; i < nam::wavenet::a2_fast::kNumLayers; i++)
  {
    const int K = nam::wavenet::a2_fast::kKernelSizes[i];
    total += bn * channels * K + bn; // conv1d weights + bias
    total += bn; // input mixin (no bias)
    total += channels * bn + channels; // layer1x1 + bias
  }
  total += channels * nam::wavenet::a2_fast::kHeadKernelSize + 1; // head rechannel + bias
  total += 1; // trailing head_scale
  return total;
}

std::vector<float> make_deterministic_weights(int count, uint32_t seed)
{
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-0.3f, 0.3f);
  std::vector<float> w(count);
  for (auto& x : w)
    x = dist(rng);
  return w;
}

std::vector<NAM_SAMPLE> make_test_input(int num_frames, double sample_rate)
{
  std::vector<NAM_SAMPLE> in(num_frames);
  for (int i = 0; i < num_frames; i++)
  {
    const double t = static_cast<double>(i) / sample_rate;
    in[i] = static_cast<NAM_SAMPLE>(0.25 * std::sin(2.0 * M_PI * 220.0 * t) + 0.10 * std::sin(2.0 * M_PI * 1230.0 * t));
  }
  return in;
}

std::vector<NAM_SAMPLE> run_dsp(nam::DSP& dsp, const std::vector<NAM_SAMPLE>& input, int block_size)
{
  dsp.Reset(48000.0, block_size); // also prewarms
  std::vector<NAM_SAMPLE> out(input.size(), static_cast<NAM_SAMPLE>(0));
  int pos = 0;
  const int total = static_cast<int>(input.size());
  while (pos < total)
  {
    const int n = std::min(block_size, total - pos);
    const NAM_SAMPLE* in_ptr = input.data() + pos;
    NAM_SAMPLE* out_ptr = out.data() + pos;
    const NAM_SAMPLE* in_arr[] = {in_ptr};
    NAM_SAMPLE* out_arr[] = {out_ptr};
    dsp.process(const_cast<NAM_SAMPLE**>(in_arr), out_arr, n);
    pos += n;
  }
  return out;
}

/// The whole point: identical bits, not a tolerance.
void assert_bit_identical(const std::vector<NAM_SAMPLE>& reference, const std::vector<NAM_SAMPLE>& planar, int channels,
                          int block_size)
{
  assert(reference.size() == planar.size());
  if (std::memcmp(reference.data(), planar.data(), reference.size() * sizeof(NAM_SAMPLE)) == 0)
    return;

  size_t first = 0;
  while (first < reference.size() && reference[first] == planar[first])
    first++;
  std::cerr << "A2 planar kernel (channels=" << channels << ", block=" << block_size
            << ") is not bit-identical to the reference: first difference at sample " << first
            << " (reference=" << reference[first] << ", planar=" << planar[first] << ")" << std::endl;
  assert(false);
}

void check_channels(int channels)
{
  const auto config = build_a2_config(channels);
  int detected = 0;
  assert(nam::wavenet::a2_fast::is_a2_shape(config, &detected));
  assert(detected == channels);

  const auto weights = make_deterministic_weights(a2_weight_count(channels), 0xA2u + channels);
  // Long enough that every layer's ring wraps and rewinds several times.
  const auto input = make_test_input(20000, 48000.0);

  // Block sizes chosen to hit each path: below one vector, not a multiple of
  // four, exactly one vector, between one vector and one tile, the tile widths
  // themselves (32 for nano, 8 for standard), and well past them.
  for (const int block_size : {1, 3, 4, 7, 8, 15, 16, 31, 32, 33, 64, 65, 128, 512})
  {
    auto reference = nam::wavenet::a2_fast::create_a2_fast_reference_model(channels, weights, 48000.0);
    auto planar = nam::wavenet::a2_fast::create_a2_planar_model(channels, weights, 48000.0);
    assert(planar != nullptr);
    assert(reference->GetPrewarmSamples() == planar->GetPrewarmSamples());

    const auto out_reference = run_dsp(*reference, input, block_size);
    const auto out_planar = run_dsp(*planar, input, block_size);
    assert_bit_identical(out_reference, out_planar, channels, block_size);
  }
}

} // namespace

void test_bit_identical_nano()
{
  check_channels(3);
}

void test_bit_identical_standard()
{
  check_channels(8);
}

/// The dispatcher must actually route to the planar kernel where it exists,
/// otherwise the tests above would be checking something nothing uses.
void test_factory_selects_planar()
{
  for (const int channels : {3, 8})
  {
    const auto config = build_a2_config(channels);
    const auto weights = make_deterministic_weights(a2_weight_count(channels), 0x5Eu + channels);
    auto model_config = nam::wavenet::a2_fast::create_a2_fast_config(config, 48000.0);
    auto from_factory = model_config->create(weights, 48000.0);
    auto planar = nam::wavenet::a2_fast::create_a2_planar_model(channels, weights, 48000.0);

    const auto input = make_test_input(4096, 48000.0);
    const auto out_factory = run_dsp(*from_factory, input, 64);
    const auto out_planar = run_dsp(*planar, input, 64);
    assert(std::memcmp(out_factory.data(), out_planar.data(), out_factory.size() * sizeof(NAM_SAMPLE)) == 0);
  }
}

  #else // NAM_A2_PLANAR

void test_bit_identical_nano() {}
void test_bit_identical_standard() {}
void test_factory_selects_planar() {}

  #endif // NAM_A2_PLANAR

} // namespace test_a2_planar

#endif // NAM_ENABLE_A2_FAST
