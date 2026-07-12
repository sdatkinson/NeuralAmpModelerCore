// Numerical verification for the fused NEON WaveNet engine:
// for configs that match the fused shape, the fused engine must produce the
// same output (within FMA-reordering tolerance) as the generic WaveNet on the
// same input and weights.
//
// Built only when NAM_ENABLE_FUSED is defined at compile time. On builds
// without NEON kernels (non-AArch64), the numerical tests are skipped and the
// detector is verified to decline everything.

#if defined(NAM_ENABLE_FUSED)

  #include <cassert>
  #include <cmath>
  #include <cstdint>
  #include <iostream>
  #include <memory>
  #include <random>
  #include <string>
  #include <vector>

  #include "json.hpp"

  #include "NAM/dsp.h"
  #include "NAM/wavenet/fused.h"
  #include "NAM/wavenet/model.h"

  #include "allocation_tracking.h"

namespace test_fused
{
namespace
{

struct ArrayShape
{
  int input_size;
  int channels;
  int head_size;
  int head_kernel_size;
  bool head_bias;
  bool layer1x1_active;
  std::vector<int> kernel_sizes;
  std::vector<int> dilations;
  nlohmann::json activation; // single config applied to all layers
};

nlohmann::json build_config(const std::vector<ArrayShape>& arrays)
{
  using nlohmann::json;
  json layers = json::array();
  for (const auto& a : arrays)
  {
    json la;
    la["input_size"] = a.input_size;
    la["condition_size"] = 1;
    la["channels"] = a.channels;
    la["bottleneck"] = a.channels;
    la["kernel_sizes"] = a.kernel_sizes;
    la["dilations"] = a.dilations;
    la["activation"] = a.activation;
    la["gating_mode"] = "none";
    la["head"] = {{"out_channels", a.head_size}, {"kernel_size", a.head_kernel_size}, {"bias", a.head_bias}};
    la["layer1x1"] = {{"active", a.layer1x1_active}, {"groups", 1}};
    la["head1x1"] = {{"active", false}, {"out_channels", 1}, {"groups", 1}};
    layers.push_back(la);
  }
  json config;
  config["layers"] = layers;
  config["head_scale"] = 0.02f;
  return config;
}

/// The classic "A1 standard" shape (what most captures in the wild use),
/// expressed with the legacy schema fields (kernel_size + head_size/head_bias)
/// to make sure the detector handles old .nam files.
nlohmann::json build_a1_legacy_config(int ch0, int ch1)
{
  using nlohmann::json;
  json config;
  json la0, la1;
  const json dil = json::array({1, 2, 4, 8, 16, 32, 64, 128, 256, 512});
  la0["input_size"] = 1;
  la0["condition_size"] = 1;
  la0["channels"] = ch0;
  la0["kernel_size"] = 3;
  la0["dilations"] = dil;
  la0["activation"] = "Tanh";
  la0["gated"] = false;
  la0["head_size"] = ch1;
  la0["head_bias"] = false;
  la1["input_size"] = ch0;
  la1["condition_size"] = 1;
  la1["channels"] = ch1;
  la1["kernel_size"] = 3;
  la1["dilations"] = dil;
  la1["activation"] = "Tanh";
  la1["gated"] = false;
  la1["head_size"] = 1;
  la1["head_bias"] = true;
  config["layers"] = json::array({la0, la1});
  config["head_scale"] = 0.02f;
  return config;
}

int count_weights(const nlohmann::json& config)
{
  int total = 0;
  int prev_channels = 1;
  for (const auto& la : config["layers"])
  {
    const int C = la["channels"].get<int>();
    const int in_size = la["input_size"].get<int>();
    (void)prev_channels;
    total += C * in_size; // rechannel
    const auto& dil = la["dilations"];
    for (size_t i = 0; i < dil.size(); i++)
    {
      int K;
      if (la.find("kernel_sizes") != la.end())
        K = la["kernel_sizes"][i].get<int>();
      else
        K = la["kernel_size"].get<int>();
      total += C * C * K + C; // conv + bias
      total += C; // mixin
      bool l1x1 = true;
      if (la.find("layer1x1") != la.end())
        l1x1 = la["layer1x1"]["active"].get<bool>();
      if (l1x1)
        total += C * C + C;
    }
    int H, Kh;
    bool hb;
    if (la.find("head") != la.end())
    {
      H = la["head"]["out_channels"].get<int>();
      Kh = la["head"]["kernel_size"].get<int>();
      hb = la["head"]["bias"].get<bool>();
    }
    else
    {
      H = la["head_size"].get<int>();
      Kh = 1;
      hb = la["head_bias"].get<bool>();
    }
    total += H * C * Kh + (hb ? H : 0);
    prev_channels = C;
  }
  total += 1; // trailing head_scale
  return total;
}

std::vector<float> make_weights(int count, uint32_t seed)
{
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-0.3f, 0.3f);
  std::vector<float> w(count);
  for (auto& x : w)
    x = dist(rng);
  return w;
}

std::vector<NAM_SAMPLE> make_input(int num_frames)
{
  std::vector<NAM_SAMPLE> in(num_frames);
  for (int i = 0; i < num_frames; i++)
  {
    const double t = static_cast<double>(i) / 48000.0;
    in[i] = static_cast<NAM_SAMPLE>(0.4 * std::sin(2.0 * M_PI * 110.0 * t) + 0.15 * std::sin(2.0 * M_PI * 997.0 * t));
  }
  return in;
}

std::vector<NAM_SAMPLE> run_dsp(nam::DSP& dsp, const std::vector<NAM_SAMPLE>& input, int block_size)
{
  dsp.Reset(48000.0, block_size);
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

void compare(const std::vector<NAM_SAMPLE>& generic, const std::vector<NAM_SAMPLE>& fast, const std::string& what,
             int block_size, double tol)
{
  assert(generic.size() == fast.size());
  double max_diff = 0.0;
  size_t max_i = 0;
  for (size_t i = 0; i < generic.size(); i++)
  {
    const double d = std::fabs(static_cast<double>(generic[i]) - static_cast<double>(fast[i]));
    if (d > max_diff)
    {
      max_diff = d;
      max_i = i;
    }
  }
  if (!(max_diff < tol))
  {
    std::cerr << "FusedWaveNet diverges from generic WaveNet for " << what << " (block=" << block_size
              << "): max |diff| = " << max_diff << " at i=" << max_i << " (generic=" << generic[max_i]
              << ", fused=" << fast[max_i] << ")" << std::endl;
    assert(false);
  }
}

void check_matches_generic(const nlohmann::json& config, const std::string& what, uint32_t seed,
                           double tol = 5e-5)
{
  assert(nam::wavenet::fused::is_fused_shape(config));

  const int weight_count = count_weights(config);
  const auto weights = make_weights(weight_count, seed);

  auto fused_cfg = nam::wavenet::fused::create_fused_config(config, 48000.0);
  std::vector<float> w_fast = weights;
  auto fused_dsp = fused_cfg->create(std::move(w_fast), 48000.0);

  // Reference: parse_config_json directly, bypassing the dispatcher shortcut.
  auto generic_cfg = nam::wavenet::parse_config_json(config, 48000.0);
  std::vector<float> w_gen = weights;
  auto generic_dsp = generic_cfg.create(std::move(w_gen), 48000.0);

  const int total = 4096;
  const auto input = make_input(total);
  // Include a non-multiple-of-4 block size to exercise the remainder tiles.
  for (int block : {64, 256, 23})
  {
    const auto out_generic = run_dsp(*generic_dsp, input, block);
    const auto out_fused = run_dsp(*fused_dsp, input, block);
    compare(out_generic, out_fused, what, block, tol);
  }
}

} // namespace

void test_detector_declines_non_fused_shapes()
{
  // Gated layers must be declined.
  {
    auto cfg = build_config({{1, 8, 1, 1, true, true, std::vector<int>(4, 3), {1, 2, 4, 8}, "Tanh"}});
    cfg["layers"][0]["gating_mode"] = "gated";
    assert(!nam::wavenet::fused::is_fused_shape(cfg));
  }
  // Channels not a multiple of 4.
  {
    auto cfg = build_config({{1, 6, 1, 1, true, true, std::vector<int>(4, 3), {1, 2, 4, 8}, "Tanh"}});
    assert(!nam::wavenet::fused::is_fused_shape(cfg));
  }
  // FiLM active.
  {
    auto cfg = build_config({{1, 8, 1, 1, true, true, std::vector<int>(4, 3), {1, 2, 4, 8}, "Tanh"}});
    cfg["layers"][0]["conv_post_film"] = {{"active", true}, {"shift", true}, {"groups", 1}};
    assert(!nam::wavenet::fused::is_fused_shape(cfg));
  }
  // head1x1 active.
  {
    auto cfg = build_config({{1, 8, 1, 1, true, true, std::vector<int>(4, 3), {1, 2, 4, 8}, "Tanh"}});
    cfg["layers"][0]["head1x1"] = {{"active", true}, {"out_channels", 4}, {"groups", 1}};
    assert(!nam::wavenet::fused::is_fused_shape(cfg));
  }
  // Bottleneck != channels.
  {
    auto cfg = build_config({{1, 8, 1, 1, true, true, std::vector<int>(4, 3), {1, 2, 4, 8}, "Tanh"}});
    cfg["layers"][0]["bottleneck"] = 4;
    assert(!nam::wavenet::fused::is_fused_shape(cfg));
  }
  // Grouped layer1x1.
  {
    auto cfg = build_config({{1, 8, 1, 1, true, true, std::vector<int>(4, 3), {1, 2, 4, 8}, "Tanh"}});
    cfg["layers"][0]["layer1x1"] = {{"active", true}, {"groups", 2}};
    assert(!nam::wavenet::fused::is_fused_shape(cfg));
  }
}

void test_detector_accepts_a1_legacy()
{
  if (!nam::wavenet::fused::available())
    return;
  assert(nam::wavenet::fused::is_fused_shape(build_a1_legacy_config(16, 8)));
}

void test_a1_standard_matches_generic()
{
  if (!nam::wavenet::fused::available())
    return;
  nam::activations::Activation::disable_fast_tanh();
  check_matches_generic(build_a1_legacy_config(16, 8), "A1 standard (16/8, exact tanh)", 0xF00D0001u);
}

void test_a1_standard_fast_tanh_matches_generic()
{
  if (!nam::wavenet::fused::available())
    return;
  nam::activations::Activation::enable_fast_tanh();
  check_matches_generic(build_a1_legacy_config(16, 8), "A1 standard (16/8, fast tanh)", 0xF00D0002u);
  nam::activations::Activation::disable_fast_tanh();
}

void test_a1_lite_feather_match_generic()
{
  if (!nam::wavenet::fused::available())
    return;
  nam::activations::Activation::enable_fast_tanh();
  check_matches_generic(build_a1_legacy_config(8, 4), "A1 lite-like (8/4)", 0xF00D0003u);
  nam::activations::Activation::disable_fast_tanh();
}

void test_activations_match_generic()
{
  if (!nam::wavenet::fused::available())
    return;
  const std::vector<int> dil = {1, 2, 4, 8, 16};
  const std::vector<int> ks(5, 3);
  uint32_t seed = 0xF00D0100u;
  for (const nlohmann::json act : {nlohmann::json("ReLU"), nlohmann::json("Hardtanh"), nlohmann::json("Softsign"),
                                   nlohmann::json("Sigmoid"), nlohmann::json({{"type", "LeakyReLU"},
                                                                              {"negative_slope", 0.02}})})
  {
    auto cfg = build_config({{1, 8, 1, 1, true, true, ks, dil, act}});
    check_matches_generic(cfg, "activation " + act.dump(), seed++);
  }
}

void test_mixed_kernel_sizes_and_head_kernel_match_generic()
{
  if (!nam::wavenet::fused::available())
    return;
  // A2-standard-like: kernel sizes 6 and 15, head kernel 16, LeakyReLU.
  std::vector<int> ks = {6, 6, 6, 6, 15, 6, 6};
  std::vector<int> dil = {1, 3, 7, 17, 1, 13, 41};
  nlohmann::json act = {{"type", "LeakyReLU"}, {"negative_slope", 0.01}};
  auto cfg = build_config({{1, 8, 1, 16, true, true, ks, dil, act}});
  check_matches_generic(cfg, "A2-like (k=6/15, head k=16)", 0xF00D0200u);
}

void test_layer1x1_inactive_matches_generic()
{
  if (!nam::wavenet::fused::available())
    return;
  auto cfg = build_config({{1, 8, 4, 1, false, false, {3, 3, 3}, {1, 2, 4}, "Tanh"},
                           {8, 4, 1, 1, true, true, {3, 3, 3}, {1, 2, 4}, "Tanh"}});
  check_matches_generic(cfg, "layer1x1 inactive in first array", 0xF00D0300u);
}

void test_channels_12_20_match_generic()
{
  if (!nam::wavenet::fused::available())
    return;
  nam::activations::Activation::enable_fast_tanh();
  auto cfg12 = build_config({{1, 12, 1, 1, true, true, {3, 3, 3, 3}, {1, 2, 4, 8}, "Tanh"}});
  check_matches_generic(cfg12, "channels=12", 0xF00D0400u);
  auto cfg20 = build_config({{1, 20, 1, 1, true, true, {3, 3, 3, 3}, {1, 2, 4, 8}, "Tanh"}});
  check_matches_generic(cfg20, "channels=20", 0xF00D0401u);
  auto cfg32 = build_config({{1, 32, 1, 1, true, true, {3, 3}, {1, 2}, "Tanh"}});
  check_matches_generic(cfg32, "channels=32", 0xF00D0402u);
  nam::activations::Activation::disable_fast_tanh();
}

void test_process_is_realtime_safe()
{
  if (!nam::wavenet::fused::available())
    return;
  auto cfg = build_a1_legacy_config(16, 8);
  const auto weights = make_weights(count_weights(cfg), 0xF00D0500u);
  auto fused_cfg = nam::wavenet::fused::create_fused_config(cfg, 48000.0);
  std::vector<float> w = weights;
  auto dsp = fused_cfg->create(std::move(w), 48000.0);
  const int block = 64;
  dsp->Reset(48000.0, block);

  const auto input = make_input(block);
  std::vector<NAM_SAMPLE> out(block, 0.0);
  const NAM_SAMPLE* in_arr[] = {input.data()};
  NAM_SAMPLE* out_arr[] = {out.data()};

  // One call to settle anything lazy, then track allocations across many
  // process() calls: expected zero allocations and zero deallocations.
  dsp->process(const_cast<NAM_SAMPLE**>(in_arr), out_arr, block);
  allocation_tracking::run_allocation_test(
    nullptr,
    [&]() {
      for (int i = 0; i < 32; i++)
        dsp->process(const_cast<NAM_SAMPLE**>(in_arr), out_arr, block);
    },
    nullptr, /*expected_allocations=*/0, /*expected_deallocations=*/0, "FusedWaveNet::process realtime safety");
}

} // namespace test_fused

#endif // NAM_ENABLE_FUSED
