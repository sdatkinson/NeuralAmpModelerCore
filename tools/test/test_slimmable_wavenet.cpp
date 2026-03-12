#include <cassert>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

#include "json.hpp"

#include "NAM/dsp.h"
#include "NAM/get_dsp.h"
#include "NAM/slimmable.h"

namespace test_slimmable_wavenet
{

// Helper: load a .nam file as JSON
nlohmann::json load_nam_json(const std::string& path)
{
  std::ifstream f(path);
  if (!f.is_open())
    throw std::runtime_error("Cannot open " + path);
  nlohmann::json j;
  f >> j;
  return j;
}

// Helper: process audio and verify finite output
void process_and_verify(nam::DSP* dsp, int num_buffers, int buffer_size)
{
  const double sample_rate = dsp->GetExpectedSampleRate() > 0 ? dsp->GetExpectedSampleRate() : 48000.0;
  dsp->Reset(sample_rate, buffer_size);

  std::vector<NAM_SAMPLE> input(buffer_size);
  std::vector<NAM_SAMPLE> output(buffer_size);
  NAM_SAMPLE* in_ptr = input.data();
  NAM_SAMPLE* out_ptr = output.data();

  for (int buf = 0; buf < num_buffers; buf++)
  {
    for (int i = 0; i < buffer_size; i++)
      input[i] = (NAM_SAMPLE)(0.1 * ((buf * buffer_size + i) % 100) / 100.0);

    dsp->process(&in_ptr, &out_ptr, buffer_size);

    for (int i = 0; i < buffer_size; i++)
      assert(std::isfinite(output[i]));
  }
}

// =====================================================================
// Tests
// =====================================================================

void test_loads_from_file()
{
  std::cout << "  test_slimmable_wavenet_loads_from_file" << std::endl;

  std::filesystem::path path("example_models/slimmable_wavenet.nam");
  auto dsp = nam::get_dsp(path);
  assert(dsp != nullptr);
}

void test_implements_slimmable()
{
  std::cout << "  test_slimmable_wavenet_implements_slimmable" << std::endl;

  auto dsp = nam::get_dsp(std::filesystem::path("example_models/slimmable_wavenet.nam"));
  assert(dsp != nullptr);

  auto* slimmable = dynamic_cast<nam::SlimmableModel*>(dsp.get());
  assert(slimmable != nullptr);
}

void test_processes_audio()
{
  std::cout << "  test_slimmable_wavenet_processes_audio" << std::endl;

  auto dsp = nam::get_dsp(std::filesystem::path("example_models/slimmable_wavenet.nam"));
  assert(dsp != nullptr);
  process_and_verify(dsp.get(), 3, 64);
}

void test_slimming_changes_output()
{
  std::cout << "  test_slimmable_wavenet_slimming_changes_output" << std::endl;

  auto dsp = nam::get_dsp(std::filesystem::path("example_models/slimmable_wavenet.nam"));
  assert(dsp != nullptr);

  const double sample_rate = dsp->GetExpectedSampleRate() > 0 ? dsp->GetExpectedSampleRate() : 48000.0;
  const int buffer_size = 64;
  dsp->Reset(sample_rate, buffer_size);

  std::vector<NAM_SAMPLE> input(buffer_size, 0.1);
  std::vector<NAM_SAMPLE> out_small(buffer_size);
  std::vector<NAM_SAMPLE> out_large(buffer_size);
  NAM_SAMPLE* in_ptr = input.data();
  NAM_SAMPLE* out_ptr;

  auto* slimmable = dynamic_cast<nam::SlimmableModel*>(dsp.get());
  assert(slimmable != nullptr);

  // Process at minimum size (ratio 0.0 -> allowed_channels[0] = 1)
  slimmable->SetSlimmableSize(0.0);
  dsp->Reset(sample_rate, buffer_size);
  out_ptr = out_small.data();
  dsp->process(&in_ptr, &out_ptr, buffer_size);

  // Process at maximum size (ratio 1.0 -> allowed_channels[2] = 3)
  slimmable->SetSlimmableSize(1.0);
  dsp->Reset(sample_rate, buffer_size);
  out_ptr = out_large.data();
  dsp->process(&in_ptr, &out_ptr, buffer_size);

  // Outputs should differ since different channel counts are used
  bool any_different = false;
  for (int i = 0; i < buffer_size; i++)
  {
    if (std::abs(out_small[i] - out_large[i]) > 1e-6)
    {
      any_different = true;
      break;
    }
  }
  assert(any_different);
}

void test_boundary_values()
{
  std::cout << "  test_slimmable_wavenet_boundary_values" << std::endl;

  auto dsp = nam::get_dsp(std::filesystem::path("example_models/slimmable_wavenet.nam"));
  assert(dsp != nullptr);

  const double sample_rate = dsp->GetExpectedSampleRate() > 0 ? dsp->GetExpectedSampleRate() : 48000.0;
  const int buffer_size = 64;
  dsp->Reset(sample_rate, buffer_size);

  std::vector<NAM_SAMPLE> input(buffer_size, 0.05);
  std::vector<NAM_SAMPLE> output(buffer_size);
  NAM_SAMPLE* in_ptr = input.data();
  NAM_SAMPLE* out_ptr = output.data();

  auto* slimmable = dynamic_cast<nam::SlimmableModel*>(dsp.get());
  assert(slimmable != nullptr);

  // Test at various ratio values — with 3 allowed channels [1,2,3]:
  // ratio_to_channels: idx = min(floor(ratio * 3), 2)
  // 0.0 -> idx=0 -> 1ch, 0.33 -> idx=0 -> 1ch, 0.34 -> idx=1 -> 2ch,
  // 0.5 -> idx=1 -> 2ch, 0.66 -> idx=1 -> 2ch, 0.67 -> idx=2 -> 3ch, 1.0 -> idx=2 -> 3ch
  double values[] = {0.0, 0.25, 0.33, 0.34, 0.5, 0.66, 0.67, 0.75, 1.0};
  for (double val : values)
  {
    slimmable->SetSlimmableSize(val);
    dsp->Reset(sample_rate, buffer_size);
    dsp->process(&in_ptr, &out_ptr, buffer_size);
    for (int i = 0; i < buffer_size; i++)
      assert(std::isfinite(output[i]));
  }
}

void test_default_is_max_size()
{
  std::cout << "  test_slimmable_wavenet_default_is_max_size" << std::endl;

  auto dsp = nam::get_dsp(std::filesystem::path("example_models/slimmable_wavenet.nam"));
  assert(dsp != nullptr);

  const double sample_rate = dsp->GetExpectedSampleRate() > 0 ? dsp->GetExpectedSampleRate() : 48000.0;
  const int buffer_size = 64;
  dsp->Reset(sample_rate, buffer_size);

  std::vector<NAM_SAMPLE> input(buffer_size, 0.1);
  std::vector<NAM_SAMPLE> out_default(buffer_size);
  std::vector<NAM_SAMPLE> out_max(buffer_size);
  NAM_SAMPLE* in_ptr = input.data();
  NAM_SAMPLE* out_ptr;

  // Process with default (should be max size = 3 channels)
  out_ptr = out_default.data();
  dsp->process(&in_ptr, &out_ptr, buffer_size);

  // Explicitly set to max
  auto* slimmable = dynamic_cast<nam::SlimmableModel*>(dsp.get());
  assert(slimmable != nullptr);
  slimmable->SetSlimmableSize(1.0);
  dsp->Reset(sample_rate, buffer_size);
  out_ptr = out_max.data();
  dsp->process(&in_ptr, &out_ptr, buffer_size);

  // Both should produce the same output
  for (int i = 0; i < buffer_size; i++)
    assert(std::abs(out_default[i] - out_max[i]) < 1e-6);
}

void test_ratio_mapping()
{
  std::cout << "  test_slimmable_wavenet_ratio_mapping" << std::endl;

  // With allowed_channels [1, 2, 3] (len=3):
  // idx = min(floor(ratio * 3), 2)
  // ratio < 1/3 -> idx=0 -> 1ch
  // 1/3 <= ratio < 2/3 -> idx=1 -> 2ch
  // 2/3 <= ratio -> idx=2 -> 3ch

  auto dsp = nam::get_dsp(std::filesystem::path("example_models/slimmable_wavenet.nam"));
  assert(dsp != nullptr);

  const double sample_rate = dsp->GetExpectedSampleRate() > 0 ? dsp->GetExpectedSampleRate() : 48000.0;
  const int buffer_size = 64;

  std::vector<NAM_SAMPLE> input(buffer_size, 0.1);
  std::vector<NAM_SAMPLE> out_a(buffer_size);
  std::vector<NAM_SAMPLE> out_b(buffer_size);
  NAM_SAMPLE* in_ptr = input.data();
  NAM_SAMPLE* out_ptr;

  auto* slimmable = dynamic_cast<nam::SlimmableModel*>(dsp.get());
  assert(slimmable != nullptr);

  // 0.32 -> floor(0.32*3)=0 -> 1ch
  slimmable->SetSlimmableSize(0.32);
  dsp->Reset(sample_rate, buffer_size);
  out_ptr = out_a.data();
  dsp->process(&in_ptr, &out_ptr, buffer_size);

  // 0.34 -> floor(0.34*3)=1 -> 2ch (different from 1ch)
  slimmable->SetSlimmableSize(0.34);
  dsp->Reset(sample_rate, buffer_size);
  out_ptr = out_b.data();
  dsp->process(&in_ptr, &out_ptr, buffer_size);

  // These should differ (different channel counts: 1 vs 2)
  bool any_different = false;
  for (int i = 0; i < buffer_size; i++)
  {
    if (std::abs(out_a[i] - out_b[i]) > 1e-6)
    {
      any_different = true;
      break;
    }
  }
  assert(any_different);
}

void test_from_json()
{
  std::cout << "  test_slimmable_wavenet_from_json" << std::endl;

  // Build a SlimmableWavenet JSON from an existing WaveNet
  auto wavenet_json = load_nam_json("example_models/wavenet_3ch.nam");

  nlohmann::json j;
  j["version"] = "0.7.0";
  j["architecture"] = "SlimmableWavenet";

  // Copy the WaveNet config and add slimmable field to the first layer
  j["config"]["model"] = wavenet_json["config"];
  j["config"]["model"]["layers"][0]["slimmable"] = {
    {"method", "slice_channels_uniform"},
    {"kwargs", {{"allowed_channels", {2, 3}}}}
  };
  j["weights"] = wavenet_json["weights"];
  j["sample_rate"] = wavenet_json["sample_rate"];

  auto dsp = nam::get_dsp(j);
  assert(dsp != nullptr);
  process_and_verify(dsp.get(), 3, 64);
}

void test_no_slimmable_layers_throws()
{
  std::cout << "  test_slimmable_wavenet_no_slimmable_layers_throws" << std::endl;

  auto wavenet_json = load_nam_json("example_models/wavenet_3ch.nam");

  nlohmann::json j;
  j["version"] = "0.7.0";
  j["architecture"] = "SlimmableWavenet";
  j["config"]["model"] = wavenet_json["config"];
  // No slimmable field on any layer -> all allowed_channels empty -> should throw
  j["weights"] = wavenet_json["weights"];
  j["sample_rate"] = wavenet_json["sample_rate"];

  bool threw = false;
  try
  {
    auto dsp = nam::get_dsp(j);
  }
  catch (const std::runtime_error&)
  {
    threw = true;
  }
  assert(threw);
}

void test_unsupported_method_throws()
{
  std::cout << "  test_slimmable_wavenet_unsupported_method_throws" << std::endl;

  auto wavenet_json = load_nam_json("example_models/wavenet_3ch.nam");

  nlohmann::json j;
  j["version"] = "0.7.0";
  j["architecture"] = "SlimmableWavenet";
  j["config"]["model"] = wavenet_json["config"];
  j["config"]["model"]["layers"][0]["slimmable"] = {
    {"method", "some_future_method"},
    {"kwargs", {{"allowed_channels", {2, 3}}}}
  };
  j["weights"] = wavenet_json["weights"];
  j["sample_rate"] = wavenet_json["sample_rate"];

  bool threw = false;
  try
  {
    auto dsp = nam::get_dsp(j);
  }
  catch (const std::runtime_error&)
  {
    threw = true;
  }
  assert(threw);
}

void test_slimmed_matches_small_model()
{
  std::cout << "  test_slimmable_wavenet_slimmed_matches_small_model" << std::endl;

  // Build a minimal WaveNet config: 1 layer array, 2 layers (dilations [1,2]),
  // kernel_size=3, no gating, no layer1x1, no head1x1, no FiLM, Tanh activation.
  // bottleneck = channels (required when layer1x1 is inactive).
  const int small_ch = 2;
  const int large_ch = 4;
  const int kernel_size = 3;
  const int input_size = 1;
  const int condition_size = 1;
  const int head_size = 1;
  const int num_layers = 2;

  auto make_layer_config = [&](int channels) -> nlohmann::json {
    nlohmann::json lc;
    lc["input_size"] = input_size;
    lc["condition_size"] = condition_size;
    lc["head_size"] = head_size;
    lc["channels"] = channels;
    lc["kernel_size"] = kernel_size;
    lc["dilations"] = {1, 2};
    lc["activation"] = "Tanh";
    lc["head_bias"] = false;
    // Disable layer1x1 so bottleneck == channels (simplest config)
    lc["layer1x1"] = {{"active", false}, {"groups", 1}};
    return lc;
  };

  // --- Generate deterministic weights for the small (2ch) model ---
  // Weight layout for 1 array, no gating, no layer1x1, no head1x1, no FiLM:
  //   rechannel: Conv1x1(input_size -> ch, no bias) = input_size * ch
  //   per layer:
  //     conv: Conv1D(ch -> ch, K, bias) = ch * ch * K + ch
  //     input_mixin: Conv1x1(condition_size -> ch, no bias) = condition_size * ch
  //   head_rechannel: Conv1x1(ch -> head_size, no bias) = ch * head_size
  //   head_scale: 1
  auto count_weights = [&](int ch) {
    int n = input_size * ch; // rechannel
    for (int l = 0; l < num_layers; l++)
    {
      n += ch * ch * kernel_size + ch; // conv
      n += condition_size * ch;        // input_mixin
    }
    n += ch * head_size; // head_rechannel
    n += 1;              // head_scale
    return n;
  };

  const int small_weight_count = count_weights(small_ch);
  std::vector<float> small_weights(small_weight_count);
  // Fill with a deterministic pattern (small non-zero values)
  for (int i = 0; i < small_weight_count; i++)
    small_weights[i] = 0.01f * ((i % 17) - 8); // values in [-0.08, 0.08]

  // --- Embed small weights into large weight vector ---
  // Walk both weight layouts in parallel: for each matrix, place small weights
  // in the top-left corner and fill the rest with arbitrary filler.
  std::vector<float> large_weights;
  auto small_it = small_weights.cbegin();

  // Helper: embed Conv1x1(small_in, small_out) into Conv1x1(full_in, full_out)
  auto embed_conv1x1 = [](std::vector<float>::const_iterator& src, int small_in, int small_out, int full_in,
                           int full_out, bool bias, std::vector<float>& dst) {
    for (int i = 0; i < full_out; i++)
      for (int j = 0; j < full_in; j++)
      {
        if (i < small_out && j < small_in)
          dst.push_back(*(src++));
        else
          dst.push_back(0.02f);
      }
    if (bias)
      for (int i = 0; i < full_out; i++)
      {
        if (i < small_out)
          dst.push_back(*(src++));
        else
          dst.push_back(0.02f);
      }
  };

  // Helper: embed Conv1D(small_in, small_out) into Conv1D(full_in, full_out)
  auto embed_conv1d = [](std::vector<float>::const_iterator& src, int small_in, int small_out, int full_in,
                          int full_out, int ks, std::vector<float>& dst) {
    for (int i = 0; i < full_out; i++)
      for (int j = 0; j < full_in; j++)
        for (int k = 0; k < ks; k++)
        {
          if (i < small_out && j < small_in)
            dst.push_back(*(src++));
          else
            dst.push_back(0.02f);
        }
    // bias
    for (int i = 0; i < full_out; i++)
    {
      if (i < small_out)
        dst.push_back(*(src++));
      else
        dst.push_back(0.02f);
    }
  };

  // rechannel: Conv1x1(input_size -> ch, no bias)
  embed_conv1x1(small_it, input_size, small_ch, input_size, large_ch, false, large_weights);
  // per layer
  for (int l = 0; l < num_layers; l++)
  {
    // conv: Conv1D(ch -> ch, K, bias)
    embed_conv1d(small_it, small_ch, small_ch, large_ch, large_ch, kernel_size, large_weights);
    // input_mixin: Conv1x1(condition_size -> ch, no bias)
    embed_conv1x1(small_it, condition_size, small_ch, condition_size, large_ch, false, large_weights);
  }
  // head_rechannel: Conv1x1(ch -> head_size, no bias)
  embed_conv1x1(small_it, small_ch, head_size, large_ch, head_size, false, large_weights);
  // head_scale
  large_weights.push_back(*(small_it++));

  assert(small_it == small_weights.cend());
  assert((int)large_weights.size() == count_weights(large_ch));

  // --- Build the 2ch WaveNet (non-slimmable) ---
  nlohmann::json small_json;
  small_json["version"] = "0.7.0";
  small_json["architecture"] = "WaveNet";
  small_json["config"]["layers"] = nlohmann::json::array({make_layer_config(small_ch)});
  small_json["config"]["head_scale"] = 1.0;
  small_json["weights"] = small_weights;
  small_json["sample_rate"] = 48000;

  auto small_dsp = nam::get_dsp(small_json);
  assert(small_dsp != nullptr);

  // --- Build the 4ch SlimmableWavenet ---
  nlohmann::json large_json;
  large_json["version"] = "0.7.0";
  large_json["architecture"] = "SlimmableWavenet";
  auto large_layer_config = make_layer_config(large_ch);
  large_layer_config["slimmable"] = {{"method", "slice_channels_uniform"},
                                     {"kwargs", {{"allowed_channels", {small_ch, large_ch}}}}};
  large_json["config"]["model"]["layers"] = nlohmann::json::array({large_layer_config});
  large_json["config"]["model"]["head_scale"] = 1.0;
  large_json["weights"] = large_weights;
  large_json["sample_rate"] = 48000;

  auto large_dsp = nam::get_dsp(large_json);
  assert(large_dsp != nullptr);

  // Slim the large model down to match the small model
  auto* slimmable = dynamic_cast<nam::SlimmableModel*>(large_dsp.get());
  assert(slimmable != nullptr);
  // ratio 0.0 -> idx = floor(0.0 * 2) = 0 -> allowed_channels[0] = small_ch
  slimmable->SetSlimmableSize(0.0);

  // --- Process audio through both and compare ---
  const double sample_rate = 48000.0;
  const int buffer_size = 64;
  const int num_buffers = 5; // process enough buffers to exercise the dilated convolutions

  small_dsp->Reset(sample_rate, buffer_size);
  large_dsp->Reset(sample_rate, buffer_size);

  for (int buf = 0; buf < num_buffers; buf++)
  {
    std::vector<NAM_SAMPLE> input(buffer_size);
    for (int i = 0; i < buffer_size; i++)
      input[i] = (NAM_SAMPLE)(0.1 * std::sin(0.1 * (buf * buffer_size + i)));

    std::vector<NAM_SAMPLE> out_small(buffer_size);
    std::vector<NAM_SAMPLE> out_large(buffer_size);
    NAM_SAMPLE* in_ptr = input.data();
    NAM_SAMPLE* out_ptr;

    out_ptr = out_small.data();
    small_dsp->process(&in_ptr, &out_ptr, buffer_size);

    out_ptr = out_large.data();
    large_dsp->process(&in_ptr, &out_ptr, buffer_size);

    for (int i = 0; i < buffer_size; i++)
    {
      assert(std::isfinite(out_small[i]));
      assert(std::isfinite(out_large[i]));
      if (std::abs(out_small[i] - out_large[i]) > 1e-6)
      {
        std::cerr << "  MISMATCH at buffer " << buf << " sample " << i << ": small=" << out_small[i]
                  << " slimmed=" << out_large[i] << " diff=" << std::abs(out_small[i] - out_large[i]) << std::endl;
        assert(false);
      }
    }
  }
}

} // namespace test_slimmable_wavenet
