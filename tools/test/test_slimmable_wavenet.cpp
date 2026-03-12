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

} // namespace test_slimmable_wavenet
