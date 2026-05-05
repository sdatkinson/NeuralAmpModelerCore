#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "json.hpp"

#include "NAM/container.h"
#include "NAM/get_dsp.h"
#include "NAM/slimmable.h"

namespace test_container
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

// Build a SlimmableContainer JSON from 3 .nam files
nlohmann::json build_container_json(const std::string& small_path, const std::string& medium_path,
                                    const std::string& large_path)
{
  nlohmann::json small_model = load_nam_json(small_path);
  nlohmann::json medium_model = load_nam_json(medium_path);
  nlohmann::json large_model = load_nam_json(large_path);

  nlohmann::json container;
  container["version"] = "0.7.0";
  container["architecture"] = "SlimmableContainer";
  container["config"]["submodels"] = nlohmann::json::array({{{"max_value", 0.33}, {"model", small_model}},
                                                            {{"max_value", 0.66}, {"model", medium_model}},
                                                            {{"max_value", 1.0}, {"model", large_model}}});
  container["weights"] = nlohmann::json::array();
  container["sample_rate"] = 48000;
  return container;
}

// Helper to process audio through a DSP model and verify finite output
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

void test_container_loads_from_json()
{
  auto j =
    build_container_json("example_models/lstm.nam", "example_models/wavenet.nam", "example_models/wavenet_a2_max.nam");
  auto dsp = nam::get_dsp(j);
  assert(dsp != nullptr);
}

void test_container_processes_audio()
{
  auto j =
    build_container_json("example_models/lstm.nam", "example_models/wavenet.nam", "example_models/wavenet_a2_max.nam");
  auto dsp = nam::get_dsp(j);
  process_and_verify(dsp.get(), 3, 64);
}

void test_container_slimmable_selects_submodel()
{
  auto j =
    build_container_json("example_models/lstm.nam", "example_models/wavenet.nam", "example_models/wavenet_a2_max.nam");
  auto dsp = nam::get_dsp(j);
  const double sample_rate = 48000.0;
  const int buffer_size = 64;
  dsp->Reset(sample_rate, buffer_size);

  std::vector<NAM_SAMPLE> input(buffer_size, 0.1);
  std::vector<NAM_SAMPLE> out_small(buffer_size);
  std::vector<NAM_SAMPLE> out_large(buffer_size);
  NAM_SAMPLE* in_ptr = input.data();
  NAM_SAMPLE* out_ptr;

  auto* slimmable = dynamic_cast<nam::SlimmableModel*>(dsp.get());
  assert(slimmable != nullptr);

  // Process at minimum size (selects first submodel)
  slimmable->SetSlimmableSize(0.0);
  out_ptr = out_small.data();
  dsp->process(&in_ptr, &out_ptr, buffer_size);

  // Process at maximum size (selects last submodel)
  slimmable->SetSlimmableSize(1.0);
  out_ptr = out_large.data();
  dsp->process(&in_ptr, &out_ptr, buffer_size);

  // The outputs should differ since different models are active
  // (Not guaranteed for every sample, but statistically they should differ)
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

void test_container_boundary_values()
{
  auto j =
    build_container_json("example_models/lstm.nam", "example_models/wavenet.nam", "example_models/wavenet_a2_max.nam");
  auto dsp = nam::get_dsp(j);
  const double sample_rate = 48000.0;
  const int buffer_size = 16;
  dsp->Reset(sample_rate, buffer_size);

  std::vector<NAM_SAMPLE> input(buffer_size, 0.05);
  std::vector<NAM_SAMPLE> output(buffer_size);
  NAM_SAMPLE* in_ptr = input.data();
  NAM_SAMPLE* out_ptr = output.data();

  auto* slimmable = dynamic_cast<nam::SlimmableModel*>(dsp.get());
  assert(slimmable != nullptr);

  // Test exact boundary values (max_value is exclusive: val < max_value selects that submodel)
  slimmable->SetSlimmableSize(0.32); // Should select first submodel (0.32 < 0.33)
  dsp->process(&in_ptr, &out_ptr, buffer_size);
  for (int i = 0; i < buffer_size; i++)
    assert(std::isfinite(output[i]));

  slimmable->SetSlimmableSize(0.33); // Should select second submodel (0.33 is NOT < 0.33, but IS < 0.66)
  dsp->process(&in_ptr, &out_ptr, buffer_size);
  for (int i = 0; i < buffer_size; i++)
    assert(std::isfinite(output[i]));

  slimmable->SetSlimmableSize(0.65); // Should select second submodel (0.65 < 0.66)
  dsp->process(&in_ptr, &out_ptr, buffer_size);
  for (int i = 0; i < buffer_size; i++)
    assert(std::isfinite(output[i]));

  slimmable->SetSlimmableSize(0.66); // Should select third/last submodel (0.66 is NOT < 0.66, fallback)
  dsp->process(&in_ptr, &out_ptr, buffer_size);
  for (int i = 0; i < buffer_size; i++)
    assert(std::isfinite(output[i]));

  slimmable->SetSlimmableSize(1.0); // Should select third/last submodel (fallback)
  dsp->process(&in_ptr, &out_ptr, buffer_size);
  for (int i = 0; i < buffer_size; i++)
    assert(std::isfinite(output[i]));
}

void test_container_empty_submodels_throws()
{
  nlohmann::json j;
  j["version"] = "0.7.0";
  j["architecture"] = "SlimmableContainer";
  j["config"]["submodels"] = nlohmann::json::array();
  j["weights"] = nlohmann::json::array();
  j["sample_rate"] = 48000;

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

void test_container_last_max_value_must_cover_one()
{
  // Build a container where last max_value < 1.0
  auto small_json = load_nam_json("example_models/lstm.nam");

  nlohmann::json j;
  j["version"] = "0.7.0";
  j["architecture"] = "SlimmableContainer";
  j["config"]["submodels"] = nlohmann::json::array({{{"max_value", 0.5}, {"model", small_json}}});
  j["weights"] = nlohmann::json::array();
  j["sample_rate"] = 48000;

  // Suppress the version warning
  std::streambuf* originalCerr = std::cerr.rdbuf();
  std::ostringstream nullStream;
  std::cerr.rdbuf(nullStream.rdbuf());

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

  std::cerr.rdbuf(originalCerr);
}

void test_container_unsorted_submodels_throws()
{
  auto small_json = load_nam_json("example_models/lstm.nam");
  auto medium_json = load_nam_json("example_models/wavenet.nam");

  nlohmann::json j;
  j["version"] = "0.7.0";
  j["architecture"] = "SlimmableContainer";
  j["config"]["submodels"] =
    nlohmann::json::array({{{"max_value", 0.8}, {"model", small_json}}, {{"max_value", 0.5}, {"model", medium_json}}});
  j["weights"] = nlohmann::json::array();
  j["sample_rate"] = 48000;

  std::streambuf* originalCerr = std::cerr.rdbuf();
  std::ostringstream nullStream;
  std::cerr.rdbuf(nullStream.rdbuf());

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

  std::cerr.rdbuf(originalCerr);
}

void test_container_sample_rate_mismatch_throws()
{
  // Create two models with different sample rates
  auto model_48k = load_nam_json("example_models/lstm.nam");
  auto model_44k = load_nam_json("example_models/lstm.nam");
  model_44k["sample_rate"] = 44100;

  nlohmann::json j;
  j["version"] = "0.7.0";
  j["architecture"] = "SlimmableContainer";
  j["config"]["submodels"] =
    nlohmann::json::array({{{"max_value", 0.5}, {"model", model_44k}}, {{"max_value", 1.0}, {"model", model_48k}}});
  j["weights"] = nlohmann::json::array();
  j["sample_rate"] = 48000;

  std::streambuf* originalCerr = std::cerr.rdbuf();
  std::ostringstream nullStream;
  std::cerr.rdbuf(nullStream.rdbuf());

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

  std::cerr.rdbuf(originalCerr);
}

void test_container_load_from_file()
{
  std::filesystem::path path("example_models/slimmable_container.nam");
  auto dsp = nam::get_dsp(path);
  assert(dsp != nullptr);
  process_and_verify(dsp.get(), 3, 64);
}

void test_container_default_is_max_size()
{
  auto j =
    build_container_json("example_models/lstm.nam", "example_models/wavenet.nam", "example_models/wavenet_a2_max.nam");
  auto dsp = nam::get_dsp(j);
  const double sample_rate = 48000.0;
  const int buffer_size = 64;
  dsp->Reset(sample_rate, buffer_size);

  std::vector<NAM_SAMPLE> input(buffer_size, 0.1);
  std::vector<NAM_SAMPLE> out_default(buffer_size);
  std::vector<NAM_SAMPLE> out_max(buffer_size);
  NAM_SAMPLE* in_ptr = input.data();
  NAM_SAMPLE* out_ptr;

  // Ensure both predictions start from identical model state.
  dsp->ResetAndPrewarm(sample_rate, buffer_size);
  // Process with default (should be max size)
  out_ptr = out_default.data();
  dsp->process(&in_ptr, &out_ptr, buffer_size);

  // Explicitly set to max
  auto* slimmable = dynamic_cast<nam::SlimmableModel*>(dsp.get());
  assert(slimmable != nullptr);
  slimmable->SetSlimmableSize(1.0);
  dsp->ResetAndPrewarm(sample_rate, buffer_size);
  out_ptr = out_max.data();
  dsp->process(&in_ptr, &out_ptr, buffer_size);

  // Both should produce the same output (same model active)
  for (int i = 0; i < buffer_size; i++)
    assert(std::abs(out_default[i] - out_max[i]) < 1e-6);
}

} // namespace test_container
