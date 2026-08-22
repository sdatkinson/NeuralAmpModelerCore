#include <algorithm>
#include <cassert>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "json.hpp"

#include "NAM/get_dsp.h"
#include "NAM/sequential.h"
#include "allocation_tracking.h"

namespace test_sequential
{
namespace
{

class TemporaryNamFile
{
public:
  explicit TemporaryNamFile(const nlohmann::json& contents)
  : path(std::filesystem::temp_directory_path() / "nam_core_sequential_test.nam")
  {
    std::ofstream output(path);
    output << contents;
  }

  ~TemporaryNamFile() { std::filesystem::remove(path); }

  const std::filesystem::path path;
};

nlohmann::json make_linear_model(const std::vector<float>& weights, const int receptive_field,
                                 const double sample_rate = 48000.0, const int in_channels = 1,
                                 const int out_channels = 1)
{
  return {{"version", "0.7.0"},
          {"architecture", "Linear"},
          {"config",
           {{"receptive_field", receptive_field},
            {"bias", false},
            {"implementation", "direct"},
            {"in_channels", in_channels},
            {"out_channels", out_channels}}},
          {"weights", weights},
          {"sample_rate", sample_rate}};
}

nlohmann::json make_sequential_model(const std::vector<nlohmann::json>& models, const double sample_rate = 48000.0)
{
  return {{"version", "0.7.0"},
          {"architecture", "Sequential"},
          {"metadata", nlohmann::json::object()},
          {"config", {{"models", models}}},
          {"weights", nlohmann::json::array()},
          {"sample_rate", sample_rate}};
}

std::vector<NAM_SAMPLE> make_input(const int num_samples)
{
  std::vector<NAM_SAMPLE> input(num_samples);
  for (int i = 0; i < num_samples; ++i)
    input[i] = (NAM_SAMPLE)(0.2 * std::sin(0.037 * i) + 0.05 * std::cos(0.011 * i));
  return input;
}

std::vector<NAM_SAMPLE> process_model(nam::DSP& dsp, const std::vector<NAM_SAMPLE>& input,
                                      const std::vector<int>& chunk_sizes)
{
  const int max_chunk = *std::max_element(chunk_sizes.begin(), chunk_sizes.end());
  dsp.Reset(48000.0, max_chunk);

  std::vector<NAM_SAMPLE> output(input.size(), (NAM_SAMPLE)0.0);
  size_t offset = 0;
  size_t chunk_index = 0;
  while (offset < input.size())
  {
    const int requested = chunk_sizes[chunk_index % chunk_sizes.size()];
    const int count = std::min<int>(requested, (int)(input.size() - offset));
    NAM_SAMPLE* input_ptr = const_cast<NAM_SAMPLE*>(&input[offset]);
    NAM_SAMPLE* output_ptr = &output[offset];
    dsp.process(&input_ptr, &output_ptr, count);
    offset += count;
    chunk_index++;
  }

  return output;
}

std::vector<NAM_SAMPLE> process_models_in_series(nam::DSP& first, nam::DSP& second,
                                                 const std::vector<NAM_SAMPLE>& input,
                                                 const std::vector<int>& chunk_sizes)
{
  const int max_chunk = *std::max_element(chunk_sizes.begin(), chunk_sizes.end());
  first.Reset(48000.0, max_chunk);
  second.Reset(48000.0, max_chunk);

  std::vector<NAM_SAMPLE> intermediate(max_chunk, (NAM_SAMPLE)0.0);
  std::vector<NAM_SAMPLE> output(input.size(), (NAM_SAMPLE)0.0);
  size_t offset = 0;
  size_t chunk_index = 0;
  while (offset < input.size())
  {
    const int requested = chunk_sizes[chunk_index % chunk_sizes.size()];
    const int count = std::min<int>(requested, (int)(input.size() - offset));
    NAM_SAMPLE* first_input_ptr = const_cast<NAM_SAMPLE*>(&input[offset]);
    NAM_SAMPLE* first_output_ptr = intermediate.data();
    first.process(&first_input_ptr, &first_output_ptr, count);

    NAM_SAMPLE* second_input_ptr = intermediate.data();
    NAM_SAMPLE* second_output_ptr = &output[offset];
    second.process(&second_input_ptr, &second_output_ptr, count);

    offset += count;
    chunk_index++;
  }

  return output;
}

bool throws_runtime_error_containing(const std::function<void()>& callback, const std::string& expected)
{
  try
  {
    callback();
  }
  catch (const std::runtime_error& error)
  {
    return std::string(error.what()).find(expected) != std::string::npos;
  }
  return false;
}

} // namespace

void test_sequential_loads_canonical_container_envelope()
{
  const auto model = make_sequential_model({make_linear_model({0.5f}, 1), make_linear_model({-2.0f}, 1)});

  assert(model.at("version") == "0.7.0");
  assert(model.at("architecture") == "Sequential");
  assert(model.at("weights").empty());
  assert(model.at("sample_rate") == 48000.0);
  assert(model.at("config").at("models").at(0).contains("architecture"));
  assert(model.at("config").at("models").at(0).contains("weights"));

  nam::dspData returned_config;
  auto dsp = nam::get_dsp(model, returned_config);

  assert(dsp != nullptr);
  assert(returned_config.weights.empty());
  assert(returned_config.expected_sample_rate == 48000.0);
  assert(dsp->GetExpectedSampleRate() == 48000.0);
}

void test_sequential_loads_from_file_path()
{
  const auto model = make_sequential_model({make_linear_model({0.5f}, 1), make_linear_model({-2.0f}, 1)});
  const TemporaryNamFile file(model);

  auto dsp = nam::get_dsp(file.path);

  assert(dsp != nullptr);
  assert(dsp->GetExpectedSampleRate() == 48000.0);
}

void test_sequential_process_matches_manual_series()
{
  const auto first_model = make_linear_model({0.25f, 0.5f}, 2);
  const auto second_model = make_linear_model({-0.75f}, 1);
  const auto sequential_model = make_sequential_model({first_model, second_model});

  auto sequential = nam::get_dsp(sequential_model);
  auto first = nam::get_dsp(first_model);
  auto second = nam::get_dsp(second_model);

  const auto input = make_input(257);
  const std::vector<int> chunks{1, 7, 32, 5, 64};
  const auto actual = process_model(*sequential, input, chunks);
  const auto expected = process_models_in_series(*first, *second, input, chunks);

  for (size_t i = 0; i < input.size(); ++i)
    assert(std::abs(actual[i] - expected[i]) < 1.0e-7);
}

void test_sequential_process_is_realtime_safe_after_warmup()
{
  auto dsp = nam::get_dsp(make_sequential_model({make_linear_model({0.5f}, 1), make_linear_model({-2.0f}, 1)}));
  constexpr int num_frames = 64;
  dsp->Reset(48000.0, num_frames);

  std::vector<NAM_SAMPLE> input(num_frames, (NAM_SAMPLE)0.25);
  std::vector<NAM_SAMPLE> output(num_frames, (NAM_SAMPLE)0.0);
  NAM_SAMPLE* input_ptr = input.data();
  NAM_SAMPLE* output_ptr = output.data();

  // Linear initializes an output buffer on its first process call; match its
  // existing real-time-safety test convention by warming that path first.
  dsp->process(&input_ptr, &output_ptr, num_frames);

  allocation_tracking::run_allocation_test_no_allocations(
    nullptr, [&]() { dsp->process(&input_ptr, &output_ptr, num_frames); }, nullptr,
    "test_sequential_process_is_realtime_safe_after_warmup");
}

void test_sequential_rejects_blocks_larger_than_reset_maximum()
{
  auto dsp = nam::get_dsp(make_sequential_model({make_linear_model({0.5f}, 1), make_linear_model({-2.0f}, 1)}));
  dsp->Reset(48000.0, 4);

  std::vector<NAM_SAMPLE> input(8, (NAM_SAMPLE)0.25);
  std::vector<NAM_SAMPLE> output(8, (NAM_SAMPLE)0.0);
  NAM_SAMPLE* input_ptr = input.data();
  NAM_SAMPLE* output_ptr = output.data();

  assert(throws_runtime_error_containing([&]() { dsp->process(&input_ptr, &output_ptr, 8); }, "maximum buffer size"));
}

void test_sequential_rejects_lowercase_architecture()
{
  auto model = make_sequential_model({make_linear_model({1.0f}, 1), make_linear_model({1.0f}, 1)});
  model["architecture"] = "sequential";

  assert(throws_runtime_error_containing([&]() { nam::get_dsp(model); },
                                         "No config parser registered for architecture: sequential"));
}

void test_sequential_accepts_nested_sequential_child()
{
  const auto inner = make_sequential_model({make_linear_model({1.0f}, 1), make_linear_model({1.0f}, 1)});
  const auto outer = make_sequential_model({inner, make_linear_model({1.0f}, 1)});

  auto dsp = nam::get_dsp(outer);

  assert(dsp != nullptr);
  assert(dsp->GetExpectedSampleRate() == 48000.0);
}

void test_sequential_rejects_empty_models()
{
  const auto model = make_sequential_model({});

  assert(throws_runtime_error_containing([&]() { auto dsp = nam::get_dsp(model); }, "non-empty"));
}

void test_sequential_rejects_nonempty_top_level_weights()
{
  auto model = make_sequential_model({make_linear_model({1.0f}, 1), make_linear_model({1.0f}, 1)});
  model["weights"] = nlohmann::json::array({1.0f});

  assert(throws_runtime_error_containing([&]() { auto dsp = nam::get_dsp(model); }, "top-level weights"));
}

void test_sequential_rejects_legacy_bare_child_configs()
{
  auto model = make_sequential_model({make_linear_model({1.0f}, 1), make_linear_model({1.0f}, 1)});
  model["config"]["models"] =
    nlohmann::json::array({{{"receptive_field", 1}, {"bias", false}}, {{"receptive_field", 1}, {"bias", false}}});

  assert(throws_runtime_error_containing([&]() { auto dsp = nam::get_dsp(model); }, "complete NAM model"));
}

void test_sequential_rejects_sample_rate_mismatch()
{
  const auto model =
    make_sequential_model({make_linear_model({1.0f}, 1, 48000.0), make_linear_model({1.0f}, 1, 44100.0)});

  assert(throws_runtime_error_containing([&]() { auto dsp = nam::get_dsp(model); }, "sample rate mismatch"));
}

void test_sequential_rejects_top_level_sample_rate_mismatch()
{
  const auto model =
    make_sequential_model({make_linear_model({1.0f}, 1, 48000.0), make_linear_model({1.0f}, 1, 48000.0)}, 44100.0);

  assert(throws_runtime_error_containing([&]() { auto dsp = nam::get_dsp(model); }, "sample rate mismatch"));
}

void test_sequential_rejects_channel_mismatch()
{
  const auto model =
    make_sequential_model({make_linear_model({1.0f}, 1, 48000.0, 1, 2), make_linear_model({1.0f}, 1, 48000.0, 1, 1)});

  assert(throws_runtime_error_containing([&]() { auto dsp = nam::get_dsp(model); }, "channel mismatch"));
}

} // namespace test_sequential
