#include <algorithm>
#include <cassert>
#include <cmath>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "json.hpp"

#include "NAM/get_dsp.h"
#include "NAM/sequential.h"

namespace test_sequential
{
namespace
{

nlohmann::json make_linear_model(const std::vector<float>& weights, const int receptive_field,
                                 const double sample_rate = 48000.0, const int in_channels = 1,
                                 const int out_channels = 1)
{
  nlohmann::json model;
  model["version"] = "0.7.0";
  model["architecture"] = "Linear";
  model["config"] = {{"receptive_field", receptive_field},
                     {"bias", false},
                     {"implementation", "direct"},
                     {"in_channels", in_channels},
                     {"out_channels", out_channels}};
  model["weights"] = weights;
  model["sample_rate"] = sample_rate;
  return model;
}

nlohmann::json make_sequential_model(const nlohmann::json& first_model, const nlohmann::json& second_model,
                                     const std::optional<int> weights_version = 2)
{
  nlohmann::json model;
  model["version"] = "0.5.5";
  model["architecture"] = "sequential";
  model["metadata"] = nlohmann::json::object();
  if (weights_version.has_value())
    model["config"]["weights_version"] = weights_version.value();
  model["config"]["models"] = nlohmann::json::array({first_model, second_model});
  return model;
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

bool throws_runtime_error(const std::function<void()>& callback)
{
  try
  {
    callback();
  }
  catch (const std::runtime_error&)
  {
    return true;
  }
  return false;
}

bool throws_runtime_error_containing(const std::function<void()>& callback, const std::string& expected)
{
  try
  {
    callback();
  }
  catch (const std::runtime_error& e)
  {
    return std::string(e.what()).find(expected) != std::string::npos;
  }
  return false;
}

} // namespace

void test_sequential_loads_nested_models_without_top_level_weights_or_sample_rate()
{
  const auto first_model = make_linear_model({0.5f}, 1);
  const auto second_model = make_linear_model({-2.0f}, 1);
  const auto sequential_model = make_sequential_model(first_model, second_model);

  assert(!sequential_model.contains("weights"));
  assert(!sequential_model.contains("sample_rate"));

  nam::dspData returned_config;
  auto dsp = nam::get_dsp(sequential_model, returned_config);

  assert(dsp != nullptr);
  assert(returned_config.weights.empty());
  assert(returned_config.expected_sample_rate == 48000.0);
  assert(dsp->GetExpectedSampleRate() == 48000.0);
}

void test_sequential_process_matches_manual_series()
{
  const auto first_model = make_linear_model({0.25f, 0.5f}, 2);
  const auto second_model = make_linear_model({-0.75f}, 1);
  const auto sequential_model = make_sequential_model(first_model, second_model);

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

void test_sequential_accepts_layers_model_wrappers_and_pascal_case_name()
{
  const auto first_model = make_linear_model({1.0f}, 1);
  const auto second_model = make_linear_model({1.0f}, 1);

  nlohmann::json sequential_model;
  sequential_model["version"] = "0.5.5";
  sequential_model["architecture"] = "Sequential";
  sequential_model["config"]["weights_version"] = 2;
  sequential_model["config"]["layers"] = nlohmann::json::array({{{"model", first_model}}, {{"model", second_model}}});

  auto dsp = nam::get_dsp(sequential_model);
  assert(dsp != nullptr);
  assert(dsp->GetExpectedSampleRate() == 48000.0);
}

void test_sequential_accepts_nested_sequential_child()
{
  const auto first_model = make_linear_model({1.0f}, 1);
  const auto second_model = make_linear_model({1.0f}, 1);
  const auto third_model = make_linear_model({1.0f}, 1);
  const auto inner_model = make_sequential_model(first_model, second_model);
  const auto outer_model = make_sequential_model(inner_model, third_model);

  assert(!inner_model.contains("weights"));
  auto dsp = nam::get_dsp(outer_model);
  assert(dsp != nullptr);
  assert(dsp->GetExpectedSampleRate() == 48000.0);
}

void test_sequential_empty_models_throw()
{
  nlohmann::json model;
  model["version"] = "0.5.5";
  model["architecture"] = "sequential";
  model["config"]["models"] = nlohmann::json::array();

  assert(throws_runtime_error([&]() { auto dsp = nam::get_dsp(model); }));
}

void test_sequential_top_level_weights_throw()
{
  auto model = make_sequential_model(make_linear_model({1.0f}, 1), make_linear_model({1.0f}, 1));
  model["weights"] = nlohmann::json::array({1.0f});

  assert(throws_runtime_error_containing([&]() { auto dsp = nam::get_dsp(model); }, "weights_version=2"));
}

void test_sequential_missing_weights_version_is_legacy_and_throws()
{
  auto model = make_sequential_model(make_linear_model({1.0f}, 1), make_linear_model({1.0f}, 1), std::nullopt);
  model["weights"] = nlohmann::json::array({1.0f, 1.0f});
  model["sample_rate"] = 48000;

  assert(throws_runtime_error_containing([&]() { auto dsp = nam::get_dsp(model); }, "weights_version=1"));
}

void test_sequential_unsupported_weights_version_throws()
{
  auto model = make_sequential_model(make_linear_model({1.0f}, 1), make_linear_model({1.0f}, 1), 3);

  assert(throws_runtime_error_containing([&]() { auto dsp = nam::get_dsp(model); }, "weights_version 3"));
}

void test_sequential_sample_rate_mismatch_throws()
{
  auto first_model = make_linear_model({1.0f}, 1, 48000.0);
  auto second_model = make_linear_model({1.0f}, 1, 44100.0);
  const auto model = make_sequential_model(first_model, second_model);

  assert(throws_runtime_error([&]() { auto dsp = nam::get_dsp(model); }));
}

void test_sequential_channel_mismatch_throws()
{
  auto first_model = make_linear_model({1.0f}, 1, 48000.0, 1, 2);
  auto second_model = make_linear_model({1.0f}, 1, 48000.0, 1, 1);
  const auto model = make_sequential_model(first_model, second_model);

  assert(throws_runtime_error([&]() { auto dsp = nam::get_dsp(model); }));
}

} // namespace test_sequential
