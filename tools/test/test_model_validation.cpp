// Tests that malformed `.nam` model files are rejected with std::runtime_error instead of
// hitting undefined behavior (see NAM/json_util.h). `.nam` files are untrusted input
// downloaded from the internet, so every field the loader treats as required must fail
// safely when it is missing or out of range.

#include <cassert>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "json.hpp"

#include "NAM/get_dsp.h"
#include "NAM/json_util.h"

namespace test_model_validation
{
namespace
{

// A minimal, known-good WaveNet model (same shape as the one in
// test_wavenet::test_factory::test_factory_without_head_key), used as a baseline that
// individual tests mutate to introduce exactly one defect.
nlohmann::json build_valid_wavenet_config()
{
  const std::string configStr = R"({
    "version": "0.5.4",
    "metadata": {},
    "architecture": "WaveNet",
    "config": {
      "layers": [{
        "input_size": 1,
        "condition_size": 1,
        "head_size": 1,
        "channels": 1,
        "kernel_size": 1,
        "dilations": [1],
        "activation": "ReLU",
        "gated": false,
        "head_bias": false
      }],
      "head_scale": 1.0
    },
    "weights": [1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
    "sample_rate": 48000
  })";
  return nlohmann::json::parse(configStr);
}

// A minimal, known-good Linear model.
nlohmann::json build_valid_linear_config()
{
  const std::string configStr = R"({
    "version": "0.5.4",
    "metadata": {},
    "architecture": "Linear",
    "config": {"receptive_field": 4, "bias": true},
    "weights": [0.1, 0.2, 0.3, 0.4, 0.05],
    "sample_rate": 48000
  })";
  return nlohmann::json::parse(configStr);
}

} // namespace

void test_missing_version_throws()
{
  nlohmann::json j = build_valid_wavenet_config();
  j.erase("version");
  try
  {
    nam::get_dsp(j);
    assert(false && "should have thrown");
  }
  catch (const std::runtime_error&)
  {
  }
}

void test_missing_architecture_throws()
{
  nlohmann::json j = build_valid_wavenet_config();
  j.erase("architecture");
  try
  {
    nam::get_dsp(j);
    assert(false && "should have thrown");
  }
  catch (const std::runtime_error&)
  {
  }
}

void test_missing_config_throws()
{
  nlohmann::json j = build_valid_wavenet_config();
  j.erase("config");
  try
  {
    nam::get_dsp(j);
    assert(false && "should have thrown");
  }
  catch (const std::runtime_error&)
  {
  }
}

void test_wavenet_missing_layers_throws()
{
  nlohmann::json j = build_valid_wavenet_config();
  j["config"].erase("layers");
  try
  {
    nam::get_dsp(j);
    assert(false && "should have thrown");
  }
  catch (const std::runtime_error&)
  {
  }
}

void test_wavenet_negative_channels_throws()
{
  nlohmann::json j = build_valid_wavenet_config();
  j["config"]["layers"][0]["channels"] = -1;
  try
  {
    nam::get_dsp(j);
    assert(false && "should have thrown");
  }
  catch (const std::runtime_error&)
  {
  }
}

void test_wavenet_absurdly_large_channels_throws()
{
  nlohmann::json j = build_valid_wavenet_config();
  j["config"]["layers"][0]["channels"] = 100000000;
  try
  {
    nam::get_dsp(j);
    assert(false && "should have thrown");
  }
  catch (const std::runtime_error&)
  {
  }
}

void test_wavenet_empty_dilations_throws()
{
  nlohmann::json j = build_valid_wavenet_config();
  j["config"]["layers"][0]["dilations"] = nlohmann::json::array();
  try
  {
    nam::get_dsp(j);
    assert(false && "should have thrown");
  }
  catch (const std::runtime_error&)
  {
  }
}

void test_linear_missing_receptive_field_throws()
{
  nlohmann::json j = build_valid_linear_config();
  j["config"].erase("receptive_field");
  try
  {
    nam::get_dsp(j);
    assert(false && "should have thrown");
  }
  catch (const std::runtime_error&)
  {
  }
}

// "layer1x1" present but empty must throw instead of hitting nlohmann's const-operator[]
// assert (JSON_ASSERT / UB under -DNDEBUG) when reading "active"/"groups".
void test_wavenet_empty_layer1x1_throws()
{
  nlohmann::json j = build_valid_wavenet_config();
  j["config"]["layers"][0]["layer1x1"] = nlohmann::json::object();
  try
  {
    nam::get_dsp(j);
    assert(false && "should have thrown");
  }
  catch (const std::runtime_error&)
  {
  }
}

// "head1x1" present with only "active" (missing "out_channels"/"groups") must throw instead
// of hitting the same const-operator[] UB.
void test_wavenet_head1x1_missing_fields_throws()
{
  nlohmann::json j = build_valid_wavenet_config();
  j["config"]["layers"][0]["head1x1"] = {{"active", true}};
  try
  {
    nam::get_dsp(j);
    assert(false && "should have thrown");
  }
  catch (const std::runtime_error&)
  {
  }
}

// An "activation" object missing its "type" field must throw instead of hitting the same
// const-operator[] UB in ActivationConfig::from_json.
void test_wavenet_activation_missing_type_throws()
{
  nlohmann::json j = build_valid_wavenet_config();
  j["config"]["layers"][0]["activation"] = nlohmann::json::object();
  try
  {
    nam::get_dsp(j);
    assert(false && "should have thrown");
  }
  catch (const std::runtime_error&)
  {
  }
}

// "groups_input": 0 must be rejected, not divide-by-zero downstream (channels % groups).
void test_wavenet_zero_groups_input_throws()
{
  nlohmann::json j = build_valid_wavenet_config();
  j["config"]["layers"][0]["groups_input"] = 0;
  try
  {
    nam::get_dsp(j);
    assert(false && "should have thrown");
  }
  catch (const std::runtime_error&)
  {
  }
}

// A FiLM block's "groups" reaches the same `x % groups` as groups_input, so zero must be rejected too.
void test_wavenet_zero_film_groups_throws()
{
  nlohmann::json j = build_valid_wavenet_config();
  j["config"]["layers"][0]["conv_pre_film"] = {{"active", true}, {"shift", true}, {"groups", 0}};
  try
  {
    nam::get_dsp(j);
    assert(false && "should have thrown");
  }
  catch (const std::runtime_error&)
  {
  }
}

// A negative kernel_size must be rejected, not become a huge size_t in a downstream resize().
void test_wavenet_negative_kernel_size_throws()
{
  nlohmann::json j = build_valid_wavenet_config();
  j["config"]["layers"][0]["kernel_size"] = -1;
  try
  {
    nam::get_dsp(j);
    assert(false && "should have thrown");
  }
  catch (const std::runtime_error&)
  {
  }
}

// An over-long "dilations" array (beyond nam::util::kMaxModelArrayLength) must be rejected--
// each element drives construction of a heap-allocated Layer object.
void test_wavenet_overlong_dilations_throws()
{
  nlohmann::json j = build_valid_wavenet_config();
  nlohmann::json dilations = nlohmann::json::array();
  for (int i = 0; i < 5000; i++)
  {
    dilations.push_back(1);
  }
  j["config"]["layers"][0]["dilations"] = dilations;
  try
  {
    nam::get_dsp(j);
    assert(false && "should have thrown");
  }
  catch (const std::runtime_error&)
  {
  }
}

// A non-integral dimension (a JSON float) must be rejected rather than silently truncated.
void test_wavenet_non_integral_channels_throws()
{
  nlohmann::json j = build_valid_wavenet_config();
  j["config"]["layers"][0]["channels"] = 4.9;
  try
  {
    nam::get_dsp(j);
    assert(false && "should have thrown");
  }
  catch (const std::runtime_error&)
  {
  }
}

// Boundary tests for nam::util::RequireDimension, exercised directly rather than through a
// full model load: a model with channels == kMaxModelDimension would try to allocate
// matrices on the order of kMaxModelDimension^2 floats, which isn't a reasonable thing for a
// unit test to do just to exercise a boundary condition.
void test_dimension_boundary_at_max_accepted()
{
  const nlohmann::json j = {{"channels", nam::util::kMaxModelDimension}};
  const int value = nam::util::RequireDimension(j, "channels", "boundary test");
  assert(value == nam::util::kMaxModelDimension);
}

void test_dimension_boundary_beyond_max_throws()
{
  const nlohmann::json j = {{"channels", nam::util::kMaxModelDimension + 1}};
  try
  {
    nam::util::RequireDimension(j, "channels", "boundary test");
    assert(false && "should have thrown");
  }
  catch (const std::runtime_error&)
  {
  }
}

// A well-formed WaveNet config must still load and process audio unchanged.
void test_valid_wavenet_config_still_loads()
{
  nlohmann::json j = build_valid_wavenet_config();
  std::unique_ptr<nam::DSP> dsp = nam::get_dsp(j);
  assert(dsp != nullptr);

  const int numFrames = 4;
  const int maxBufferSize = 64;
  dsp->Reset(48000.0, maxBufferSize);

  std::vector<NAM_SAMPLE> input(numFrames, 1.0f);
  std::vector<NAM_SAMPLE> output(numFrames, 0.0f);
  NAM_SAMPLE* inputPtrs[] = {input.data()};
  NAM_SAMPLE* outputPtrs[] = {output.data()};

  dsp->process(inputPtrs, outputPtrs, numFrames);

  for (int i = 0; i < numFrames; i++)
  {
    assert(std::isfinite(output[i]));
  }
}

// A well-formed Linear config must still load and process audio unchanged.
void test_valid_linear_config_still_loads()
{
  nlohmann::json j = build_valid_linear_config();
  std::unique_ptr<nam::DSP> dsp = nam::get_dsp(j);
  assert(dsp != nullptr);

  const int numFrames = 4;
  const int maxBufferSize = 64;
  dsp->Reset(48000.0, maxBufferSize);

  std::vector<NAM_SAMPLE> input(numFrames, 1.0f);
  std::vector<NAM_SAMPLE> output(numFrames, 0.0f);
  NAM_SAMPLE* inputPtrs[] = {input.data()};
  NAM_SAMPLE* outputPtrs[] = {output.data()};

  dsp->process(inputPtrs, outputPtrs, numFrames);

  for (int i = 0; i < numFrames; i++)
  {
    assert(std::isfinite(output[i]));
  }
}
}; // namespace test_model_validation
