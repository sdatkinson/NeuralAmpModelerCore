// Tests for per-layer kernel sizes in WaveNet LayerArrayParams and config parsing

#include <cassert>
#include <vector>

#include "json.hpp"

#include "NAM/wavenet.h"

namespace test_wavenet
{
namespace test_kernel_sizes
{

// Verify that an integer kernel_size in JSON is expanded to all layers
void test_kernel_size_int_compat()
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
        "kernel_size": 3,
        "dilations": [1, 2, 4],
        "activation": "ReLU",
        "gated": false,
        "head_bias": false
      }],
      "head_scale": 1.0
    },
    "weights": [1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
    "sample_rate": 48000
  })";

  const nlohmann::json j = nlohmann::json::parse(configStr);
  const auto wc = nam::wavenet::parse_config_json(j["config"], 48000.0);

  assert(wc.layer_array_params.size() == 1);
  const nam::wavenet::LayerArrayParams& p = wc.layer_array_params[0];

  const int expected_kernel_size = 3;
  const std::vector<int> expected_dilations{1, 2, 4};

  assert(p.kernel_size == expected_kernel_size);
  assert(p.dilations == expected_dilations);
  assert(p.kernel_sizes.size() == expected_dilations.size());
  for (size_t i = 0; i < p.kernel_sizes.size(); ++i)
  {
    assert(p.kernel_sizes[i] == expected_kernel_size);
  }

  // Verify that receptive field computation uses the (uniform) kernel size
  nam::wavenet::_LayerArray array(p);
  const long receptive_field = array.get_receptive_field();

  long expected_receptive_field = 0;
  for (size_t i = 0; i < expected_dilations.size(); ++i)
  {
    expected_receptive_field += expected_dilations[i] * (expected_kernel_size - 1);
  }
  assert(receptive_field == expected_receptive_field);
}

// Verify that an array kernel_size in JSON is honored per layer
void test_kernel_size_per_layer_array()
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
        "kernel_size": [2, 3, 5],
        "dilations": [1, 2, 4],
        "activation": "ReLU",
        "gated": false,
        "head_bias": false
      }],
      "head_scale": 1.0
    },
    "weights": [1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0],
    "sample_rate": 48000
  })";

  const nlohmann::json j = nlohmann::json::parse(configStr);
  const auto wc = nam::wavenet::parse_config_json(j["config"], 48000.0);

  assert(wc.layer_array_params.size() == 1);
  const nam::wavenet::LayerArrayParams& p = wc.layer_array_params[0];

  const std::vector<int> expected_kernel_sizes{2, 3, 5};
  const std::vector<int> expected_dilations{1, 2, 4};

  assert(p.kernel_sizes == expected_kernel_sizes);
  assert(p.dilations == expected_dilations);

  // Verify that receptive field computation uses the per-layer kernel sizes
  nam::wavenet::_LayerArray array(p);
  const long receptive_field = array.get_receptive_field();

  long expected_receptive_field = 0;
  for (size_t i = 0; i < expected_dilations.size(); ++i)
  {
    expected_receptive_field += expected_dilations[i] * (expected_kernel_sizes[i] - 1);
  }
  assert(receptive_field == expected_receptive_field);
}

} // namespace test_kernel_sizes
} // namespace test_wavenet
