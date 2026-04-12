// Layer-array head JSON: legacy head_size/head_bias vs nested "head" (out_channels, kernel_size, bias)

#include <cassert>
#include <string>

#include "json.hpp"

#include "NAM/wavenet.h"

namespace test_wavenet
{
namespace test_layer_head_config
{

void test_legacy_head_size_and_head_bias_implies_kernel_one()
{
  const std::string configStr = R"({
    "layers": [{
      "input_size": 1,
      "condition_size": 1,
      "head_size": 2,
      "channels": 2,
      "kernel_size": 1,
      "dilations": [1],
      "activation": "ReLU",
      "head_bias": false
    }],
    "head_scale": 1.0
  })";

  const nlohmann::json j = nlohmann::json::parse(configStr);
  const auto wc = nam::wavenet::parse_config_json(j, 48000.0);
  assert(wc.layer_array_params.size() == 1);
  const auto& p = wc.layer_array_params[0];
  assert(p.head_size == 2);
  assert(p.head_kernel_size == 1);
  assert(p.head_bias == false);
}

void test_nested_head_with_kernel_size_three()
{
  const std::string configStr = R"({
    "layers": [{
      "input_size": 1,
      "condition_size": 1,
      "head": {"out_channels": 1, "kernel_size": 3, "bias": true},
      "channels": 2,
      "kernel_size": 1,
      "dilations": [1],
      "activation": "ReLU"
    }],
    "head_scale": 1.0
  })";

  const nlohmann::json j = nlohmann::json::parse(configStr);
  const auto wc = nam::wavenet::parse_config_json(j, 48000.0);
  assert(wc.layer_array_params.size() == 1);
  const auto& p = wc.layer_array_params[0];
  assert(p.head_size == 1);
  assert(p.head_kernel_size == 3);
  assert(p.head_bias == true);

  nam::wavenet::_LayerArray array(p);
  assert(array.get_receptive_field() == 2); // one dilated layer: 0 + (3-1) head rechannel
}

} // namespace test_layer_head_config
} // namespace test_wavenet
