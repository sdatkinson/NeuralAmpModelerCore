// Tests for WaveNet Factory

#include <cassert>
#include <cmath>
#include <memory>
#include <vector>

#include "json.hpp"

#include "NAM/get_dsp.h"
#include "NAM/wavenet.h"

namespace test_wavenet
{
namespace test_factory
{
/// Asserts that the model is instantiated correctly when no "head" key is provided.
/// The deprecated "head" key is optional; when absent, with_head should be false.
void test_factory_without_head_key()
{
  // Minimal WaveNet config - deliberately omits the "head" key entirely.
  // Same structure as wavenet.nam but without "head" in config.
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

  nlohmann::json j = nlohmann::json::parse(configStr);

  // Verify the config does not contain "head" key
  assert(j["config"].find("head") == j["config"].end());

  // Load model via get_dsp - exercises Factory path
  std::unique_ptr<nam::DSP> dsp = nam::get_dsp(j);
  assert(dsp != nullptr);

  // Process audio to verify model works correctly
  const int numFrames = 4;
  const int maxBufferSize = 64;
  dsp->Reset(48000.0, maxBufferSize);

  std::vector<NAM_SAMPLE> input(numFrames, 1.0f);
  std::vector<NAM_SAMPLE> output(numFrames, 0.0f);
  NAM_SAMPLE* inputPtrs[] = {input.data()};
  NAM_SAMPLE* outputPtrs[] = {output.data()};

  dsp->process(inputPtrs, outputPtrs, numFrames);

  assert(static_cast<int>(output.size()) == numFrames);
  for (int i = 0; i < numFrames; i++)
  {
    assert(std::isfinite(output[i]));
  }
}
}; // namespace test_factory
}; // namespace test_wavenet
