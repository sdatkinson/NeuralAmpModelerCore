// Test for issue #230: asserts that the model registry is extensible from outside.
// Defines a minimal "DummyArchitecture" that passes input through to output and
// registers it; get_dsp must detect the architecture and instantiate the model.

#include <cassert>
#include <cmath>
#include <fstream>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "json.hpp"

#include "NAM/dsp.h"
#include "NAM/get_dsp.h"
#include "NAM/registry.h"

namespace test_extensible
{

// --- DummyArchitecture: pass-through model (input -> output unaltered) ---

class DummyArchitecture : public nam::DSP
{
public:
  DummyArchitecture(const int in_channels, const int out_channels, const double expected_sample_rate)
  : nam::DSP(in_channels, out_channels, expected_sample_rate)
  {
  }

  // Use base class process() which copies input to output (and zeros extra output channels).
  // No override needed.
};

// --- Factory: build DummyArchitecture from JSON config ---

static std::unique_ptr<nam::DSP> DummyArchitectureFactory(const nlohmann::json& config, std::vector<float>& weights,
                                                          const double expectedSampleRate)
{
  (void)weights;
  const int in_channels = config.value("in_channels", 1);
  const int out_channels = config.value("out_channels", 1);
  return std::make_unique<DummyArchitecture>(in_channels, out_channels, expectedSampleRate);
}

// Register so get_dsp can instantiate "DummyArchitecture" .nam files.
namespace
{
static nam::factory::Helper _register_DummyArchitecture("DummyArchitecture", DummyArchitectureFactory);
}

// --- Tests ---

static void test_get_dsp_detects_dummy_architecture_from_json()
{
  const std::string configStr = R"({
    "version": "0.6.0",
    "metadata": {},
    "architecture": "DummyArchitecture",
    "config": { "in_channels": 1, "out_channels": 1 },
    "weights": [],
    "sample_rate": 48000
  })";

  nlohmann::json j = nlohmann::json::parse(configStr);
  std::unique_ptr<nam::DSP> dsp = nam::get_dsp(j); // This should work because the architecture is registered.
  assert(dsp != nullptr);
  assert(dsp->NumInputChannels() == 1);
  assert(dsp->NumOutputChannels() == 1);

  const int numFrames = 8;
  const double sampleRate = 48000.0;
  dsp->Reset(sampleRate, numFrames);

  std::vector<NAM_SAMPLE> input(numFrames);
  std::vector<NAM_SAMPLE> output(numFrames, 0.0);
  for (int i = 0; i < numFrames; i++)
    input[i] = static_cast<NAM_SAMPLE>(0.1 * (i + 1));

  NAM_SAMPLE* inputPtrs[] = {input.data()};
  NAM_SAMPLE* outputPtrs[] = {output.data()};
  dsp->process(inputPtrs, outputPtrs, numFrames);

  for (int i = 0; i < numFrames; i++)
  {
    assert(std::isfinite(output[i]));
    assert(std::abs(output[i] - input[i]) < 1e-9);
  }
}

static void test_get_dsp_detects_dummy_architecture_from_nam_file()
{
  const std::string namContent = R"({
    "version": "0.5.4",
    "metadata": {},
    "architecture": "DummyArchitecture",
    "config": { "in_channels": 2, "out_channels": 2 },
    "weights": [],
    "sample_rate": 44100
  })";

  std::filesystem::path path = std::filesystem::temp_directory_path() / "nam_test_dummy_extensible.nam";
  {
    std::ofstream f(path);
    assert(f);
    f << namContent;
  }

  std::unique_ptr<nam::DSP> dsp = nam::get_dsp(path);
  assert(dsp != nullptr);
  assert(dsp->NumInputChannels() == 2);
  assert(dsp->NumOutputChannels() == 2);
  assert(std::abs(dsp->GetExpectedSampleRate() - 44100.0) < 1e-9);

  const int numFrames = 4;
  dsp->Reset(44100.0, numFrames);

  std::vector<NAM_SAMPLE> in0(numFrames, 0.5f);
  std::vector<NAM_SAMPLE> in1(numFrames, -0.25f);
  std::vector<NAM_SAMPLE> out0(numFrames, 0.0f);
  std::vector<NAM_SAMPLE> out1(numFrames, 0.0f);
  NAM_SAMPLE* inputPtrs[] = {in0.data(), in1.data()};
  NAM_SAMPLE* outputPtrs[] = {out0.data(), out1.data()};
  dsp->process(inputPtrs, outputPtrs, numFrames);

  for (int i = 0; i < numFrames; i++)
  {
    assert(std::abs(out0[i] - in0[i]) < 1e-9);
    assert(std::abs(out1[i] - in1[i]) < 1e-9);
  }

  std::filesystem::remove(path);
}

void run_extensibility_tests()
{
  test_get_dsp_detects_dummy_architecture_from_json();
  test_get_dsp_detects_dummy_architecture_from_nam_file();
}

} // namespace test_extensible
