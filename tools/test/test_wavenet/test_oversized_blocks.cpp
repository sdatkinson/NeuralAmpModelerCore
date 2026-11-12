// WaveNet::process() with more frames than the maximum buffer size: it used to write past the end of every
// buffer once asserts were compiled out. It must match the host making calls of at most that size, and not allocate.

#include <algorithm>
#include <cassert>
#include <cmath>
#include <filesystem>
#include <memory>
#include <vector>

#include "json.hpp"

#include "NAM/dsp.h"
#include "NAM/get_dsp.h"
#include "NAM/wavenet/model.h"
#include "../allocation_tracking.h"

namespace test_wavenet
{
namespace test_oversized_blocks
{
using Buffers = std::vector<std::vector<NAM_SAMPLE>>;

constexpr double kSampleRate = 48000.0;
constexpr int kMaxBufferSize = 32;
constexpr int kCallSize = 100; // Not a multiple of kMaxBufferSize, so each call's last chunk is short
constexpr int kNumCalls = 20;

// Two inputs and two outputs, so that every channel's pointer has to be advanced; optionally with a post-stack head
static std::unique_ptr<nam::DSP> make_two_channel_wavenet(const bool post_stack_head)
{
  nlohmann::json config = nlohmann::json::parse(R"({"in_channels": 2, "head": null, "head_scale": 0.02,
    "layers": [{"input_size": 2, "condition_size": 2, "head_size": 2, "channels": 3, "kernel_size": 3,
                "dilations": [1, 2], "activation": "Tanh", "gated": false, "head_bias": false}]})");
  // Rechannel 6; per layer: conv 27 + 3 bias, input mixin 6, 1x1 9 + 3 bias; head rechannel 6; head scale 1
  int num_weights = 6 + 2 * (30 + 6 + 12) + 6 + 1;
  if (post_stack_head)
  {
    config["head"] = nlohmann::json::parse(R"({"channels": 3, "out_channels": 2, "kernel_sizes": [3, 2],
                                               "activation": "Tanh"})");
    num_weights += 2 * 3 * 3 + 3 + 3 * 2 * 2 + 2; // head convs: 2->3 (kernel 3) + bias, 3->2 (kernel 2) + bias
  }
  std::vector<float> weights(num_weights);
  for (int i = 0; i < num_weights; i++)
  {
    weights[i] = 0.5f * std::sin(0.7f * (float)i);
  }
  return nam::wavenet::parse_config_json(config, kSampleRate).create(std::move(weights), kSampleRate);
}

// A plain WaveNet, one with the A2-era layer features, one with a condition DSP, and multichannel ones without
// and with a post-stack head
static std::vector<std::unique_ptr<nam::DSP>> make_models()
{
  std::vector<std::unique_ptr<nam::DSP>> models;
  for (const char* path : {"example_models/wavenet.nam", "example_models/wavenet_a2_feature_test.nam",
                           "example_models/wavenet_condition_dsp.nam"})
  {
    models.push_back(nam::get_dsp(std::filesystem::path(path)));
  }
  models.push_back(make_two_channel_wavenet(false));
  models.push_back(make_two_channel_wavenet(true));
  for (auto& model : models)
  {
    model->Reset(kSampleRate, kMaxBufferSize);
  }
  return models;
}

// One signal, interleaved across the channels
static Buffers make_input(const int num_channels)
{
  Buffers input(num_channels, std::vector<NAM_SAMPLE>(kNumCalls * kCallSize));
  for (size_t i = 0; i < input.size() * input[0].size(); i++)
  {
    input[i % input.size()][i / input.size()] =
      (NAM_SAMPLE)(0.5 * std::sin(0.01 * (double)i) * std::sin(0.37 * (double)i));
  }
  return input;
}

static void point_at(std::vector<NAM_SAMPLE*>& ptrs, Buffers& buffers, const int offset)
{
  for (size_t ch = 0; ch < buffers.size(); ch++)
  {
    ptrs[ch] = buffers[ch].data() + offset;
  }
}

// Process the input in calls of kCallSize frames, each made by the host as calls of at most host_block frames.
// No call may allocate.
static Buffers render(nam::DSP& dsp, Buffers input, const int host_block)
{
  Buffers output(dsp.NumOutputChannels(), std::vector<NAM_SAMPLE>(input[0].size()));
  std::vector<NAM_SAMPLE*> input_ptrs(input.size());
  std::vector<NAM_SAMPLE*> output_ptrs(output.size());
  for (int call = 0; call < kNumCalls * kCallSize; call += kCallSize)
  {
    for (int start = call; start < call + kCallSize; start += host_block)
    {
      point_at(input_ptrs, input, start);
      point_at(output_ptrs, output, start);
      const int num_frames = std::min(host_block, call + kCallSize - start);
      allocation_tracking::run_allocation_test_no_allocations(
        nullptr, [&]() { dsp.process(input_ptrs.data(), output_ptrs.data(), num_frames); }, nullptr,
        "WaveNet::process() with more frames than the maximum buffer size");
    }
  }
  return output;
}

void test_oversized_call_matches_consecutive_calls()
{
  auto oversized = make_models();
  auto consecutive = make_models();
  for (size_t m = 0; m < oversized.size(); m++)
  {
    const Buffers input = make_input(oversized[m]->NumInputChannels());
    const Buffers expected = render(*consecutive[m], input, kMaxBufferSize);
    const Buffers actual = render(*oversized[m], input, kCallSize);
    assert(actual == expected);
  }
}

} // namespace test_oversized_blocks
} // namespace test_wavenet
