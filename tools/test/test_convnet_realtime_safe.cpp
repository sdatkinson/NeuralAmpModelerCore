// Test to verify ConvNet::process is real-time safe (no allocations/frees).
//
// Regression test for upstream issue #350 ("ConvNet::process() allocates and
// frees on every call"). The ConvNet counterpart of the LSTM fix for #218.
// Every hand-off between layers must use pre-sized buffers because process()
// runs on the audio thread.

#include <Eigen/Dense>
#include <cassert>
#include <cmath>
#include <string>
#include <vector>

#include "NAM/convnet.h"
#include "allocation_tracking.h"

namespace test_convnet_realtime_safe
{
using namespace allocation_tracking;

static const auto kActivation = nam::activations::ActivationConfig::simple(nam::activations::ActivationType::ReLU);

// Self-consistent weight vector for a ConvNet with the given shape.
// Layout: per block, Conv1D weights (kernel=2, row-major), optional Conv1D
// bias, optional batchnorm (mean, var, weight, bias, eps); then head weights
// (out_channels x channels, row-major) and head biases.
static std::vector<float> make_weights(const int in_channels, const int out_channels, const int channels,
                                       const std::vector<int>& dilations, const bool batchnorm)
{
  std::vector<float> weights;
  for (size_t i = 0; i < dilations.size(); i++)
  {
    const int block_in = (i == 0) ? in_channels : channels;
    const int num_conv_weights = 2 * block_in * channels + (batchnorm ? 0 : channels);
    for (int j = 0; j < num_conv_weights; j++)
      weights.push_back(0.1f); // small weights for numerical stability
    if (batchnorm)
    {
      for (int j = 0; j < 4 * channels; j++)
        weights.push_back((j < channels || j >= 3 * channels) ? 0.0f : 1.0f); // mean, var, weight, bias
      weights.push_back(1e-5f); // eps
    }
  }
  for (int j = 0; j < out_channels * channels; j++)
    weights.push_back(0.1f); // head weight
  for (int j = 0; j < out_channels; j++)
    weights.push_back(0.0f); // head bias
  return weights;
}

static void check_no_allocations(const int in_channels, const int out_channels, const int channels,
                                 const std::vector<int>& dilations, const bool batchnorm, const int max_buffer_size,
                                 const int num_frames, const char* test_name)
{
  std::vector<float> weights = make_weights(in_channels, out_channels, channels, dilations, batchnorm);
  nam::convnet::ConvNet convnet(in_channels, out_channels, channels, dilations, batchnorm, kActivation, weights,
                                48000.0);

  // Reset + prewarm move all one-time allocations before tracking starts.
  convnet.Reset(48000.0, max_buffer_size);
  convnet.prewarm();

  std::vector<std::vector<NAM_SAMPLE>> input_bufs(in_channels, std::vector<NAM_SAMPLE>(num_frames, 0.25f));
  std::vector<std::vector<NAM_SAMPLE>> output_bufs(out_channels, std::vector<NAM_SAMPLE>(num_frames, 0.0f));
  std::vector<NAM_SAMPLE*> input_ptrs(in_channels);
  std::vector<NAM_SAMPLE*> output_ptrs(out_channels);
  for (int ch = 0; ch < in_channels; ch++)
    input_ptrs[ch] = input_bufs[ch].data();
  for (int ch = 0; ch < out_channels; ch++)
    output_ptrs[ch] = output_bufs[ch].data();

  // One extra warm-up block at this frame count, outside tracking.
  convnet.process(input_ptrs.data(), output_ptrs.data(), num_frames);

  run_allocation_test_no_allocations(
    nullptr, [&]() { convnet.process(input_ptrs.data(), output_ptrs.data(), num_frames); }, nullptr, test_name);

  for (int ch = 0; ch < out_channels; ch++)
    for (int i = 0; i < num_frames; i++)
      assert(std::isfinite(output_bufs[ch][i]));
}

// One block: the input matrix was allocated per call before the fix.
void test_convnet_process_single_block_realtime_safe()
{
  check_no_allocations(/*in*/ 1, /*out*/ 1, /*channels*/ 2, {1}, /*batchnorm*/ false, /*max*/ 64, /*frames*/ 64,
                       "ConvNet process (1 block)");
}

// Several blocks: the inter-block copies were allocated per call.
void test_convnet_process_multiple_blocks_realtime_safe()
{
  check_no_allocations(/*in*/ 1, /*out*/ 1, /*channels*/ 2, {1, 2, 4}, /*batchnorm*/ false, /*max*/ 128,
                       /*frames*/ 128, "ConvNet process (3 blocks)");
}

// Batchnorm path must remain allocation-free too.
void test_convnet_process_batchnorm_realtime_safe()
{
  check_no_allocations(/*in*/ 1, /*out*/ 1, /*channels*/ 2, {1, 2}, /*batchnorm*/ true, /*max*/ 64, /*frames*/ 64,
                       "ConvNet process (batchnorm)");
}

// Multi-channel input/output.
void test_convnet_process_multichannel_realtime_safe()
{
  check_no_allocations(/*in*/ 2, /*out*/ 2, /*channels*/ 2, {1, 2}, /*batchnorm*/ false, /*max*/ 64, /*frames*/ 32,
                       "ConvNet process (2in/2out)");
}

// Hosts may pass fewer frames than the maximum: the head must not reallocate.
void test_convnet_process_varying_frames_realtime_safe()
{
  const int max_buffer_size = 512;
  std::vector<float> weights = make_weights(1, 1, 2, {1, 2}, false);
  nam::convnet::ConvNet convnet(1, 1, 2, {1, 2}, false, kActivation, weights, 48000.0);
  convnet.Reset(48000.0, max_buffer_size);
  convnet.prewarm();

  for (const int num_frames : {512, 256, 64, 1})
  {
    std::vector<NAM_SAMPLE> input(num_frames, 0.25f), output(num_frames, 0.0f);
    NAM_SAMPLE* input_ptrs[] = {input.data()};
    NAM_SAMPLE* output_ptrs[] = {output.data()};
    // Warm up this frame count before tracking.
    convnet.process(input_ptrs, output_ptrs, num_frames);
    run_allocation_test_no_allocations(
      nullptr, [&]() { convnet.process(input_ptrs, output_ptrs, num_frames); }, nullptr,
      "ConvNet process (varying frames)");
    for (int i = 0; i < num_frames; i++)
      assert(std::isfinite(output[i]));
  }
}

// State persists across calls; consecutive calls must remain allocation-free.
void test_convnet_process_consecutive_calls_realtime_safe()
{
  const int num_frames = 48;
  std::vector<float> weights = make_weights(1, 1, 2, {1, 2}, false);
  nam::convnet::ConvNet convnet(1, 1, 2, {1, 2}, false, kActivation, weights, 48000.0);
  convnet.Reset(48000.0, num_frames);
  convnet.prewarm();

  std::vector<NAM_SAMPLE> input(num_frames, 0.25f), output(num_frames, 0.0f);
  NAM_SAMPLE* input_ptrs[] = {input.data()};
  NAM_SAMPLE* output_ptrs[] = {output.data()};
  convnet.process(input_ptrs, output_ptrs, num_frames);

  run_allocation_test_no_allocations(
    nullptr,
    [&]() {
      for (int call = 0; call < 8; call++)
        convnet.process(input_ptrs, output_ptrs, num_frames);
    },
    nullptr, "ConvNet process (8 consecutive calls)");

  for (int i = 0; i < num_frames; i++)
    assert(std::isfinite(output[i]));
}

} // namespace test_convnet_realtime_safe
