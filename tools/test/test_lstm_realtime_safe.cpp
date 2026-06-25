// Test to verify LSTM::process is real-time safe (no allocations/frees).
//
// Regression test for upstream issue #218 ("Fix LSTM real-time safety issues").
// The LSTM processes audio sample-by-sample on the audio thread, so its hot path
// (LSTM::process -> LSTM::_process_sample -> LSTMCell::process_) must not allocate
// or free any memory. The historical offender was LSTMCell::get_hidden_state()
// returning an Eigen::VectorXf by value, which heap-allocated once per layer hop
// (and once for the head) on every single sample.

#include <Eigen/Dense>
#include <cassert>
#include <cmath>
#include <string>
#include <vector>

#include "NAM/lstm.h"
#include "allocation_tracking.h"

namespace test_lstm_realtime_safe
{
using namespace allocation_tracking;

// Build a self-consistent weights vector for an LSTM.
// Layout matches nam::lstm::LSTM / LSTMCell construction order:
//   Per layer: weight matrix (4*hidden x (in+hidden), row-major), bias (4*hidden),
//              initial hidden state (hidden), initial cell state (hidden).
//   Head: weight matrix (out_channels x hidden, row-major), bias (out_channels).
static std::vector<float> make_weights(int num_layers, int input_size, int hidden_size, int out_channels)
{
  std::vector<float> weights;
  for (int layer = 0; layer < num_layers; layer++)
  {
    const int layer_input_size = (layer == 0) ? input_size : hidden_size;
    const int w_rows = 4 * hidden_size;
    const int w_cols = layer_input_size + hidden_size;
    for (int i = 0; i < w_rows * w_cols; i++)
      weights.push_back(0.1f); // small weights for numerical stability
    for (int i = 0; i < 4 * hidden_size; i++)
      weights.push_back(0.0f); // bias
    for (int i = 0; i < hidden_size; i++)
      weights.push_back(0.0f); // initial hidden state
    for (int i = 0; i < hidden_size; i++)
      weights.push_back(0.0f); // initial cell state
  }
  for (int out_ch = 0; out_ch < out_channels; out_ch++)
    for (int h = 0; h < hidden_size; h++)
      weights.push_back(0.1f); // head weight
  for (int out_ch = 0; out_ch < out_channels; out_ch++)
    weights.push_back(0.0f); // head bias
  return weights;
}

// Core helper: build an LSTM with the given shape, prewarm it, then assert that
// processing a block of audio performs zero allocations and zero frees.
static void check_no_allocations(const int in_channels, const int out_channels, const int num_layers,
                                 const int input_size, const int hidden_size, const int num_frames,
                                 const char* test_name)
{
  const double sample_rate = 48000.0;
  std::vector<float> weights = make_weights(num_layers, input_size, hidden_size, out_channels);
  nam::lstm::LSTM lstm(in_channels, out_channels, num_layers, input_size, hidden_size, weights, sample_rate);

  // Reset + prewarm before tracking so any one-time allocation happens up front.
  lstm.Reset(sample_rate, num_frames);
  lstm.prewarm();

  // Allocate the audio buffers and pointer arrays before tracking starts.
  std::vector<std::vector<NAM_SAMPLE>> input_bufs(in_channels, std::vector<NAM_SAMPLE>(num_frames, 0.25f));
  std::vector<std::vector<NAM_SAMPLE>> output_bufs(out_channels, std::vector<NAM_SAMPLE>(num_frames, 0.0f));
  std::vector<NAM_SAMPLE*> input_ptrs(in_channels);
  std::vector<NAM_SAMPLE*> output_ptrs(out_channels);
  for (int ch = 0; ch < in_channels; ch++)
    input_ptrs[ch] = input_bufs[ch].data();
  for (int ch = 0; ch < out_channels; ch++)
    output_ptrs[ch] = output_bufs[ch].data();

  run_allocation_test_no_allocations(
    nullptr, // no setup
    [&]() { lstm.process(input_ptrs.data(), output_ptrs.data(), num_frames); },
    nullptr, // no teardown
    test_name);

  // Sanity: output must be finite.
  for (int ch = 0; ch < out_channels; ch++)
    for (int i = 0; i < num_frames; i++)
      assert(std::isfinite(output_bufs[ch][i]));
}

// Single-layer, single-channel LSTM is real-time safe.
void test_lstm_process_single_layer_realtime_safe()
{
  check_no_allocations(/*in*/ 1, /*out*/ 1, /*layers*/ 1, /*input_size*/ 1, /*hidden*/ 8, /*frames*/ 64,
                       "LSTM process (1 layer, hidden=8)");
}

// Multi-layer LSTM exercises the inter-layer get_hidden_state() hops, which were
// the primary allocation source before the fix.
void test_lstm_process_multi_layer_realtime_safe()
{
  check_no_allocations(/*in*/ 1, /*out*/ 1, /*layers*/ 3, /*input_size*/ 1, /*hidden*/ 16, /*frames*/ 64,
                       "LSTM process (3 layers, hidden=16)");
}

// Multi-channel input/output path.
void test_lstm_process_multichannel_realtime_safe()
{
  check_no_allocations(/*in*/ 2, /*out*/ 2, /*layers*/ 2, /*input_size*/ 2, /*hidden*/ 8, /*frames*/ 32,
                       "LSTM process (2 layers, 2in/2out)");
}

// Larger hidden size, to make sure nothing scales into a per-call allocation.
void test_lstm_process_large_hidden_realtime_safe()
{
  check_no_allocations(/*in*/ 1, /*out*/ 1, /*layers*/ 2, /*input_size*/ 1, /*hidden*/ 64, /*frames*/ 128,
                       "LSTM process (2 layers, hidden=64)");
}

// Several consecutive process() calls (state persists across calls) must remain allocation-free.
void test_lstm_process_consecutive_calls_realtime_safe()
{
  const double sample_rate = 48000.0;
  const int in_channels = 1, out_channels = 1, num_layers = 2, input_size = 1, hidden_size = 16, num_frames = 48;
  std::vector<float> weights = make_weights(num_layers, input_size, hidden_size, out_channels);
  nam::lstm::LSTM lstm(in_channels, out_channels, num_layers, input_size, hidden_size, weights, sample_rate);
  lstm.Reset(sample_rate, num_frames);
  lstm.prewarm();

  std::vector<NAM_SAMPLE> input(num_frames, 0.3f);
  std::vector<NAM_SAMPLE> output(num_frames, 0.0f);
  NAM_SAMPLE* input_ptrs[] = {input.data()};
  NAM_SAMPLE* output_ptrs[] = {output.data()};

  run_allocation_test_no_allocations(
    nullptr,
    [&]() {
      for (int call = 0; call < 8; call++)
        lstm.process(input_ptrs, output_ptrs, num_frames);
    },
    nullptr,
    "LSTM process (8 consecutive calls)");

  for (int i = 0; i < num_frames; i++)
    assert(std::isfinite(output[i]));
}

} // namespace test_lstm_realtime_safe
