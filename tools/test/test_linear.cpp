// Tests for Linear DSP models

#include "NAM/dsp.h"
#include "NAM/linear.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <stdexcept>
#include <limits>
#include <vector>

#include "allocation_tracking.h"

namespace test_linear
{
namespace
{

std::vector<NAM_SAMPLE> process_model(nam::Linear& model, const std::vector<NAM_SAMPLE>& input,
                                      const std::vector<int>& chunk_sizes)
{
  std::vector<NAM_SAMPLE> output(input.size(), (NAM_SAMPLE)0.0);
  NAM_SAMPLE* input_ptrs[1];
  NAM_SAMPLE* output_ptrs[1];

  size_t offset = 0;
  size_t chunk_index = 0;
  while (offset < input.size())
  {
    const int requested = chunk_sizes[chunk_index % chunk_sizes.size()];
    const int count = std::min<int>(requested, (int)(input.size() - offset));
    input_ptrs[0] = const_cast<NAM_SAMPLE*>(&input[offset]);
    output_ptrs[0] = &output[offset];
    model.process(input_ptrs, output_ptrs, count);
    offset += count;
    chunk_index++;
  }
  return output;
}

std::vector<NAM_SAMPLE> make_input(const int num_samples)
{
  std::vector<NAM_SAMPLE> input(num_samples);
  for (int i = 0; i < num_samples; i++)
    input[i] = (NAM_SAMPLE)(0.2 * std::sin(0.013 * i) + 0.05 * std::cos(0.071 * i));
  return input;
}

std::vector<float> make_weights(const int receptive_field, const bool bias)
{
  std::vector<float> weights;
  weights.reserve(receptive_field + (bias ? 1 : 0));
  for (int i = 0; i < receptive_field; i++)
    weights.push_back((float)(std::exp(-0.001 * i) * std::sin(0.037 * (i + 1)) * 0.01));
  if (bias)
    weights.push_back(0.03125f);
  return weights;
}

void assert_near(const NAM_SAMPLE actual, const NAM_SAMPLE expected, const NAM_SAMPLE tolerance)
{
  assert(std::abs(actual - expected) <= tolerance);
}

void assert_process_realtime_safe(const int receptive_field, const nam::LinearImplementation requested_implementation,
                                  const nam::LinearImplementation expected_active_implementation, const char* test_name)
{
  const int max_buffer_size = 512;
  const auto weights = make_weights(receptive_field, true);
  nam::Linear model(1, 1, receptive_field, true, weights, 48000.0, requested_implementation);
  model.Reset(48000.0, max_buffer_size);
  assert(model.GetActiveImplementation() == expected_active_implementation);

  std::vector<NAM_SAMPLE> input(max_buffer_size);
  std::vector<NAM_SAMPLE> output(max_buffer_size);
  for (int i = 0; i < max_buffer_size; i++)
    input[i] = (NAM_SAMPLE)(0.1 * std::sin(0.021 * i) + 0.03 * std::cos(0.017 * i));

  NAM_SAMPLE* input_ptrs[1] = {input.data()};
  NAM_SAMPLE* output_ptrs[1] = {output.data()};

  model.process(input_ptrs, output_ptrs, max_buffer_size);

  const int block_sizes[] = {1, 7, 32, 64, 128, 256, 3, 511, 512};
  allocation_tracking::run_allocation_test_no_allocations(
    nullptr,
    [&]() {
      for (int pass = 0; pass < 8; pass++)
      {
        for (const int block_size : block_sizes)
          model.process(input_ptrs, output_ptrs, block_size);
      }
    },
    nullptr, test_name);

  for (int i = 0; i < max_buffer_size; i++)
    assert(std::isfinite(output[i]));
}

} // namespace

void test_direct_known_values()
{
  const std::vector<float> weights{0.5f, -0.25f, 0.125f};
  nam::Linear model(1, 1, 3, false, weights, 48000.0, nam::LinearImplementation::Direct);

  const std::vector<NAM_SAMPLE> input{(NAM_SAMPLE)1.0, (NAM_SAMPLE)2.0, (NAM_SAMPLE)3.0, (NAM_SAMPLE)4.0};
  const auto output = process_model(model, input, {4});

  assert_near(output[0], 0.5, 1.0e-7);
  assert_near(output[1], 0.75, 1.0e-7);
  assert_near(output[2], 1.125, 1.0e-7);
  assert_near(output[3], 1.5, 1.0e-7);
}

void test_fft_matches_direct_irregular_chunks()
{
  const int receptive_field = 1536;
  const bool bias = true;
  const auto weights = make_weights(receptive_field, bias);
  const auto input = make_input(4096);

  nam::Linear direct(1, 1, receptive_field, bias, weights, 48000.0, nam::LinearImplementation::Direct);
  nam::Linear fft(1, 1, receptive_field, bias, weights, 48000.0, nam::LinearImplementation::FFT);

  const std::vector<int> chunks{1, 17, 64, 255, 3, 512, 31};
  const auto direct_output = process_model(direct, input, chunks);
  const auto fft_output = process_model(fft, input, chunks);

  NAM_SAMPLE max_abs_diff = 0.0;
  for (size_t i = 0; i < input.size(); i++)
    max_abs_diff = std::max<NAM_SAMPLE>(max_abs_diff, std::abs(direct_output[i] - fft_output[i]));

  assert(max_abs_diff < 5.0e-5);
}

void test_auto_selection()
{
  const auto short_weights = make_weights(128, false);
  nam::Linear short_model(1, 1, 128, false, short_weights, 48000.0);
  assert(short_model.GetRequestedImplementation() == nam::LinearImplementation::Auto);
  assert(short_model.GetActiveImplementation() == nam::LinearImplementation::Direct);

  const auto cutoff_weights = make_weights(1024, false);
  nam::Linear cutoff_model(1, 1, 1024, false, cutoff_weights, 48000.0);
  assert(cutoff_model.GetRequestedImplementation() == nam::LinearImplementation::Auto);
  assert(cutoff_model.GetActiveImplementation() == nam::LinearImplementation::Direct);

  const auto fft_weights = make_weights(2048, false);
  nam::Linear fft_model(1, 1, 2048, false, fft_weights, 48000.0);
  assert(fft_model.GetRequestedImplementation() == nam::LinearImplementation::Auto);
  assert(fft_model.GetActiveImplementation() == nam::LinearImplementation::FFT);
}

void test_fft_dispatch_table()
{
  assert(nam::linear::select_implementation(1024) == nam::LinearImplementation::Direct);
  assert(nam::linear::select_implementation(1025) == nam::LinearImplementation::FFT);
  assert(nam::linear::select_fft_plan(1024).direct_taps == 128);
  assert(nam::linear::select_fft_plan(8192).max_partition_size == 2048);
  assert(nam::linear::select_fft_plan(48000).max_partition_size == 4096);
  assert(nam::linear::select_fft_plan(240000).max_partition_size == 8192);
  assert(nam::linear::select_fft_plan(2880000).max_partition_size == 8192);
}

void test_fft_impulse_response_across_dispatch_sizes()
{
  const std::vector<int> receptive_fields{1024, 2048, 4096, 8192, 48000};
  for (const int receptive_field : receptive_fields)
  {
    const auto weights = make_weights(receptive_field, false);
    nam::Linear model(1, 1, receptive_field, false, weights, 48000.0, nam::LinearImplementation::FFT);
    std::vector<NAM_SAMPLE> input(receptive_field + 257, 0.0);
    input[0] = 1.0;
    const auto output = process_model(model, input, {1, 17, 32, 63, 128, 511});
    for (int i = 0; i < receptive_field; ++i)
      assert_near(output[i], weights[i], 5.0e-5);
    for (size_t i = receptive_field; i < output.size(); ++i)
      assert_near(output[i], 0.0, 5.0e-5);
  }
}

void test_parse_implementation()
{
  assert(nam::linear::parse_implementation("auto") == nam::LinearImplementation::Auto);
  assert(nam::linear::parse_implementation("legacy") == nam::LinearImplementation::Direct);
  assert(nam::linear::parse_implementation("partitioned-fft") == nam::LinearImplementation::FFT);
  assert(nam::linear::implementation_to_string(nam::LinearImplementation::Direct) == "direct");

  bool threw = false;
  try
  {
    nam::linear::parse_implementation("not-a-real-implementation");
  }
  catch (const std::runtime_error&)
  {
    threw = true;
  }
  assert(threw);
}

void test_direct_process_realtime_safe()
{
  assert_process_realtime_safe(
    512, nam::LinearImplementation::Direct, nam::LinearImplementation::Direct, "Linear direct process real-time safe");
}

void test_fft_process_realtime_safe()
{
  assert_process_realtime_safe(
    4096, nam::LinearImplementation::FFT, nam::LinearImplementation::FFT, "Linear FFT process real-time safe");
}

void test_auto_direct_process_realtime_safe()
{
  assert_process_realtime_safe(128, nam::LinearImplementation::Auto, nam::LinearImplementation::Direct,
                               "Linear auto direct process real-time safe");
}

void test_auto_fft_process_realtime_safe()
{
  assert_process_realtime_safe(
    4096, nam::LinearImplementation::Auto, nam::LinearImplementation::FFT, "Linear auto FFT process real-time safe");
}


void test_arbitrary_sample_rate_capability()
{
  nam::DSP fixed_rate(1, 1, 48000.0);
  assert(!fixed_rate.SupportsArbitrarySampleRate());
  nam::Linear linear(1, 1, 1, false, {1.0f}, 48000.0);
  nam::DSP& model = linear;
  assert(model.SupportsArbitrarySampleRate());
  model.Reset(96000.0, 4);
  assert(model.SupportsArbitrarySampleRate());

  nam::Linear unknown(1, 1, 1, false, {1.0f});
  nam::DSP& unknown_model = unknown;
  assert(!unknown_model.SupportsArbitrarySampleRate());
  unknown_model.Reset(44100.0, 4);
  assert(!unknown_model.SupportsArbitrarySampleRate());

  for (const double rate :
       {0.0, -2.0, std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()})
  {
    nam::Linear invalid_rate(1, 1, 1, false, {1.0f}, rate);
    assert(!invalid_rate.SupportsArbitrarySampleRate());
  }
}

// The cubic interpolation of a delayed unit impulse has these exact values.
// The factor originalRate / desiredRate preserves the IR's gain.
void test_sample_rate_known_values()
{
  for (const auto implementation : {nam::LinearImplementation::Direct, nam::LinearImplementation::FFT})
  {
    const std::vector<float> weights{0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.125f};
    nam::Linear model(1, 1, 5, true, weights, 48000.0, implementation);
    model.SetPrewarmOnReset(false);
    std::vector<NAM_SAMPLE> input(16, 0.0);
    input[0] = 1.0;
    // Repeated changes must always use the original weights, including when
    // returning to the training rate after processing nonzero audio.
    for (int repeat = 0; repeat < 2; ++repeat)
    {
      for (const double rate : {96000.0, 24000.0, 48000.0})
      {
        model.Reset(rate, 7);
        assert(model.GetExpectedSampleRate() == 48000.0);
        const auto output = process_model(model, input, {1, 7, 3});
        const std::vector<double> expected =
          rate == 96000.0   ? std::vector<double>{0.0, -0.03125, 0.0, 0.28125, 0.5, 0.28125, 0.0, -0.03125, 0.0, 0.0}
          : rate == 24000.0 ? std::vector<double>{0.0, 2.0, 0.0}
                            : std::vector<double>{0.0, 0.0, 1.0, 0.0, 0.0};
        for (size_t i = 0; i < output.size(); ++i)
          assert_near(output[i], 0.125 + (i < expected.size() ? expected[i] : 0.0), 1.0e-6);
      }
    }
  }
}

void test_sample_rate_fractional_and_fft()
{
  const int taps = 1200;
  std::vector<float> weights(taps, 0.0f);
  // A linear ramp is reproduced exactly by cubic interpolation away from the boundaries.
  for (int i = 0; i < taps; ++i)
    weights[i] = (float)i / taps;
  nam::Linear direct(1, 1, taps, false, weights, 48000.0, nam::LinearImplementation::Direct);
  nam::Linear automatic(1, 1, taps, false, weights, 48000.0);
  for (const double rate : {44100.0, 32000.0, 96000.0, 48000.0})
  {
    const int length = (int)std::ceil(taps * rate / 48000.0);
    direct.Reset(rate, 127);
    automatic.Reset(rate, 127);
    assert(automatic.GetActiveImplementation() == nam::linear::select_implementation(length));
    std::vector<NAM_SAMPLE> input(length + 256, 0.0);
    input[0] = 1.0;
    const auto reference = process_model(direct, input, {127, 1, 13});
    const auto output = process_model(automatic, input, {3, 64, 1, 127});
    for (size_t i = 0; i < output.size(); ++i)
    {
      assert_near(output[i], reference[i], 2.0e-5);
      const double source_position = i * 48000.0 / rate;
      if (source_position >= 1.0 && source_position < taps - 2)
        assert_near(output[i], source_position / taps * 48000.0 / rate, 2.0e-5);
      if (i >= (size_t)length)
        assert_near(output[i], 0.0, 2.0e-5);
    }
  }
}

void test_sample_rate_short_unknown_and_invalid()
{
  for (const auto implementation : {nam::LinearImplementation::Direct, nam::LinearImplementation::FFT})
  {
    nam::Linear short_ir(1, 1, 1, false, {1.0f}, 48000.0, implementation);
    short_ir.Reset(96000.0, 4);
    const auto output = process_model(short_ir, {1.0, 0.0, 0.0, 0.0}, {4});
    assert_near(output[0], 0.5, 1.0e-7);
    assert_near(output[1], 0.28125, 1.0e-7);
    assert_near(output[2], 0.0, 1.0e-7);
    short_ir.Reset(8000.0, 4);
    assert_near(process_model(short_ir, {1.0}, {1})[0], 6.0, 1.0e-7);

    // Without a training rate there is no conversion ratio; keep legacy weights.
    nam::Linear unknown(1, 1, 2, false, {0.5f, 0.25f}, -1.0, implementation);
    unknown.Reset(44100.0, 4);
    const auto unchanged = process_model(unknown, {1.0, 0.0, 0.0}, {3});
    assert_near(unchanged[0], 0.5, 1.0e-7);
    assert_near(unchanged[1], 0.25, 1.0e-7);
    assert(unknown.GetExpectedSampleRate() == -1.0);
    for (const double rate : {0.0, -1.0, std::numeric_limits<double>::infinity(),
                              std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::max()})
    {
      bool threw = false;
      try
      {
        short_ir.Reset(rate, 4);
      }
      catch (const std::exception&)
      {
        threw = true;
      }
      assert(threw);
    }
  }
}

void test_sample_rate_multichannel_realtime_safe()
{
  for (const auto implementation : {nam::LinearImplementation::Direct, nam::LinearImplementation::FFT})
  {
    std::vector<float> weights(2048, 0.0f);
    weights[300] = 1.0f;
    nam::Linear model(2, 3, 2048, false, weights, 48000.0, implementation);
    model.SetPrewarmOnReset(false);
    model.Reset(96000.0, 64);
    std::vector<NAM_SAMPLE> input0(64, 0.0), input1(64, 0.0);
    std::vector<NAM_SAMPLE> output0(64), output1(64), output2(64);
    NAM_SAMPLE* inputs[] = {input0.data(), input1.data()};
    NAM_SAMPLE* outputs[] = {output0.data(), output1.data(), output2.data()};
    allocation_tracking::run_allocation_test_no_allocations(
      nullptr,
      [&]() {
        for (int block = 0; block < 80; ++block)
        {
          input0[0] = block == 0 ? 1.0 : 0.0;
          input1[0] = block == 0 ? 2.0 : 0.0;
          model.process(inputs, outputs, 64);
          for (int i = 0; i < 64; ++i)
          {
            const int position = block * 64 + i;
            double expected = 0.0;
            if (position == 600)
              expected = 0.5;
            if (position == 599 || position == 601)
              expected = 0.28125;
            if (position == 597 || position == 603)
              expected = -0.03125;
            assert_near(output0[i], expected, 1.0e-6);
            assert_near(output1[i], 2.0 * expected, 1.0e-6);
            assert_near(output2[i], 0.0, 1.0e-7);
          }
        }
      },
      nullptr, "Linear resampled first process real-time safe");
    // Populate direct history and start pending FFT work before resetting.
    std::fill(input0.begin(), input0.end(), 1.0);
    std::fill(input1.begin(), input1.end(), 2.0);
    for (int block = 0; block < 4; ++block)
      model.process(inputs, outputs, 64);
    // Same-rate reset must clear both kinds of state without relying on prewarm.
    model.Reset(96000.0, 64);
    std::fill(input0.begin(), input0.end(), 0.0);
    std::fill(input1.begin(), input1.end(), 0.0);
    for (int block = 0; block < 80; ++block)
    {
      model.process(inputs, outputs, 64);
      for (int i = 0; i < 64; ++i)
      {
        assert_near(output0[i], 0.0, 1.0e-6);
        assert_near(output1[i], 0.0, 1.0e-6);
      }
    }
  }
}

} // namespace test_linear
