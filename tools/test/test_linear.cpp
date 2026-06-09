// Tests for Linear DSP models

#include "NAM/dsp.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <stdexcept>
#include <vector>

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

  const auto cutoff_weights = make_weights(256, false);
  nam::Linear cutoff_model(1, 1, 256, false, cutoff_weights, 48000.0);
  assert(cutoff_model.GetRequestedImplementation() == nam::LinearImplementation::Auto);
  assert(cutoff_model.GetActiveImplementation() == nam::LinearImplementation::Direct);

  const auto fft_weights = make_weights(512, false);
  nam::Linear fft_model(1, 1, 512, false, fft_weights, 48000.0);
  assert(fft_model.GetRequestedImplementation() == nam::LinearImplementation::Auto);
  assert(fft_model.GetActiveImplementation() == nam::LinearImplementation::FFT);
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

} // namespace test_linear
