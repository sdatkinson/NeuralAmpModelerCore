#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "NAM/dsp.h"

namespace
{
using Clock = std::chrono::high_resolution_clock;

struct Options
{
  int sample_rate = 48000;
  double input_seconds = 10.0;
  int buffer_size = 64;
  int receptive_field = 0;
  double receptive_field_seconds = 2.0;
  bool sweep = false;
};

struct RunResult
{
  double seconds = 0.0;
  std::vector<NAM_SAMPLE> output;
  nam::LinearImplementation active_implementation = nam::LinearImplementation::Direct;
};

int parse_int(const char* value, const std::string& name)
{
  char* end = nullptr;
  const long parsed = std::strtol(value, &end, 10);
  if (end == value || *end != '\0' || parsed <= 0)
    throw std::runtime_error(name + " must be a positive integer");
  return (int)parsed;
}

double parse_double(const char* value, const std::string& name)
{
  char* end = nullptr;
  const double parsed = std::strtod(value, &end);
  if (end == value || *end != '\0' || parsed <= 0.0)
    throw std::runtime_error(name + " must be a positive number");
  return parsed;
}

Options parse_options(int argc, char* argv[])
{
  Options options;
  for (int i = 1; i < argc; i++)
  {
    const std::string arg(argv[i]);
    auto require_value = [&](const std::string& name) {
      if (i + 1 >= argc)
        throw std::runtime_error(name + " requires a value");
      return argv[++i];
    };

    if (arg == "--sample-rate")
      options.sample_rate = parse_int(require_value(arg), arg);
    else if (arg == "--input-seconds")
      options.input_seconds = parse_double(require_value(arg), arg);
    else if (arg == "--rf")
      options.receptive_field = parse_int(require_value(arg), arg);
    else if (arg == "--rf-seconds")
      options.receptive_field_seconds = parse_double(require_value(arg), arg);
    else if (arg == "--buffer-size")
      options.buffer_size = parse_int(require_value(arg), arg);
    else if (arg == "--sweep")
      options.sweep = true;
    else
      throw std::runtime_error("Unknown argument: " + arg);
  }
  if (options.receptive_field == 0)
    options.receptive_field = (int)std::llround(options.receptive_field_seconds * options.sample_rate);
  return options;
}

std::vector<float> make_weights(const int receptive_field)
{
  std::vector<float> weights;
  weights.reserve(receptive_field);
  for (int i = 0; i < receptive_field; i++)
  {
    const double envelope = std::exp(-5.0 * (double)i / std::max(1, receptive_field));
    const double modulated = std::sin(0.019 * (i + 1)) + 0.35 * std::cos(0.071 * (i + 1));
    weights.push_back((float)(0.01 * envelope * modulated));
  }
  return weights;
}

std::vector<NAM_SAMPLE> make_input(const int num_samples)
{
  std::vector<NAM_SAMPLE> input(num_samples);
  for (int i = 0; i < num_samples; i++)
  {
    const double sample = 0.2 * std::sin(0.011 * i) + 0.1 * std::sin(0.037 * i) + 0.03 * std::cos(0.101 * i);
    input[i] = (NAM_SAMPLE)sample;
  }
  return input;
}

RunResult run_model(const std::vector<float>& weights, const std::vector<NAM_SAMPLE>& input, const Options& options,
                    const nam::LinearImplementation implementation)
{
  nam::Linear model(1, 1, (int)weights.size(), false, weights, (double)options.sample_rate, implementation);
  model.Reset((double)options.sample_rate, options.buffer_size);

  RunResult result;
  result.output.assign(input.size(), (NAM_SAMPLE)0.0);
  result.active_implementation = model.GetActiveImplementation();

  NAM_SAMPLE* input_ptrs[1];
  NAM_SAMPLE* output_ptrs[1];

  const auto start = Clock::now();
  for (size_t offset = 0; offset < input.size(); offset += options.buffer_size)
  {
    const int count = std::min<int>(options.buffer_size, (int)(input.size() - offset));
    input_ptrs[0] = const_cast<NAM_SAMPLE*>(&input[offset]);
    output_ptrs[0] = &result.output[offset];
    model.process(input_ptrs, output_ptrs, count);
  }
  const auto end = Clock::now();
  result.seconds = std::chrono::duration<double>(end - start).count();

  return result;
}

NAM_SAMPLE max_abs_diff(const std::vector<NAM_SAMPLE>& a, const std::vector<NAM_SAMPLE>& b)
{
  NAM_SAMPLE result = 0.0;
  for (size_t i = 0; i < a.size(); i++)
    result = std::max<NAM_SAMPLE>(result, std::abs(a[i] - b[i]));
  return result;
}

void print_result(const int receptive_field, const std::string& requested, const RunResult& result,
                  const double input_seconds, const NAM_SAMPLE max_diff)
{
  const double rtf = result.seconds / input_seconds;
  std::cout << receptive_field << "," << requested << ","
            << nam::linear::implementation_to_string(result.active_implementation) << "," << result.seconds << ","
            << rtf << "," << max_diff << "\n";
}

void run_case(const Options& options)
{
  const auto weights = make_weights(options.receptive_field);
  const auto input = make_input((int)std::llround(options.input_seconds * options.sample_rate));

  const auto direct = run_model(weights, input, options, nam::LinearImplementation::Direct);
  const auto fft = run_model(weights, input, options, nam::LinearImplementation::FFT);
  const auto automatic = run_model(weights, input, options, nam::LinearImplementation::Auto);

  const auto fft_diff = max_abs_diff(direct.output, fft.output);
  const auto auto_diff = max_abs_diff(direct.output, automatic.output);

  print_result(options.receptive_field, "direct", direct, options.input_seconds, 0.0);
  print_result(options.receptive_field, "fft", fft, options.input_seconds, fft_diff);
  print_result(options.receptive_field, "auto", automatic, options.input_seconds, auto_diff);
}

void run_sweep(Options options)
{
  const std::vector<int> receptive_fields{64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384};
  for (const int receptive_field : receptive_fields)
  {
    options.receptive_field = receptive_field;
    run_case(options);
  }
}

} // namespace

int main(int argc, char* argv[])
{
  try
  {
    const auto options = parse_options(argc, argv);
    std::cout << std::setprecision(10);
    std::cout << "receptive_field,requested,active,seconds,rtf,max_abs_diff_vs_direct\n";
    if (options.sweep)
      run_sweep(options);
    else
      run_case(options);
  }
  catch (const std::exception& e)
  {
    std::cerr << "Error: " << e.what() << "\n";
    std::cerr << "Usage: bench_linear [--sample-rate N] [--input-seconds S] [--rf N] [--rf-seconds S]"
                 " [--buffer-size N] [--sweep]\n";
    return 1;
  }
  return 0;
}
