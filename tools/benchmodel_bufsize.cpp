#include <iostream>
#include <chrono>
#include <filesystem>
#include <cstdlib>

#include "NAM/dsp.h"
#include "NAM/get_dsp.h"

using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::microseconds;

int main(int argc, char* argv[])
{
  if (argc < 3)
  {
    std::cerr << "Usage: benchmodel_bufsize <model_path> <buffer_size> [num_iterations]\n";
    exit(1);
  }

  const char* modelPath = argv[1];
  const int bufferSize = std::atoi(argv[2]);
  const int numIterations = (argc > 3) ? std::atoi(argv[3]) : 5;

  if (bufferSize <= 0 || bufferSize > 4096)
  {
    std::cerr << "Buffer size must be between 1 and 4096\n";
    exit(1);
  }

  // Turn on fast tanh approximation
  nam::activations::Activation::enable_fast_tanh();

  std::unique_ptr<nam::DSP> model;
  model = nam::get_dsp(std::filesystem::path(modelPath));

  if (model == nullptr)
  {
    std::cerr << "Failed to load model\n";
    exit(1);
  }

  model->Reset(model->GetExpectedSampleRate(), bufferSize);

  // Process 2 seconds of audio
  const size_t totalSamples = 48000 * 2;
  const size_t numBuffers = totalSamples / bufferSize;

  // Allocate multi-channel buffers
  const int in_channels = model->NumInputChannels();
  const int out_channels = model->NumOutputChannels();

  std::vector<std::vector<double>> inputBuffers(in_channels);
  std::vector<std::vector<double>> outputBuffers(out_channels);
  std::vector<double*> inputPtrs(in_channels);
  std::vector<double*> outputPtrs(out_channels);

  for (int ch = 0; ch < in_channels; ch++)
  {
    inputBuffers[ch].resize(bufferSize, 0.0);
    inputPtrs[ch] = inputBuffers[ch].data();
  }
  for (int ch = 0; ch < out_channels; ch++)
  {
    outputBuffers[ch].resize(bufferSize, 0.0);
    outputPtrs[ch] = outputBuffers[ch].data();
  }

  // Warm-up run
  for (size_t i = 0; i < numBuffers; i++)
  {
    model->process(inputPtrs.data(), outputPtrs.data(), bufferSize);
  }

  // Timed runs
  double totalMicroseconds = 0.0;
  for (int iter = 0; iter < numIterations; iter++)
  {
    auto t1 = high_resolution_clock::now();
    for (size_t i = 0; i < numBuffers; i++)
    {
      model->process(inputPtrs.data(), outputPtrs.data(), bufferSize);
    }
    auto t2 = high_resolution_clock::now();
    duration<double, std::micro> us_double = t2 - t1;
    totalMicroseconds += us_double.count();
  }

  double avgMicroseconds = totalMicroseconds / numIterations;

  // Output format: buffer_size,avg_microseconds
  std::cout << bufferSize << "," << avgMicroseconds << std::endl;

  return 0;
}
