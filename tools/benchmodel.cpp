#include <iostream>
#include <chrono>
#include <cstring>
#include <filesystem>

#include "NAM/dsp.h"
#include "NAM/get_dsp.h"

using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;

#define AUDIO_BUFFER_SIZE 64

double inputBuffer[AUDIO_BUFFER_SIZE];
double outputBuffer[AUDIO_BUFFER_SIZE];

int main(int argc, char* argv[])
{
  if (argc < 2)
  {
    std::cerr << "Usage: benchmodel <model_path> [--no-fast-tanh]\n";
    exit(1);
  }

  const char* modelPath = argv[1];

  // Check for --no-fast-tanh flag
  bool useFastTanh = true;
  for (int i = 2; i < argc; i++)
  {
    if (std::strcmp(argv[i], "--no-fast-tanh") == 0)
      useFastTanh = false;
  }

  if (useFastTanh)
  {
    nam::activations::Activation::enable_fast_tanh();
    std::cout << "Fast tanh: enabled\n";
  }
  else
  {
    nam::activations::Activation::disable_fast_tanh();
    std::cout << "Fast tanh: disabled\n";
  }

  std::cout << "Loading model " << modelPath << "\n";

  std::unique_ptr<nam::DSP> model;
  model = nam::get_dsp(std::filesystem::path(modelPath));

  if (model == nullptr)
  {
    std::cerr << "Failed to load model\n";
    exit(1);
  }

  size_t bufferSize = AUDIO_BUFFER_SIZE;
  model->Reset(model->GetExpectedSampleRate(), bufferSize);
  size_t numBuffers = (48000 / bufferSize) * 2;

  // Allocate multi-channel buffers
  const int in_channels = model->NumInputChannels();
  const int out_channels = model->NumOutputChannels();

  std::vector<std::vector<double>> inputBuffers(in_channels);
  std::vector<std::vector<double>> outputBuffers(out_channels);
  std::vector<double*> inputPtrs(in_channels);
  std::vector<double*> outputPtrs(out_channels);

  for (int ch = 0; ch < in_channels; ch++)
  {
    inputBuffers[ch].resize(AUDIO_BUFFER_SIZE, 0.0);
    inputPtrs[ch] = inputBuffers[ch].data();
  }
  for (int ch = 0; ch < out_channels; ch++)
  {
    outputBuffers[ch].resize(AUDIO_BUFFER_SIZE, 0.0);
    outputPtrs[ch] = outputBuffers[ch].data();
  }

  std::cout << "Running benchmark\n";
  auto t1 = high_resolution_clock::now();
  for (size_t i = 0; i < numBuffers; i++)
  {
    model->process(inputPtrs.data(), outputPtrs.data(), AUDIO_BUFFER_SIZE);
  }
  auto t2 = high_resolution_clock::now();
  std::cout << "Finished\n";

  /* Getting number of milliseconds as an integer. */
  auto ms_int = duration_cast<milliseconds>(t2 - t1);

  /* Getting number of milliseconds as a double. */
  duration<double, std::milli> ms_double = t2 - t1;

  std::cout << ms_int.count() << "ms\n";
  std::cout << ms_double.count() << "ms\n";

  return 0;
}
