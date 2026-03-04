#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

#include "NAM/dsp.h"
#include "NAM/get_dsp.h"
#include "wav.h"

namespace
{
// Write mono 32-bit float WAV file (IEEE float format 3).
bool SaveWavFloat32(const char* fileName, const float* samples, size_t numSamples, double sampleRate)
{
  std::ofstream out(fileName, std::ios::binary);
  if (!out.is_open())
  {
    std::cerr << "Error: Failed to open output file " << fileName << "\n";
    return false;
  }

  const uint32_t dataSize = static_cast<uint32_t>(numSamples * sizeof(float));
  const uint32_t chunkSize = 36 + dataSize;

  // RIFF header
  out.write("RIFF", 4);
  out.write(reinterpret_cast<const char*>(&chunkSize), 4);
  out.write("WAVE", 4);

  // fmt chunk (16 bytes for PCM/IEEE)
  const uint32_t fmtSize = 16;
  out.write("fmt ", 4);
  out.write(reinterpret_cast<const char*>(&fmtSize), 4);
  const uint16_t audioFormat = 3; // IEEE float
  out.write(reinterpret_cast<const char*>(&audioFormat), 2);
  const uint16_t numChannels = 1;
  out.write(reinterpret_cast<const char*>(&numChannels), 2);
  const uint32_t sr = static_cast<uint32_t>(sampleRate);
  out.write(reinterpret_cast<const char*>(&sr), 4);
  const uint32_t byteRate = sr * sizeof(float);
  out.write(reinterpret_cast<const char*>(&byteRate), 4);
  const uint16_t blockAlign = sizeof(float);
  out.write(reinterpret_cast<const char*>(&blockAlign), 2);
  const uint16_t bitsPerSample = 32;
  out.write(reinterpret_cast<const char*>(&bitsPerSample), 2);

  // data chunk
  out.write("data", 4);
  out.write(reinterpret_cast<const char*>(&dataSize), 4);
  out.write(reinterpret_cast<const char*>(samples), dataSize);

  return out.good();
}

} // namespace

int main(int argc, char* argv[])
{
  if (argc < 3 || argc > 4)
  {
    std::cerr << "Usage: render <model.nam> <input.wav> [output.wav]\n";
    return 1;
  }

  const char* modelPath = argv[1];
  const char* inputPath = argv[2];
  const char* outputPath = (argc >= 4) ? argv[3] : "output.wav";

  std::cerr << "Loading model [" << modelPath << "]\n";
  auto model = nam::get_dsp(std::filesystem::path(modelPath));
  if (!model)
  {
    std::cerr << "Failed to load model\n";
    return 1;
  }
  std::cerr << "Model loaded successfully\n";

  std::vector<float> inputAudio;
  double inputSampleRate = 0.0;
  auto loadResult = dsp::wav::Load(inputPath, inputAudio, inputSampleRate);
  if (loadResult != dsp::wav::LoadReturnCode::SUCCESS)
  {
    std::cerr << "Failed to load input WAV: " << dsp::wav::GetMsgForLoadReturnCode(loadResult) << "\n";
    return 1;
  }

  const double expectedRate = model->GetExpectedSampleRate();
  if (expectedRate > 0 && std::abs(inputSampleRate - expectedRate) > 0.5)
  {
    std::cerr << "Error: Input WAV sample rate (" << inputSampleRate << " Hz) does not match model expected rate ("
              << expectedRate << " Hz)\n";
    return 1;
  }

  const double sampleRate = expectedRate > 0 ? expectedRate : inputSampleRate;
  const int bufferSize = 64;
  model->Reset(sampleRate, bufferSize);

  const int inChannels = model->NumInputChannels();
  const int outChannels = model->NumOutputChannels();

  if (inChannels != 1)
  {
    std::cerr << "Error: render tool currently supports mono input only (model has " << inChannels
              << " input channels)\n";
    return 1;
  }

  std::vector<std::vector<NAM_SAMPLE>> inputBuffers(inChannels);
  std::vector<std::vector<NAM_SAMPLE>> outputBuffers(outChannels);
  std::vector<NAM_SAMPLE*> inputPtrs(inChannels);
  std::vector<NAM_SAMPLE*> outputPtrs(outChannels);

  for (int ch = 0; ch < inChannels; ch++)
  {
    inputBuffers[ch].resize(bufferSize, 0.0);
    inputPtrs[ch] = inputBuffers[ch].data();
  }
  for (int ch = 0; ch < outChannels; ch++)
  {
    outputBuffers[ch].resize(bufferSize, 0.0);
    outputPtrs[ch] = outputBuffers[ch].data();
  }

  std::vector<float> outputAudio;
  outputAudio.reserve(static_cast<size_t>(outChannels) * inputAudio.size());

  size_t readPos = 0;
  const size_t totalSamples = inputAudio.size();

  while (readPos < totalSamples)
  {
    const size_t toRead = std::min(static_cast<size_t>(bufferSize), totalSamples - readPos);

    for (size_t i = 0; i < toRead; i++)
      inputBuffers[0][i] = static_cast<NAM_SAMPLE>(inputAudio[readPos + i]);
    for (size_t i = toRead; i < static_cast<size_t>(bufferSize); i++)
      inputBuffers[0][i] = 0;

    model->process(inputPtrs.data(), outputPtrs.data(), static_cast<int>(toRead));

    for (size_t i = 0; i < toRead; i++)
      outputAudio.push_back(static_cast<float>(outputBuffers[0][i]));

    readPos += toRead;
  }

  if (!SaveWavFloat32(outputPath, outputAudio.data(), outputAudio.size(), sampleRate))
  {
    return 1;
  }

  std::cerr << "Wrote " << outputAudio.size() << " samples to " << outputPath << "\n";
  return 0;
}
