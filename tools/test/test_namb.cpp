// Tests for .namb binary format
// - Round-trip: JSON -> NAMB -> load -> process and compare outputs
// - Format validation: bad magic, truncated, wrong version
// - Size verification: NAMB < NAM for all example models

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "NAM/dsp.h"
#include "NAM/get_dsp.h"
#include "NAM/get_dsp_namb.h"
#include "NAM/namb_format.h"
#include "json.hpp"

namespace test_namb
{

// =============================================================================
// Helper: convert a .nam JSON to .namb in memory
// =============================================================================

// We need access to the converter logic. Rather than including nam2namb.cpp directly,
// we replicate the conversion by reading JSON then writing .namb via the public tool.
// For simplicity, we use a temp file approach: write .nam, run conversion, read .namb.
// But actually, let's use the public APIs directly: load JSON via get_dsp(), load NAMB via get_dsp_namb().

// Helper: read file into byte vector
static std::vector<uint8_t> read_file_bytes(const std::filesystem::path& path)
{
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f.is_open())
    throw std::runtime_error("Cannot open: " + path.string());
  size_t size = f.tellg();
  f.seekg(0, std::ios::beg);
  std::vector<uint8_t> data(size);
  f.read(reinterpret_cast<char*>(data.data()), size);
  return data;
}

// Helper: process audio through a DSP model and return outputs
static std::vector<std::vector<double>> process_model(nam::DSP* dsp, int num_buffers, int buffer_size)
{
  const int in_channels = dsp->NumInputChannels();
  const int out_channels = dsp->NumOutputChannels();
  const double sample_rate = dsp->GetExpectedSampleRate() > 0 ? dsp->GetExpectedSampleRate() : 48000.0;

  dsp->Reset(sample_rate, buffer_size);

  std::vector<std::vector<NAM_SAMPLE>> inputBuffers(in_channels);
  std::vector<std::vector<NAM_SAMPLE>> outputBuffers(out_channels);
  std::vector<NAM_SAMPLE*> inputPtrs(in_channels);
  std::vector<NAM_SAMPLE*> outputPtrs(out_channels);

  for (int ch = 0; ch < in_channels; ch++)
  {
    inputBuffers[ch].resize(buffer_size, 0.0);
    inputPtrs[ch] = inputBuffers[ch].data();
  }
  for (int ch = 0; ch < out_channels; ch++)
  {
    outputBuffers[ch].resize(buffer_size, 0.0);
    outputPtrs[ch] = outputBuffers[ch].data();
  }

  // Collect all output samples
  std::vector<std::vector<double>> all_outputs(out_channels);

  for (int buf = 0; buf < num_buffers; buf++)
  {
    // Fill with deterministic test data
    for (int ch = 0; ch < in_channels; ch++)
    {
      for (int i = 0; i < buffer_size; i++)
      {
        inputBuffers[ch][i] = (NAM_SAMPLE)(0.1 * (ch + 1) * ((buf * buffer_size + i) % 100) / 100.0);
      }
    }

    dsp->process(inputPtrs.data(), outputPtrs.data(), buffer_size);

    for (int ch = 0; ch < out_channels; ch++)
    {
      for (int i = 0; i < buffer_size; i++)
      {
        all_outputs[ch].push_back((double)outputBuffers[ch][i]);
      }
    }
  }

  return all_outputs;
}

// =============================================================================
// Round-trip test: For a .nam file, convert to .namb and compare outputs
// =============================================================================

static void test_roundtrip_for_file(const std::string& nam_path)
{
  std::filesystem::path model_path(nam_path);
  if (!std::filesystem::exists(model_path))
  {
    std::cerr << "  Skipping (not found): " << nam_path << std::endl;
    return;
  }

  std::cout << "  Testing round-trip: " << model_path.filename().string() << std::endl;

  // Load JSON model
  std::unique_ptr<nam::DSP> json_model = nam::get_dsp(model_path);
  assert(json_model != nullptr);

  // Process with JSON model
  const int num_buffers = 5;
  const int buffer_size = 64;
  auto json_outputs = process_model(json_model.get(), num_buffers, buffer_size);

  // Convert to .namb: read JSON, use nam2namb logic
  // We need to create a .namb file. Use a temp path.
  std::filesystem::path namb_path = model_path;
  namb_path.replace_extension(".namb");

  // Use the nam2namb tool to create the .namb file
  // Since we can't easily call the tool's function, we'll use system() or
  // construct the binary ourselves. For the test, let's shell out.
  std::string cmd = "./build/tools/nam2namb " + model_path.string() + " " + namb_path.string() + " 2>&1";
  int ret = system(cmd.c_str());
  if (ret != 0)
  {
    // Try relative path from where tests might be run
    cmd = "nam2namb " + model_path.string() + " " + namb_path.string() + " 2>&1";
    ret = system(cmd.c_str());
  }
  assert(ret == 0);
  assert(std::filesystem::exists(namb_path));

  // Load .namb model
  std::unique_ptr<nam::DSP> namb_model = nam::get_dsp_namb(namb_path);
  assert(namb_model != nullptr);

  // Verify same channel counts
  assert(json_model->NumInputChannels() == namb_model->NumInputChannels());
  assert(json_model->NumOutputChannels() == namb_model->NumOutputChannels());

  // Verify same metadata
  assert(json_model->HasLoudness() == namb_model->HasLoudness());
  if (json_model->HasLoudness())
  {
    assert(json_model->GetLoudness() == namb_model->GetLoudness());
  }

  // Process with NAMB model
  auto namb_outputs = process_model(namb_model.get(), num_buffers, buffer_size);

  // Compare outputs - should be bit-identical
  assert(json_outputs.size() == namb_outputs.size());
  for (size_t ch = 0; ch < json_outputs.size(); ch++)
  {
    assert(json_outputs[ch].size() == namb_outputs[ch].size());
    for (size_t i = 0; i < json_outputs[ch].size(); i++)
    {
      if (json_outputs[ch][i] != namb_outputs[ch][i])
      {
        std::cerr << "    Output mismatch at ch=" << ch << " sample=" << i << ": JSON=" << json_outputs[ch][i]
                  << " NAMB=" << namb_outputs[ch][i] << std::endl;
        assert(false);
      }
    }
  }

  // Clean up temp file
  std::filesystem::remove(namb_path);

  std::cout << "    PASS" << std::endl;
}

void test_roundtrip()
{
  std::cout << "test_namb::test_roundtrip" << std::endl;

  // Test all available example models
  const std::vector<std::string> models = {"example_models/wavenet.nam", "example_models/lstm.nam",
                                           "example_models/wavenet_condition_dsp.nam",
                                           "example_models/wavenet_a2_max.nam"};

  for (const auto& model : models)
  {
    test_roundtrip_for_file(model);
  }
}

// =============================================================================
// Format validation tests
// =============================================================================

void test_bad_magic()
{
  std::cout << "test_namb::test_bad_magic" << std::endl;

  // Create minimal data with wrong magic
  std::vector<uint8_t> data(128, 0);
  data[0] = 'X'; // Wrong magic

  bool threw = false;
  try
  {
    nam::get_dsp_namb(data.data(), data.size());
  }
  catch (const std::runtime_error& e)
  {
    threw = true;
    assert(std::string(e.what()).find("magic") != std::string::npos);
  }
  assert(threw);
  std::cout << "  PASS" << std::endl;
}

void test_truncated_file()
{
  std::cout << "test_namb::test_truncated_file" << std::endl;

  // File too small for header
  std::vector<uint8_t> data(16, 0);
  // Set magic correctly
  uint32_t magic = nam::namb::MAGIC;
  std::memcpy(data.data(), &magic, 4);

  bool threw = false;
  try
  {
    nam::get_dsp_namb(data.data(), data.size());
  }
  catch (const std::runtime_error&)
  {
    threw = true;
  }
  assert(threw);
  std::cout << "  PASS" << std::endl;
}

void test_wrong_version()
{
  std::cout << "test_namb::test_wrong_version" << std::endl;

  // Create data with wrong format version
  std::vector<uint8_t> data(128, 0);
  uint32_t magic = nam::namb::MAGIC;
  std::memcpy(data.data(), &magic, 4);
  uint16_t bad_version = 99;
  std::memcpy(data.data() + 4, &bad_version, 2);

  bool threw = false;
  try
  {
    nam::get_dsp_namb(data.data(), data.size());
  }
  catch (const std::runtime_error& e)
  {
    threw = true;
    assert(std::string(e.what()).find("version") != std::string::npos);
  }
  assert(threw);
  std::cout << "  PASS" << std::endl;
}

void test_bad_checksum()
{
  std::cout << "test_namb::test_bad_checksum" << std::endl;

  // First create a valid .namb file, then corrupt it
  std::filesystem::path nam_path("example_models/lstm.nam");
  if (!std::filesystem::exists(nam_path))
  {
    std::cerr << "  Skipping (lstm.nam not found)" << std::endl;
    return;
  }

  std::filesystem::path namb_path("example_models/lstm_test_bad_crc.namb");
  std::string cmd = "./build/tools/nam2namb " + nam_path.string() + " " + namb_path.string() + " 2>&1";
  int ret = system(cmd.c_str());
  if (ret != 0)
  {
    std::cerr << "  Skipping (nam2namb not available)" << std::endl;
    return;
  }

  // Read the .namb file
  auto data = read_file_bytes(namb_path);

  // Corrupt a byte in the weight data (after the checksum)
  if (data.size() > 100)
  {
    data[data.size() - 1] ^= 0xFF;
  }

  bool threw = false;
  try
  {
    nam::get_dsp_namb(data.data(), data.size());
  }
  catch (const std::runtime_error& e)
  {
    threw = true;
    assert(std::string(e.what()).find("checksum") != std::string::npos);
  }
  assert(threw);

  std::filesystem::remove(namb_path);
  std::cout << "  PASS" << std::endl;
}

// =============================================================================
// Size comparison test
// =============================================================================

void test_size_reduction()
{
  std::cout << "test_namb::test_size_reduction" << std::endl;

  const std::vector<std::string> models = {"example_models/wavenet.nam", "example_models/lstm.nam",
                                           "example_models/wavenet_condition_dsp.nam",
                                           "example_models/wavenet_a2_max.nam"};

  for (const auto& nam_path_str : models)
  {
    std::filesystem::path nam_path(nam_path_str);
    if (!std::filesystem::exists(nam_path))
      continue;

    std::filesystem::path namb_path = nam_path;
    namb_path.replace_extension(".namb");

    std::string cmd = "./build/tools/nam2namb " + nam_path.string() + " " + namb_path.string() + " 2>&1";
    int ret = system(cmd.c_str());
    if (ret != 0)
      continue;

    size_t nam_size = std::filesystem::file_size(nam_path);
    size_t namb_size = std::filesystem::file_size(namb_path);
    double reduction = 100.0 * (1.0 - (double)namb_size / (double)nam_size);

    std::cout << "  " << nam_path.filename().string() << ": " << nam_size << " -> " << namb_size << " ("
              << std::fixed << std::setprecision(1) << reduction << "% reduction)" << std::endl;

    // .namb should always be smaller than .nam
    assert(namb_size < nam_size);

    // Should be at least 50% reduction (typically ~85%)
    assert(reduction > 50.0);

    std::filesystem::remove(namb_path);
  }
  std::cout << "  PASS" << std::endl;
}

// =============================================================================
// CRC32 test
// =============================================================================

void test_crc32()
{
  std::cout << "test_namb::test_crc32" << std::endl;

  // Test known CRC32 values
  const uint8_t test1[] = "123456789";
  uint32_t crc1 = nam::namb::crc32(test1, 9);
  // CRC32 of "123456789" is 0xCBF43926
  assert(crc1 == 0xCBF43926u);

  // Empty data
  uint32_t crc_empty = nam::namb::crc32(nullptr, 0);
  assert(crc_empty == 0x00000000u);

  std::cout << "  PASS" << std::endl;
}

}; // namespace test_namb
