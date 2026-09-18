#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <vector>

#include "NAM/get_dsp.h"
#include "NAM/linear.h"

namespace test_get_dsp_wav
{
using Bytes = std::vector<unsigned char>;

void append(Bytes& bytes, uint32_t value, int count)
{
  for (int i = 0; i < count; ++i)
    bytes.push_back(static_cast<unsigned char>(value >> (8 * i)));
}

void chunk(Bytes& bytes, const char* id, const Bytes& payload)
{
  bytes.insert(bytes.end(), id, id + 4);
  append(bytes, static_cast<uint32_t>(payload.size()), 4);
  bytes.insert(bytes.end(), payload.begin(), payload.end());
  if (payload.size() % 2)
    bytes.push_back(0);
}

Bytes wav(int format, int bits, const Bytes& samples, bool extensible = false, int channels = 1)
{
  Bytes fmt;
  append(fmt, extensible ? 65534 : format, 2);
  append(fmt, channels, 2);
  append(fmt, 44100, 4);
  append(fmt, 44100 * channels * (bits / 8), 4);
  append(fmt, channels * (bits / 8), 2);
  append(fmt, bits, 2);
  if (extensible)
  {
    append(fmt, 22, 2);
    append(fmt, bits, 2);
    append(fmt, 0, 4);
    append(fmt, format, 4);
    const Bytes guidTail{0, 0, 0x10, 0, 0x80, 0, 0, 0xaa, 0, 0x38, 0x9b, 0x71};
    fmt.insert(fmt.end(), guidTail.begin(), guidTail.end());
  }
  Bytes body{'W', 'A', 'V', 'E'};
  chunk(body, "fmt ", fmt);
  chunk(body, "JUNK", {42}); // Odd-sized unknown chunks require padding.
  chunk(body, "data", samples);
  Bytes result{'R', 'I', 'F', 'F'};
  append(result, static_cast<uint32_t>(body.size()), 4);
  result.insert(result.end(), body.begin(), body.end());
  return result;
}

struct Fixture
{
  std::filesystem::path path =
    std::filesystem::temp_directory_path()
    / ("nam-ir-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".WaV");
  explicit Fixture(const Bytes& bytes)
  {
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    assert(out.good());
  }
  ~Fixture() { std::filesystem::remove(path); }
};

void check_response(nam::DSP& dsp, int channels = 1)
{
  assert(dynamic_cast<nam::Linear*>(&dsp));
  assert(dsp.NumInputChannels() == 1 && dsp.NumOutputChannels() == channels);
  assert(dsp.GetExpectedSampleRate() == 44100.0);
  assert(!dsp.HasLoudness() && !dsp.HasInputLevel() && !dsp.HasOutputLevel());
  dsp.Reset(44100.0, 2);
  NAM_SAMPLE input[]{1, 0}, output[2]{}, right[2]{};
  NAM_SAMPLE* inputs[]{input};
  NAM_SAMPLE* outputs[]{output, right};
  dsp.process(inputs, outputs, 2);
  assert(std::abs(output[0] - 0.5) < 1e-6);
  assert(std::abs(output[1] + 0.25) < 1e-6);
  if (channels == 2)
  {
    assert(std::abs(right[0] + 0.5) < 1e-6);
    assert(std::abs(right[1] - 0.25) < 1e-6);
  }
  input[0] = 0;
  dsp.process(inputs, outputs, 2);
  assert(std::abs(output[0] - 0.125) < 1e-6);
  assert(std::abs(output[1]) < 1e-6);
  if (channels == 2)
  {
    assert(std::abs(right[0] + 0.125) < 1e-6);
    assert(std::abs(right[1]) < 1e-6);
  }
}

void test_formats_and_configuration()
{
  for (const int format : {1, 3})
    for (const int bits : {16, 24, 32})
      for (const int channels : {1, 2})
        for (const bool extensible : {false, true})
        {
          if (format == 3 && bits != 32)
            continue;
          Bytes samples;
          if (format == 3)
            for (uint32_t value : {0x3f000000u, 0xbe800000u, 0x3e000000u})
            {
              append(samples, value, 4);
              if (channels == 2)
                append(samples, value ^ 0x80000000u, 4);
            }
          else
            for (int32_t value : {int32_t(1 << (bits - 2)), -int32_t(1 << (bits - 3)), int32_t(1 << (bits - 4))})
            {
              append(samples, static_cast<uint32_t>(value), bits / 8);
              if (channels == 2)
                append(samples, static_cast<uint32_t>(-value), bits / 8);
            }
          Fixture fixture(wav(format, bits, samples, extensible, channels));
          nam::dspData config;
          nam::DspLoadOptions options;
          options.prewarm = false;
          auto dsp = nam::get_dsp(fixture.path, config, options);
          check_response(*dsp, channels);
          assert(config.architecture == "Linear");
          assert(config.version == nam::LATEST_FULLY_SUPPORTED_NAM_FILE_VERSION);
          assert(config.config.at("receptive_field") == 3);
          assert(config.config.at("bias") == false);
          assert(config.expected_sample_rate == 44100.0);
          assert(config.metadata.is_null());
          assert(config.config.value("in_channels", 1) == 1);
          assert(config.config.value("out_channels", 1) == channels);
          std::vector<float> expected{0.5f, -0.25f, 0.125f};
          if (channels == 2)
            expected.insert(expected.end(), {-0.5f, 0.25f, -0.125f});
          assert(config.weights == expected);
          auto reloaded = nam::get_dsp(config);
          check_response(*reloaded, channels);
          nam::ScopedPrewarmOnResetDefault scoped(false);
          auto convenience = nam::get_dsp(fixture.path);
          assert(!convenience->GetPrewarmOnReset());
          check_response(*convenience, channels);
        }
}

void expect_invalid(const Bytes& bytes)
{
  Fixture fixture(bytes);
  bool threw = false;
  try
  {
    auto dsp = nam::get_dsp(fixture.path);
  }
  catch (const std::runtime_error&)
  {
    threw = true;
  }
  assert(threw);
}

void test_invalid_files()
{
  const auto valid = wav(1, 16, {0, 64, 0, 0});
  // Every truncated prefix must fail without reading outside the file.
  for (size_t n = 0; n < valid.size(); ++n)
    expect_invalid(Bytes(valid.begin(), valid.begin() + n));
  expect_invalid(wav(1, 16, {}));
  expect_invalid(wav(1, 16, {1}));
  expect_invalid(wav(6, 16, {0, 0}));
  expect_invalid(wav(1, 8, {0}));
  expect_invalid(wav(3, 32, {0, 0, 0x80, 0x7f})); // Infinity
  for (const size_t offset : {size_t(0), size_t(8), size_t(20), size_t(22), size_t(24), size_t(28), size_t(32)})
  {
    auto invalid = valid;
    invalid[offset] = 0;
    expect_invalid(invalid);
  }
  expect_invalid(wav(1, 16, {0, 0, 0, 0, 0, 0}, false, 3));
  expect_invalid(wav(1, 16, {0, 0}, false, 2)); // Partial stereo frame
  expect_invalid(wav(1, 16, {}, false, 2));
  auto stereo = valid; // Stereo header with inconsistent mono alignment/byte rate
  stereo[22] = 2;
  expect_invalid(stereo);
  auto badGuid = wav(1, 16, {0, 0}, true);
  badGuid[59] = 0;
  expect_invalid(badGuid);
  Fixture removed(valid);
  std::filesystem::remove(removed.path);
  bool threw = false;
  try
  {
    auto dsp = nam::get_dsp(removed.path);
  }
  catch (const std::runtime_error&)
  {
    threw = true;
  }
  assert(threw);
}
} // namespace test_get_dsp_wav
