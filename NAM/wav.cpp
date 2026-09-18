#include "wav.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>

namespace nam
{
namespace detail
{
std::vector<float> load_wav_ir(const std::filesystem::path& filename, double& sample_rate)
{
  // Format and scaling follow AudioDSPTools/dsp/wav.cpp. Read little-endian fields explicitly and
  // check chunk bounds before allocation or decoding, including odd-byte RIFF padding.
  const auto fail = [&filename](const char* reason) {
    throw std::runtime_error("Could not load WAV impulse response [" + filename.string() + "]: " + reason);
  };
  std::ifstream file(filename, std::ios::binary | std::ios::ate);
  if (!file)
    fail("cannot open file");
  const auto file_size = file.tellg();
  if (file_size < 12)
    fail("missing RIFF/WAVE header");
  file.seekg(0);
  const auto read = [&file, &fail](unsigned char* bytes, size_t count) {
    if (!file.read(reinterpret_cast<char*>(bytes), static_cast<std::streamsize>(count)))
      fail("truncated file");
  };
  const auto uint_le = [](const unsigned char* bytes, int count) {
    uint32_t value = 0;
    for (int i = 0; i < count; ++i)
      value |= uint32_t(bytes[i]) << (8 * i);
    return value;
  };
  std::array<unsigned char, 12> header{};
  read(header.data(), header.size());
  if (std::memcmp(header.data(), "RIFF", 4) || std::memcmp(header.data() + 8, "WAVE", 4))
    fail("expected RIFF/WAVE");
  const uint64_t end = uint64_t(uint_le(header.data() + 4, 4)) + 8;
  if (end < 12 || end > static_cast<uint64_t>(file_size))
    fail("invalid RIFF size");

  bool have_format = false, have_data = false;
  uint32_t format = 0, bits = 0, rate = 0, alignment = 0, data_size = 0;
  uint64_t data_position = 0;
  for (uint64_t position = 12; position < end;)
  {
    if (end - position < 8)
      fail("incomplete chunk header");
    file.seekg(static_cast<std::streamoff>(position));
    std::array<unsigned char, 8> chunk{};
    read(chunk.data(), chunk.size());
    const uint32_t size = uint_le(chunk.data() + 4, 4);
    const uint64_t padded_size = uint64_t(size) + (size % 2);
    position += 8;
    if (padded_size > end - position)
      fail("chunk exceeds RIFF bounds");
    if (std::memcmp(chunk.data(), "fmt ", 4) == 0)
    {
      if (have_format || size < 16)
        fail("invalid or duplicate format chunk");
      std::array<unsigned char, 40> fmt{};
      read(fmt.data(), std::min(size_t(size), fmt.size()));
      format = uint_le(fmt.data(), 2);
      if (uint_le(fmt.data() + 2, 2) != 1)
        fail("only mono impulse responses are supported");
      rate = uint_le(fmt.data() + 4, 4);
      const uint32_t byte_rate = uint_le(fmt.data() + 8, 4);
      alignment = uint_le(fmt.data() + 12, 2);
      bits = uint_le(fmt.data() + 14, 2);
      if (format == 65534)
      {
        if (size < 40 || uint_le(fmt.data() + 16, 2) < 22 || uint64_t(uint_le(fmt.data() + 16, 2)) + 18 > size)
          fail("invalid extensible format");
        const uint32_t valid_bits = uint_le(fmt.data() + 18, 2);
        if (valid_bits == 0 || valid_bits > bits)
          fail("invalid valid-bits field");
        // KSDATAFORMAT_SUBTYPE_PCM / IEEE_FLOAT GUID tail.
        const unsigned char guid_tail[]{0, 0, 0x10, 0, 0x80, 0, 0, 0xaa, 0, 0x38, 0x9b, 0x71};
        if (std::memcmp(fmt.data() + 28, guid_tail, sizeof(guid_tail)))
          fail("unsupported extensible subtype");
        format = uint_le(fmt.data() + 24, 4);
        if (format == 3 && valid_bits != bits)
          fail("invalid floating-point valid-bits field");
      }
      if (!((format == 1 && (bits == 16 || bits == 24 || bits == 32)) || (format == 3 && bits == 32)))
        fail("supported formats are PCM 16/24/32-bit and IEEE float 32-bit");
      if (rate == 0 || alignment != bits / 8 || uint64_t(rate) * alignment != byte_rate)
        fail("invalid sample rate or block alignment");
      have_format = true;
    }
    else if (std::memcmp(chunk.data(), "data", 4) == 0)
    {
      if (!have_format || have_data)
        fail("missing format or duplicate data chunk");
      data_position = position;
      data_size = size;
      have_data = true;
    }
    position += padded_size;
  }
  if (!have_data || data_size == 0 || data_size % alignment != 0)
    fail("missing, empty, or incomplete sample data");
  const size_t count = data_size / alignment;
  if (count > static_cast<size_t>(std::numeric_limits<int>::max()))
    fail("impulse response is too long");
  std::vector<float> samples(count);
  file.seekg(static_cast<std::streamoff>(data_position));
  for (auto& sample : samples)
  {
    unsigned char bytes[4]{};
    read(bytes, alignment);
    const uint32_t raw = uint_le(bytes, alignment);
    if (format == 3)
    {
      static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559, "WAV requires IEEE float32");
      std::memcpy(&sample, &raw, sizeof(sample));
      if (!std::isfinite(sample))
        fail("non-finite sample");
    }
    else
    {
      // PCM is signed, with valid bits left-aligned in extensible containers.
      const int64_t signed_sample =
        (raw & (uint32_t(1) << (bits - 1))) ? int64_t(raw) - (int64_t(1) << bits) : int64_t(raw);
      sample = static_cast<float>(signed_sample / double(uint64_t(1) << (bits - 1)));
    }
  }
  sample_rate = rate;
  return samples;
}
} // namespace detail
} // namespace nam
