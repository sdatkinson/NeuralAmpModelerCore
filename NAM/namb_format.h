#pragma once
// Compact binary model format (.namb) for NAM
// Format version 1 - no external dependencies required for reading

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace nam
{
namespace namb
{

// Magic number: "NAMB" as little-endian uint32
static constexpr uint32_t MAGIC = 0x4E414D42;
static constexpr uint16_t FORMAT_VERSION = 1;

// File offsets
static constexpr size_t FILE_HEADER_SIZE = 32;
static constexpr size_t METADATA_BLOCK_SIZE = 48;
static constexpr size_t MODEL_BLOCK_OFFSET = FILE_HEADER_SIZE + METADATA_BLOCK_SIZE; // 80

// Architecture IDs (must match order in binary format spec)
static constexpr uint8_t ARCH_LINEAR = 0;
static constexpr uint8_t ARCH_CONVNET = 1;
static constexpr uint8_t ARCH_LSTM = 2;
static constexpr uint8_t ARCH_WAVENET = 3;

// Metadata flags
static constexpr uint8_t META_HAS_LOUDNESS = 0x01;
static constexpr uint8_t META_HAS_INPUT_LEVEL = 0x02;
static constexpr uint8_t META_HAS_OUTPUT_LEVEL = 0x04;

// GatingMode values (matches wavenet::GatingMode enum)
static constexpr uint8_t GATING_NONE = 0;
static constexpr uint8_t GATING_GATED = 1;
static constexpr uint8_t GATING_BLENDED = 2;

// =============================================================================
// CRC32 (IEEE 802.3 polynomial, same as zlib)
// =============================================================================

inline uint32_t crc32_table(uint8_t byte)
{
  uint32_t crc = byte;
  for (int i = 0; i < 8; i++)
  {
    if (crc & 1)
      crc = (crc >> 1) ^ 0xEDB88320u;
    else
      crc >>= 1;
  }
  return crc;
}

inline uint32_t crc32(const uint8_t* data, size_t size)
{
  uint32_t crc = 0xFFFFFFFFu;
  for (size_t i = 0; i < size; i++)
  {
    crc = crc32_table((uint8_t)(crc ^ data[i])) ^ (crc >> 8);
  }
  return crc ^ 0xFFFFFFFFu;
}

// CRC32 of all bytes except the checksum field (bytes 24..27)
inline uint32_t compute_file_crc32(const uint8_t* data, size_t size)
{
  // Hash bytes 0..23, then 28..end, skipping the checksum field at offset 24
  uint32_t crc = 0xFFFFFFFFu;
  for (size_t i = 0; i < size; i++)
  {
    if (i >= 24 && i < 28)
      continue; // Skip checksum field
    crc = crc32_table((uint8_t)(crc ^ data[i])) ^ (crc >> 8);
  }
  return crc ^ 0xFFFFFFFFu;
}

// =============================================================================
// BinaryReader - reads from a memory buffer with bounds checking
// =============================================================================

class BinaryReader
{
public:
  BinaryReader(const uint8_t* data, size_t size)
  : _data(data)
  , _size(size)
  , _pos(0)
  {
  }

  uint8_t read_u8()
  {
    check(1);
    return _data[_pos++];
  }

  uint16_t read_u16()
  {
    check(2);
    uint16_t v;
    std::memcpy(&v, _data + _pos, 2);
    _pos += 2;
    return v;
  }

  uint32_t read_u32()
  {
    check(4);
    uint32_t v;
    std::memcpy(&v, _data + _pos, 4);
    _pos += 4;
    return v;
  }

  int32_t read_i32()
  {
    check(4);
    int32_t v;
    std::memcpy(&v, _data + _pos, 4);
    _pos += 4;
    return v;
  }

  float read_f32()
  {
    check(4);
    float v;
    std::memcpy(&v, _data + _pos, 4);
    _pos += 4;
    return v;
  }

  double read_f64()
  {
    check(8);
    double v;
    std::memcpy(&v, _data + _pos, 8);
    _pos += 8;
    return v;
  }

  void skip(size_t n)
  {
    check(n);
    _pos += n;
  }

  size_t position() const { return _pos; }
  size_t remaining() const { return _size - _pos; }

  const uint8_t* current_ptr() const { return _data + _pos; }

private:
  void check(size_t n) const
  {
    if (_pos + n > _size)
      throw std::runtime_error("NAMB: unexpected end of data at offset " + std::to_string(_pos));
  }

  const uint8_t* _data;
  size_t _size;
  size_t _pos;
};

// =============================================================================
// BinaryWriter - builds a byte buffer
// =============================================================================

class BinaryWriter
{
public:
  void write_u8(uint8_t v) { _data.push_back(v); }

  void write_u16(uint16_t v)
  {
    size_t pos = _data.size();
    _data.resize(pos + 2);
    std::memcpy(_data.data() + pos, &v, 2);
  }

  void write_u32(uint32_t v)
  {
    size_t pos = _data.size();
    _data.resize(pos + 4);
    std::memcpy(_data.data() + pos, &v, 4);
  }

  void write_i32(int32_t v)
  {
    size_t pos = _data.size();
    _data.resize(pos + 4);
    std::memcpy(_data.data() + pos, &v, 4);
  }

  void write_f32(float v)
  {
    size_t pos = _data.size();
    _data.resize(pos + 4);
    std::memcpy(_data.data() + pos, &v, 4);
  }

  void write_f64(double v)
  {
    size_t pos = _data.size();
    _data.resize(pos + 8);
    std::memcpy(_data.data() + pos, &v, 8);
  }

  void write_zeros(size_t n) { _data.resize(_data.size() + n, 0); }

  // Backpatch a uint32 at a specific offset
  void set_u32(size_t offset, uint32_t v) { std::memcpy(_data.data() + offset, &v, 4); }

  size_t position() const { return _data.size(); }
  const uint8_t* data() const { return _data.data(); }
  uint8_t* data() { return _data.data(); }
  size_t size() const { return _data.size(); }

  const std::vector<uint8_t>& buffer() const { return _data; }

private:
  std::vector<uint8_t> _data;
};

} // namespace namb
} // namespace nam
