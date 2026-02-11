#pragma once
// Binary .namb loader for NAM models
// No dependency on nlohmann/json - suitable for embedded targets

#include <cstdint>
#include <filesystem>
#include <memory>

#include "dsp.h"

namespace nam
{

/// \brief Load a NAM model from a .namb binary file
/// \param filename Path to the .namb file
/// \return Unique pointer to a DSP object
/// \throws std::runtime_error on format errors
std::unique_ptr<DSP> get_dsp_namb(const std::filesystem::path& filename);

/// \brief Load a NAM model from a memory buffer containing .namb data
/// \param data Pointer to the binary data
/// \param size Size of the data in bytes
/// \return Unique pointer to a DSP object
/// \throws std::runtime_error on format errors
std::unique_ptr<DSP> get_dsp_namb(const uint8_t* data, size_t size);

} // namespace nam
