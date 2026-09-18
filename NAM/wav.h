#pragma once

#include <filesystem>
#include <vector>

namespace nam
{
namespace detail
{
// Decode a mono RIFF/WAVE impulse response. Throws std::runtime_error on invalid or unsupported files.
std::vector<float> load_wav_ir(const std::filesystem::path& filename, double& sample_rate);
} // namespace detail
} // namespace nam
