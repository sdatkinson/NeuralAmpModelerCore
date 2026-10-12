#pragma once

#include <vector>
#include <version>

#if defined(__cpp_lib_filesystem)
  #include <filesystem>
#endif

#if defined(__cpp_lib_filesystem)
namespace nam
{
namespace detail
{
// Decode a mono or stereo RIFF/WAVE impulse response into channel-major weights. Throws std::runtime_error on invalid
// or unsupported files.
std::vector<float> load_wav_ir(const std::filesystem::path& filename, double& sample_rate, int& out_channels);
} // namespace detail
} // namespace nam
#endif
