#pragma once

#include <fstream>

#include "dsp.h"

namespace nam
{
class Version
{
public:
  Version(int major, int minor, int patch)
  : major(major)
  , minor(minor)
  , patch(patch)
  {
  }

  std::string toString() const
  {
    return std::to_string(major) + "." + std::to_string(minor) + "." + std::to_string(patch);
  }

  bool operator>(const Version& other) const
  {
    return major > other.major
           || (major == other.major && (minor > other.minor || (minor == other.minor && patch > other.patch)));
  }

  bool operator<(const Version& other) const
  {
    return major < other.major
           || (major == other.major && (minor < other.minor || (minor == other.minor && patch < other.patch)));
  }

  int major;
  int minor;
  int patch;
};

Version ParseVersion(const std::string& versionStr);

void verify_config_version(const std::string versionStr);

const std::string LATEST_FULLY_SUPPORTED_NAM_FILE_VERSION = "0.6.0";
const std::string EARLIEST_SUPPORTED_NAM_FILE_VERSION = "0.5.0";

/// \brief Get NAM from a .nam file at the provided location
/// \param config_filename Path to the .nam model file
/// \return Unique pointer to a DSP object
std::unique_ptr<DSP> get_dsp(const std::filesystem::path config_filename);

/// \brief Get NAM from a provided configuration struct
/// \param conf DSP data structure containing model configuration and weights
/// \return Unique pointer to a DSP object
std::unique_ptr<DSP> get_dsp(dspData& conf);

/// \brief Get NAM from a .nam file and store its configuration
///
/// Creates an instance of DSP and also returns a dspData struct that holds the data of the model.
/// \param config_filename Path to the .nam model file
/// \param returnedConfig Output parameter that will be filled with the model data
/// \return Unique pointer to a DSP object
std::unique_ptr<DSP> get_dsp(const std::filesystem::path config_filename, dspData& returnedConfig);

/// \brief Get NAM from a provided configuration JSON object
/// \param config JSON configuration object
/// \param returnedConfig Output parameter that will be filled with the model data
/// \return Unique pointer to a DSP object
std::unique_ptr<DSP> get_dsp(const nlohmann::json& config, dspData& returnedConfig);

/// \brief Get NAM from a provided configuration JSON object (convenience overload)
/// \param config JSON configuration object
/// \return Unique pointer to a DSP object
std::unique_ptr<DSP> get_dsp(const nlohmann::json& config);

/// \brief Get sample rate from a .nam file
/// \param j JSON object from the .nam file
/// \return Sample rate in Hz, or -1 if not known (really old .nam files)
double get_sample_rate_from_nam_file(const nlohmann::json& j);
}; // namespace nam
