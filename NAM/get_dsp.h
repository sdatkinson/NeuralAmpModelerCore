#pragma once

#include <fstream>
#include <memory>
#include <optional>
#include <vector>

#include "dsp.h"

namespace nam
{
enum class Supported
{
  NO = 0,
  PARTIAL = 1,
  YES = 2
};

class IVersionSupportChecker
{
public:
  virtual ~IVersionSupportChecker() = default;
  virtual Supported support(const std::string& version) const = 0;
};

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

void register_version_support_checker(std::shared_ptr<const IVersionSupportChecker> checker);

Supported is_version_supported(const std::string version);

void verify_config_version(const std::string versionStr);

const std::string LATEST_FULLY_SUPPORTED_NAM_FILE_VERSION = "0.7.0";
const std::string EARLIEST_SUPPORTED_NAM_FILE_VERSION = "0.5.0";

/// \brief Get NAM from a .nam file at the provided location
/// \param config_filename Path to the .nam model file
/// \param expectedSampleRate Expected sample rate to configure the model with; std::nullopt uses the file default
/// \param maxBufferSize Maximum buffer size to configure the model with; std::nullopt uses NAM_DEFAULT_MAX_BUFFER_SIZE
/// \param slimmableSize Slimmable size to configure the model with; std::nullopt uses the model default
/// \return Unique pointer to a DSP object
std::unique_ptr<DSP> get_dsp(const std::filesystem::path config_filename,
                             std::optional<double> expectedSampleRate = std::nullopt,
                             std::optional<int> maxBufferSize = std::nullopt,
                             std::optional<double> slimmableSize = std::nullopt);

/// \brief Get NAM from a provided configuration struct
/// \param conf DSP data structure containing model configuration and weights
/// \param expectedSampleRate Expected sample rate to configure the model with; std::nullopt uses the config default
/// \param maxBufferSize Maximum buffer size to configure the model with; std::nullopt uses NAM_DEFAULT_MAX_BUFFER_SIZE
/// \param slimmableSize Slimmable size to configure the model with; std::nullopt uses the model default
/// \return Unique pointer to a DSP object
std::unique_ptr<DSP> get_dsp(dspData& conf, std::optional<double> expectedSampleRate = std::nullopt,
                             std::optional<int> maxBufferSize = std::nullopt,
                             std::optional<double> slimmableSize = std::nullopt);

/// \brief Get NAM from a .nam file and store its configuration
///
/// Creates an instance of DSP and also returns a dspData struct that holds the data of the model.
/// \param config_filename Path to the .nam model file
/// \param returnedConfig Output parameter that will be filled with the model data
/// \param expectedSampleRate Expected sample rate to configure the model with; std::nullopt uses the file default
/// \param maxBufferSize Maximum buffer size to configure the model with; std::nullopt uses NAM_DEFAULT_MAX_BUFFER_SIZE
/// \param slimmableSize Slimmable size to configure the model with; std::nullopt uses the model default
/// \return Unique pointer to a DSP object
std::unique_ptr<DSP> get_dsp(const std::filesystem::path config_filename, dspData& returnedConfig,
                             std::optional<double> expectedSampleRate = std::nullopt,
                             std::optional<int> maxBufferSize = std::nullopt,
                             std::optional<double> slimmableSize = std::nullopt);

/// \brief Get NAM from a provided configuration JSON object
/// \param config JSON configuration object
/// \param returnedConfig Output parameter that will be filled with the model data
/// \param expectedSampleRate Expected sample rate to configure the model with; std::nullopt uses the file default
/// \param maxBufferSize Maximum buffer size to configure the model with; std::nullopt uses NAM_DEFAULT_MAX_BUFFER_SIZE
/// \param slimmableSize Slimmable size to configure the model with; std::nullopt uses the model default
/// \return Unique pointer to a DSP object
std::unique_ptr<DSP> get_dsp(const nlohmann::json& config, dspData& returnedConfig,
                             std::optional<double> expectedSampleRate = std::nullopt,
                             std::optional<int> maxBufferSize = std::nullopt,
                             std::optional<double> slimmableSize = std::nullopt);

/// \brief Get NAM from a provided configuration JSON object (convenience overload)
/// \param config JSON configuration object
/// \param expectedSampleRate Expected sample rate to configure the model with; std::nullopt uses the file default
/// \param maxBufferSize Maximum buffer size to configure the model with; std::nullopt uses NAM_DEFAULT_MAX_BUFFER_SIZE
/// \param slimmableSize Slimmable size to configure the model with; std::nullopt uses the model default
/// \return Unique pointer to a DSP object
std::unique_ptr<DSP> get_dsp(const nlohmann::json& config, std::optional<double> expectedSampleRate = std::nullopt,
                             std::optional<int> maxBufferSize = std::nullopt,
                             std::optional<double> slimmableSize = std::nullopt);

/// \brief Get sample rate from a .nam file
/// \param j JSON object from the .nam file
/// \return Sample rate in Hz, or -1 if not known (really old .nam files)
double get_sample_rate_from_nam_file(const nlohmann::json& j);
}; // namespace nam
