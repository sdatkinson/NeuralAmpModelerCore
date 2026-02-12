#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

#include "dsp.h"
#include "registry.h"
#include "json.hpp"
#include "lstm.h"
#include "convnet.h"
#include "wavenet.h"
#include "get_dsp.h"

namespace nam
{
Version ParseVersion(const std::string& versionStr)
{
  // Split the version string into major, minor, and patch components
  std::stringstream ss(versionStr);
  std::string majorStr, minorStr, patchStr;
  std::getline(ss, majorStr, '.');
  std::getline(ss, minorStr, '.');
  std::getline(ss, patchStr);

  // Parse the components as integers and assign them to the version struct
  int major;
  int minor;
  int patch;
  try
  {
    major = std::stoi(majorStr);
    minor = std::stoi(minorStr);
    patch = std::stoi(patchStr);
  }
  catch (const std::invalid_argument&)
  {
    throw std::invalid_argument("Invalid version string: " + versionStr);
  }
  catch (const std::out_of_range&)
  {
    throw std::out_of_range("Version string out of range: " + versionStr);
  }

  // Validate the semver components
  if (major < 0 || minor < 0 || patch < 0)
  {
    throw std::invalid_argument("Negative version component: " + versionStr);
  }
  return Version(major, minor, patch);
}

void verify_config_version(const std::string versionStr)
{
  Version version = ParseVersion(versionStr);
  Version currentVersion = ParseVersion(LATEST_FULLY_SUPPORTED_NAM_FILE_VERSION);
  Version earliestSupportedVersion = ParseVersion(EARLIEST_SUPPORTED_NAM_FILE_VERSION);

  if (version < earliestSupportedVersion)
  {
    std::stringstream ss;
    ss << "Model config is an unsupported version " << versionStr << ". The earliest supported version is "
       << earliestSupportedVersion.toString()
       << ". Try either converting the model to a more recent version, or "
          "update your version of the NAM plugin.";
    throw std::runtime_error(ss.str());
  }
  if (version.major > currentVersion.major || version.minor > currentVersion.minor)
  {
    std::stringstream ss;
    ss << "Model config is an unsupported version " << versionStr << ". The latest fully-supported version is "
       << currentVersion.toString();
    throw std::runtime_error(ss.str());
  }
  else if (version.major == 0 && version.minor == 6 && version.patch > 0)
  {
    std::cerr << "Model config is a partially-supported version " << versionStr
              << ". The latest fully-supported version is " << currentVersion.toString()
              << ". Continuing with partial support." << std::endl;
  }
}

std::vector<float> GetWeights(nlohmann::json const& j)
{
  auto it = j.find("weights");
  if (it != j.end())
  {
    return *it;
  }
  else
    throw std::runtime_error("Corrupted model file is missing weights.");
}

std::unique_ptr<DSP> get_dsp(const std::filesystem::path config_filename)
{
  dspData temp;
  return get_dsp(config_filename, temp);
}

std::unique_ptr<DSP> get_dsp(const nlohmann::json& config)
{
  dspData temp;
  return get_dsp(config, temp);
}

std::unique_ptr<DSP> get_dsp(const std::filesystem::path config_filename, dspData& returnedConfig)
{
  if (!std::filesystem::exists(config_filename))
    throw std::runtime_error("Config file doesn't exist!\n");
  std::ifstream i(config_filename);
  nlohmann::json j;
  i >> j;
  get_dsp(j, returnedConfig);

  /*Copy to a new dsp_config object for get_dsp below,
   since not sure if weights actually get modified as being non-const references on some
   model constructors inside get_dsp(dsp_config& conf).
   We need to return unmodified version of dsp_config via returnedConfig.*/
  dspData conf = returnedConfig;

  return get_dsp(conf);
}

std::unique_ptr<DSP> get_dsp(const nlohmann::json& config, dspData& returnedConfig)
{
  verify_config_version(config["version"].get<std::string>());

  auto architecture = config["architecture"];
  nlohmann::json config_json = config["config"];
  std::vector<float> weights = GetWeights(config);

  // Assign values to returnedConfig
  returnedConfig.version = config["version"].get<std::string>();
  returnedConfig.architecture = config["architecture"].get<std::string>();
  returnedConfig.config = config_json;
  returnedConfig.metadata = config.value("metadata", nlohmann::json());
  returnedConfig.weights = weights;
  returnedConfig.expected_sample_rate = nam::get_sample_rate_from_nam_file(config);

  /*Copy to a new dsp_config object for get_dsp below,
   since not sure if weights actually get modified as being non-const references on some
   model constructors inside get_dsp(dsp_config& conf).
   We need to return unmodified version of dsp_config via returnedConfig.*/
  dspData conf = returnedConfig;

  return get_dsp(conf);
}

struct OptionalValue
{
  bool have = false;
  double value = 0.0;
};

std::unique_ptr<DSP> get_dsp(dspData& conf)
{
  verify_config_version(conf.version);

  // Explicit registration avoids missing factories when NAM is linked as a static library.
  nam::lstm::RegisterFactory();
  nam::convnet::RegisterFactory();
  nam::wavenet::RegisterFactory();

  auto& architecture = conf.architecture;
  nlohmann::json& config = conf.config;
  std::vector<float>& weights = conf.weights;
  OptionalValue loudness, inputLevel, outputLevel;

  auto AssignOptional = [&conf](const std::string key, OptionalValue& v) {
    if (conf.metadata.find(key) != conf.metadata.end())
    {
      if (!conf.metadata[key].is_null())
      {
        v.value = conf.metadata[key];
        v.have = true;
      }
    }
  };

  if (!conf.metadata.is_null())
  {
    AssignOptional("loudness", loudness);
    AssignOptional("input_level_dbu", inputLevel);
    AssignOptional("output_level_dbu", outputLevel);
  }
  const double expectedSampleRate = conf.expected_sample_rate;

  // Initialize using registry-based factory
  std::unique_ptr<DSP> out =
    nam::factory::FactoryRegistry::instance().create(architecture, config, weights, expectedSampleRate);

  if (loudness.have)
  {
    out->SetLoudness(loudness.value);
  }
  if (inputLevel.have)
  {
    out->SetInputLevel(inputLevel.value);
  }
  if (outputLevel.have)
  {
    out->SetOutputLevel(outputLevel.value);
  }

  // "pre-warm" the model to settle initial conditions
  // Can this be removed now that it's part of Reset()?
  out->prewarm();

  return out;
}

double get_sample_rate_from_nam_file(const nlohmann::json& j)
{
  if (j.find("sample_rate") != j.end())
    return j["sample_rate"];
  else
    return -1.0;
}

}; // namespace nam
