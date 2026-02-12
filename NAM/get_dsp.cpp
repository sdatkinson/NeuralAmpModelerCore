#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <unordered_set>
#include <variant>

#include "dsp.h"
#include "registry.h"
#include "json.hpp"
#include "lstm.h"
#include "convnet.h"
#include "wavenet.h"
#include "get_dsp.h"
#include "model_config.h"

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

// =============================================================================
// Unified construction path
// =============================================================================

ModelConfig parse_model_config_json(const std::string& architecture, const nlohmann::json& config, double sample_rate)
{
  if (architecture == "Linear")
    return linear::parse_config_json(config);
  else if (architecture == "LSTM")
    return lstm::parse_config_json(config);
  else if (architecture == "ConvNet")
    return convnet::parse_config_json(config);
  else if (architecture == "WaveNet")
    return wavenet::parse_config_json(config, sample_rate);
  else
    throw std::runtime_error("Unknown architecture: " + architecture);
}

namespace
{

void apply_metadata(DSP& dsp, const ModelMetadata& metadata)
{
  if (metadata.loudness.has_value())
    dsp.SetLoudness(metadata.loudness.value());
  if (metadata.input_level.has_value())
    dsp.SetInputLevel(metadata.input_level.value());
  if (metadata.output_level.has_value())
    dsp.SetOutputLevel(metadata.output_level.value());
}

} // anonymous namespace

std::unique_ptr<DSP> create_dsp(ModelConfig config, std::vector<float> weights, const ModelMetadata& metadata)
{
  const double sample_rate = metadata.sample_rate;

  std::unique_ptr<DSP> out = std::visit(
    [&](auto&& cfg) -> std::unique_ptr<DSP> {
      using T = std::decay_t<decltype(cfg)>;
      if constexpr (std::is_same_v<T, linear::LinearConfig>)
      {
        return std::make_unique<Linear>(cfg.in_channels, cfg.out_channels, cfg.receptive_field, cfg.bias, weights,
                                        sample_rate);
      }
      else if constexpr (std::is_same_v<T, lstm::LSTMConfig>)
      {
        return std::make_unique<lstm::LSTM>(cfg.in_channels, cfg.out_channels, cfg.num_layers, cfg.input_size,
                                            cfg.hidden_size, weights, sample_rate);
      }
      else if constexpr (std::is_same_v<T, convnet::ConvNetConfig>)
      {
        return std::make_unique<convnet::ConvNet>(cfg.in_channels, cfg.out_channels, cfg.channels, cfg.dilations,
                                                  cfg.batchnorm, cfg.activation, weights, sample_rate, cfg.groups);
      }
      else if constexpr (std::is_same_v<T, wavenet::WaveNetConfig>)
      {
        return std::make_unique<wavenet::WaveNet>(cfg.in_channels, cfg.layer_array_params, cfg.head_scale,
                                                  cfg.with_head, std::move(weights), std::move(cfg.condition_dsp),
                                                  sample_rate);
      }
    },
    std::move(config));

  apply_metadata(*out, metadata);
  // FIXME should we remove prewarming from model load?
  out->prewarm();
  return out;
}

// =============================================================================
// get_dsp(dspData&) — now uses unified path
// =============================================================================

std::unique_ptr<DSP> get_dsp(dspData& conf)
{
  verify_config_version(conf.version);

  // Extract metadata from JSON
  ModelMetadata metadata;
  metadata.version = conf.version;
  metadata.sample_rate = conf.expected_sample_rate;

  if (!conf.metadata.is_null())
  {
    auto extract = [&conf](const std::string& key) -> std::optional<double> {
      if (conf.metadata.find(key) != conf.metadata.end() && !conf.metadata[key].is_null())
        return conf.metadata[key].get<double>();
      return std::nullopt;
    };
    metadata.loudness = extract("loudness");
    metadata.input_level = extract("input_level_dbu");
    metadata.output_level = extract("output_level_dbu");
  }

  ModelConfig model_config = parse_model_config_json(conf.architecture, conf.config, conf.expected_sample_rate);
  return create_dsp(std::move(model_config), std::move(conf.weights), metadata);
}

double get_sample_rate_from_nam_file(const nlohmann::json& j)
{
  if (j.find("sample_rate") != j.end())
    return j["sample_rate"];
  else
    return -1.0;
}

}; // namespace nam
