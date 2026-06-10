#include <fstream>
#include <iostream>
#include <mutex>
#include <regex>
#include <sstream>
#include <stdexcept>

#include "dsp.h"
#include "registry.h"
#include "json.hpp"
#include "get_dsp.h"
#include "model_config.h"
#include "slimmable.h"

namespace nam
{
std::vector<float> GetWeights(nlohmann::json const& j);

namespace
{

struct LoadOptions
{
  std::optional<double> expectedSampleRate;
  std::optional<int> maxBufferSize;
  std::optional<double> slimmableSize;

  bool requires_initial_reset() const
  {
    return expectedSampleRate.has_value() || maxBufferSize.has_value() || slimmableSize.has_value();
  }
};

class CoreVersionSupportChecker : public IVersionSupportChecker
{
public:
  Supported support(const std::string& version) const override
  {
    static const std::regex semver_regex(R"(^\d+\.\d+\.\d+$)");
    if (!std::regex_match(version, semver_regex))
      return Supported::NO;

    const Version parsed = ParseVersion(version);
    const Version latest = ParseVersion(LATEST_FULLY_SUPPORTED_NAM_FILE_VERSION);
    const Version earliest = ParseVersion(EARLIEST_SUPPORTED_NAM_FILE_VERSION);

    if (parsed < earliest)
      return Supported::NO;
    if (parsed.major > latest.major || parsed.minor > latest.minor)
      return Supported::NO;
    if (latest < parsed)
      return Supported::PARTIAL;
    return Supported::YES;
  }
};

std::vector<std::shared_ptr<const IVersionSupportChecker>>& version_support_registry()
{
  static std::vector<std::shared_ptr<const IVersionSupportChecker>> registry{
    std::make_shared<CoreVersionSupportChecker>()};
  return registry;
}

std::mutex& version_support_registry_mutex()
{
  static std::mutex registry_mutex;
  return registry_mutex;
}

dspData parse_dsp_data(const nlohmann::json& config, std::optional<double> expectedSampleRate)
{
  verify_config_version(config["version"].get<std::string>());

  dspData out;
  out.version = config["version"].get<std::string>();
  out.architecture = config["architecture"].get<std::string>();
  out.config = config["config"];
  out.metadata = config.value("metadata", nlohmann::json());
  out.weights = GetWeights(config);
  out.expected_sample_rate = expectedSampleRate.value_or(nam::get_sample_rate_from_nam_file(config));
  return out;
}

void apply_initial_slimmable_size(DSP& dsp, const double slimmableSize)
{
  auto* slimmable = dynamic_cast<SlimmableModel*>(&dsp);
  if (slimmable == nullptr)
    throw std::runtime_error("Cannot set slimmable size on a model that is not slimmable.");
  slimmable->SetSlimmableSize(slimmableSize);
}

void apply_metadata(DSP& dsp, const ModelMetadata& metadata)
{
  if (metadata.loudness.has_value())
    dsp.SetLoudness(metadata.loudness.value());
  if (metadata.input_level.has_value())
    dsp.SetInputLevel(metadata.input_level.value());
  if (metadata.output_level.has_value())
    dsp.SetOutputLevel(metadata.output_level.value());
}

void configure_initial_state(DSP& dsp, const ModelMetadata& metadata, const LoadOptions& options)
{
  if (options.slimmableSize.has_value())
    apply_initial_slimmable_size(dsp, options.slimmableSize.value());

  if (options.requires_initial_reset())
  {
    const double sampleRate = options.expectedSampleRate.value_or(metadata.sample_rate);
    const int maxBufferSize = options.maxBufferSize.value_or(NAM_DEFAULT_MAX_BUFFER_SIZE);
    dsp.Reset(sampleRate, maxBufferSize);
  }
  else
  {
    // Preserve the historical load behavior when no load-time configuration is requested.
    dsp.prewarm();
  }
}

std::unique_ptr<DSP> create_dsp_with_options(std::unique_ptr<ModelConfig> config, std::vector<float> weights,
                                             const ModelMetadata& metadata, const LoadOptions& options)
{
  auto out = config->create(std::move(weights), metadata.sample_rate);
  apply_metadata(*out, metadata);
  configure_initial_state(*out, metadata, options);
  return out;
}

} // namespace

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

void register_version_support_checker(std::shared_ptr<const IVersionSupportChecker> checker)
{
  if (!checker)
    throw std::invalid_argument("version support checker cannot be null");
  std::lock_guard<std::mutex> lock(version_support_registry_mutex());
  version_support_registry().push_back(std::move(checker));
}

Supported is_version_supported(const std::string version)
{
  std::lock_guard<std::mutex> lock(version_support_registry_mutex());
  Supported best_support = Supported::NO;
  for (const auto& checker : version_support_registry())
  {
    const auto candidate_support = checker->support(version);
    if (static_cast<int>(candidate_support) > static_cast<int>(best_support))
      best_support = candidate_support;
  }
  return best_support;
}

void verify_config_version(const std::string versionStr)
{
  const Supported support = is_version_supported(versionStr);
  if (support == Supported::NO)
  {
    std::stringstream ss;
    ss << "Model config is an unsupported version " << versionStr << ".";
    throw std::runtime_error(ss.str());
  }
  if (support == Supported::PARTIAL)
  {
    std::stringstream ss;
    std::cerr << "Model config is a partially-supported version " << versionStr << ". Continuing with partial support."
              << std::endl;
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

std::unique_ptr<DSP> get_dsp(const std::filesystem::path config_filename, std::optional<double> expectedSampleRate,
                             std::optional<int> maxBufferSize, std::optional<double> slimmableSize)
{
  dspData temp;
  return get_dsp(config_filename, temp, expectedSampleRate, maxBufferSize, slimmableSize);
}

std::unique_ptr<DSP> get_dsp(const nlohmann::json& config, std::optional<double> expectedSampleRate,
                             std::optional<int> maxBufferSize, std::optional<double> slimmableSize)
{
  dspData temp;
  return get_dsp(config, temp, expectedSampleRate, maxBufferSize, slimmableSize);
}

std::unique_ptr<DSP> get_dsp(const std::filesystem::path config_filename, dspData& returnedConfig,
                             std::optional<double> expectedSampleRate, std::optional<int> maxBufferSize,
                             std::optional<double> slimmableSize)
{
  if (!std::filesystem::exists(config_filename))
    throw std::runtime_error("Config file doesn't exist!\n");
  std::ifstream i(config_filename);
  nlohmann::json j;
  i >> j;
  return get_dsp(j, returnedConfig, expectedSampleRate, maxBufferSize, slimmableSize);
}

std::unique_ptr<DSP> get_dsp(const nlohmann::json& config, dspData& returnedConfig,
                             std::optional<double> expectedSampleRate, std::optional<int> maxBufferSize,
                             std::optional<double> slimmableSize)
{
  returnedConfig = parse_dsp_data(config, expectedSampleRate);

  /*Copy to a new dsp_config object for get_dsp below,
   since not sure if weights actually get modified as being non-const references on some
   model constructors inside get_dsp(dsp_config& conf).
   We need to return unmodified version of dsp_config via returnedConfig.*/
  dspData conf = returnedConfig;

  return get_dsp(conf, expectedSampleRate, maxBufferSize, slimmableSize);
}

// =============================================================================
// Unified construction path
// =============================================================================

std::unique_ptr<ModelConfig> parse_model_config_json(const std::string& architecture, const nlohmann::json& config,
                                                     double sample_rate)
{
  return ConfigParserRegistry::instance().parse(architecture, config, sample_rate);
}

std::unique_ptr<DSP> create_dsp(std::unique_ptr<ModelConfig> config, std::vector<float> weights,
                                const ModelMetadata& metadata)
{
  return create_dsp_with_options(std::move(config), std::move(weights), metadata, LoadOptions{});
}

// =============================================================================
// get_dsp(dspData&) — now uses unified path
// =============================================================================

std::unique_ptr<DSP> get_dsp(dspData& conf, std::optional<double> expectedSampleRate, std::optional<int> maxBufferSize,
                             std::optional<double> slimmableSize)
{
  verify_config_version(conf.version);
  const double effectiveSampleRate = expectedSampleRate.value_or(conf.expected_sample_rate);
  const LoadOptions options{expectedSampleRate, maxBufferSize, slimmableSize};

  // Extract metadata from JSON
  ModelMetadata metadata;
  metadata.version = conf.version;
  metadata.sample_rate = effectiveSampleRate;

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

  auto model_config = ConfigParserRegistry::instance().parse(conf.architecture, conf.config, effectiveSampleRate);
  return create_dsp_with_options(std::move(model_config), std::move(conf.weights), metadata, options);
}

double get_sample_rate_from_nam_file(const nlohmann::json& j)
{
  if (j.find("sample_rate") != j.end())
    return j["sample_rate"];
  else
    return -1.0;
}

}; // namespace nam
