#include "compiler.h"
#include "dsp.h"
#include "get_dsp.h"
#include "model_config.h"

#if NAM_HAS_JSON
  #include <fstream>
  #include "json.hpp"
  #include "nam_file.h"
  #include "registry.h"
#endif

namespace nam
{
namespace
{

/// Parses "major.minor.patch" without exceptions or <regex>.
bool try_parse_version(const std::string& versionStr, Version& parsed)
{
  int components[3] = {0, 0, 0};
  size_t pos = 0;
  for (int i = 0; i < 3; i++)
  {
    if (i > 0)
    {
      if (pos >= versionStr.size() || versionStr[pos] != '.')
        return false;
      pos++;
    }
    const size_t start = pos;
    int value = 0;
    while (pos < versionStr.size() && versionStr[pos] >= '0' && versionStr[pos] <= '9')
    {
      if (value > 100000)
        return false;
      value = value * 10 + (versionStr[pos] - '0');
      pos++;
    }
    if (pos == start)
      return false;
    components[i] = value;
  }
  if (pos != versionStr.size())
    return false;

  parsed = Version(components[0], components[1], components[2]);
  return true;
}

class CoreVersionSupportChecker : public IVersionSupportChecker
{
public:
  Supported support(const std::string& version) const override
  {
    Version parsed(0, 0, 0);
    if (!try_parse_version(version, parsed))
      return Supported::NO;

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

Mutex& version_support_registry_mutex()
{
  static Mutex registry_mutex;
  return registry_mutex;
}

} // namespace

Version ParseVersion(const std::string& versionStr)
{
  Version parsed(0, 0, 0);
  if (!try_parse_version(versionStr, parsed))
    NAM_THROW(std::invalid_argument("Invalid version string: " + versionStr));
  return parsed;
}

void register_version_support_checker(std::shared_ptr<const IVersionSupportChecker> checker)
{
  if (!checker)
    NAM_THROW(std::invalid_argument("version support checker cannot be null"));
  LockGuard lock(version_support_registry_mutex());
  version_support_registry().push_back(std::move(checker));
}

Supported is_version_supported(const std::string version)
{
  LockGuard lock(version_support_registry_mutex());
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
    NAM_THROW(std::runtime_error("Model config is an unsupported version " + versionStr + "."));
  // Partially-supported versions are accepted as-is.
}

#if NAM_HAS_JSON
std::vector<float> GetWeights(nlohmann::json const& j)
{
  auto it = j.find("weights");
  if (it != j.end())
  {
    return *it;
  }
  else
    NAM_THROW(std::runtime_error("Corrupted model file is missing weights."));
}

void populate_dsp_data(const nlohmann::json& config, dspData& returnedConfig)
{
  verify_config_version(config["version"].get<std::string>());

  nlohmann::json config_json = config["config"];
  std::vector<float> weights = GetWeights(config);

  returnedConfig.version = config["version"].get<std::string>();
  returnedConfig.architecture = config["architecture"].get<std::string>();
  returnedConfig.config = config_json;
  returnedConfig.metadata = config.value("metadata", nlohmann::json());
  returnedConfig.weights = weights;
  returnedConfig.expected_sample_rate = nam::get_sample_rate_from_nam_file(config);
}

std::unique_ptr<DSP> get_dsp(const std::filesystem::path config_filename, DspLoadOptions options)
{
  dspData temp;
  return get_dsp(config_filename, temp, options);
}

std::unique_ptr<DSP> get_dsp(const nlohmann::json& config, DspLoadOptions options)
{
  dspData temp;
  return get_dsp(config, temp, options);
}

std::unique_ptr<DSP> get_dsp(const std::filesystem::path config_filename, dspData& returnedConfig,
                             DspLoadOptions options)
{
  const auto j = validate_nam_file(config_filename);
  populate_dsp_data(j, returnedConfig);

  /*Copy to a new dsp_config object for get_dsp below,
   since not sure if weights actually get modified as being non-const references on some
   model constructors inside get_dsp(dsp_config& conf).
   We need to return unmodified version of dsp_config via returnedConfig.*/
  dspData conf = returnedConfig;

  return get_dsp(conf, options);
}

std::unique_ptr<DSP> get_dsp(const nlohmann::json& config, dspData& returnedConfig, DspLoadOptions options)
{
  populate_dsp_data(config, returnedConfig);

  /*Copy to a new dsp_config object for get_dsp below,
   since not sure if weights actually get modified as being non-const references on some
   model constructors inside get_dsp(dsp_config& conf).
   We need to return unmodified version of dsp_config via returnedConfig.*/
  dspData conf = returnedConfig;

  return get_dsp(conf, options);
}

// =============================================================================
// Unified construction path
// =============================================================================

std::unique_ptr<ModelConfig> parse_model_config_json(const std::string& architecture, const nlohmann::json& config,
                                                     double sample_rate)
{
  return ConfigParserRegistry::instance().parse(architecture, config, sample_rate);
}
#endif // NAM_HAS_JSON

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

std::unique_ptr<DSP> create_dsp(std::unique_ptr<ModelConfig> config, std::vector<float> weights,
                                const ModelMetadata& metadata)
{
  auto out = config->create(std::move(weights), metadata.sample_rate);
  apply_metadata(*out, metadata);
  return out;
}

#if NAM_HAS_JSON
namespace
{

std::unique_ptr<DSP> get_dsp_with_current_prewarm_default(dspData& conf)
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

  auto model_config = ConfigParserRegistry::instance().parse(conf.architecture, conf.config, conf.expected_sample_rate);
  return create_dsp(std::move(model_config), std::move(conf.weights), metadata);
}

} // anonymous namespace

std::unique_ptr<DSP> get_dsp(dspData& conf, DspLoadOptions options)
{
  if (!options.prewarm.has_value())
    return get_dsp_with_current_prewarm_default(conf);

  ScopedPrewarmOnResetDefault scoped_prewarm_default(*options.prewarm);
  auto dsp = get_dsp_with_current_prewarm_default(conf);
  if (dsp != nullptr)
    dsp->SetPrewarmOnReset(scoped_prewarm_default.PreviousPrewarmOnReset());
  return dsp;
}

double get_sample_rate_from_nam_file(const nlohmann::json& j)
{
  if (j.find("sample_rate") != j.end())
    return j["sample_rate"];
  else
    return -1.0;
}
#endif // NAM_HAS_JSON

}; // namespace nam
