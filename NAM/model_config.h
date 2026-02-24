#pragma once
// Unified model configuration: abstract base class + config parser registry.
// No circular dependencies: forward-declares DSP (no #include "dsp.h").

#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "json.hpp"

namespace nam
{

// Forward declaration — no #include "dsp.h"
class DSP;

/// \brief Metadata common to all model formats
struct ModelMetadata
{
  std::string version;
  double sample_rate = -1.0;
  std::optional<double> loudness;
  std::optional<double> input_level;
  std::optional<double> output_level;
};

/// \brief Abstract base class for architecture-specific configuration
///
/// Each architecture defines a concrete config struct that inherits from this
/// and implements create() to construct the DSP object.
class ModelConfig
{
public:
  virtual ~ModelConfig() = default;

  /// \brief Construct a DSP object from this configuration
  /// \param weights Model weights (taken by value to allow move for WaveNet)
  /// \param sampleRate Expected sample rate in Hz
  /// \return Unique pointer to a DSP object
  virtual std::unique_ptr<DSP> create(std::vector<float> weights, double sampleRate) = 0;
};

/// \brief Function type for parsing a ModelConfig from JSON
using ConfigParserFunction = std::function<std::unique_ptr<ModelConfig>(const nlohmann::json&, double)>;

/// \brief Singleton registry mapping architecture names to config parser functions
///
/// Both built-in and external architectures register here. There is one
/// construction path for all architectures.
class ConfigParserRegistry
{
public:
  static ConfigParserRegistry& instance()
  {
    static ConfigParserRegistry inst;
    return inst;
  }

  /// \brief Register a config parser for an architecture
  /// \param name Architecture name (e.g., "WaveNet", "LSTM")
  /// \param func Parser function that returns a unique_ptr<ModelConfig>
  /// \throws std::runtime_error If the name is already registered
  void registerParser(const std::string& name, ConfigParserFunction func)
  {
    if (parsers_.find(name) != parsers_.end())
    {
      throw std::runtime_error("Config parser already registered for: " + name);
    }
    parsers_[name] = std::move(func);
  }

  /// \brief Check whether an architecture name is registered
  bool has(const std::string& name) const { return parsers_.find(name) != parsers_.end(); }

  /// \brief Parse a ModelConfig from an architecture name, JSON config, and sample rate
  /// \throws std::runtime_error If no parser is registered for the given name
  std::unique_ptr<ModelConfig> parse(const std::string& name, const nlohmann::json& config, double sampleRate) const
  {
    auto it = parsers_.find(name);
    if (it == parsers_.end())
    {
      throw std::runtime_error("No config parser registered for architecture: " + name);
    }
    return it->second(config, sampleRate);
  }

private:
  std::unordered_map<std::string, ConfigParserFunction> parsers_;
};

/// \brief Auto-registration helper for config parsers
///
/// Create a static instance to register a config parser at program startup.
struct ConfigParserHelper
{
  ConfigParserHelper(const std::string& name, ConfigParserFunction func)
  {
    ConfigParserRegistry::instance().registerParser(name, std::move(func));
  }
};

/// \brief Construct a DSP object from a typed config, weights, and metadata
///
/// This is the single construction path used by both JSON and binary loaders.
/// Handles construction, metadata application, and prewarm.
/// \param config Architecture-specific configuration (abstract base)
/// \param weights Model weights (taken by value to allow move for WaveNet)
/// \param metadata Model metadata (version, sample rate, loudness, levels)
/// \return Unique pointer to a DSP object
std::unique_ptr<DSP> create_dsp(std::unique_ptr<ModelConfig> config, std::vector<float> weights,
                                const ModelMetadata& metadata);

/// \brief Parse a ModelConfig from a JSON architecture name and config block
/// \param architecture Architecture name string (e.g., "WaveNet", "LSTM")
/// \param config JSON config block for this architecture
/// \param sample_rate Expected sample rate from metadata
/// \return unique_ptr<ModelConfig>
std::unique_ptr<ModelConfig> parse_model_config_json(const std::string& architecture, const nlohmann::json& config,
                                                     double sample_rate);

} // namespace nam
