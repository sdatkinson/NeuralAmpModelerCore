#pragma once

// Registry for DSP objects — external code compatibility layer.
//
// The primary registration mechanism is ConfigParserRegistry in model_config.h.
// This header provides FactoryConfig (a ModelConfig wrapper around legacy factory
// functions) and factory::Helper (which registers into ConfigParserRegistry).

#include <string>
#include <memory>
#include <vector>
#include <functional>

#include "dsp.h"

namespace nam
{
namespace factory
{
/// \brief Factory function type for creating DSP objects (legacy interface)
using FactoryFunction = std::function<std::unique_ptr<DSP>(const nlohmann::json&, std::vector<float>&, const double)>;

/// \brief ModelConfig wrapper around a legacy FactoryFunction
///
/// Stores the factory function, JSON config, and sample rate so that
/// create() can delegate to the factory. This allows external code that
/// registers a FactoryFunction to work transparently with ConfigParserRegistry.
class FactoryConfig : public ModelConfig
{
public:
  FactoryConfig(FactoryFunction factory, nlohmann::json config, double sampleRate)
  : _factory(std::move(factory))
  , _config(std::move(config))
  , _sampleRate(sampleRate)
  {
  }

  std::unique_ptr<DSP> create(std::vector<float> weights, double sampleRate) override
  {
    (void)sampleRate; // Use stored sample rate from construction
    return _factory(_config, weights, _sampleRate);
  }

private:
  FactoryFunction _factory;
  nlohmann::json _config;
  double _sampleRate;
};

/// \brief Registration helper for factories (external code compatibility)
///
/// Wraps a FactoryFunction into a ConfigParserRegistry entry via FactoryConfig.
/// Use this to register external architectures. Create a static instance to
/// automatically register a factory when the program starts.
struct Helper
{
  /// \param name Architecture name
  /// \param factory Factory function
  Helper(const std::string& name, FactoryFunction factory)
  {
    // Capture factory by value in the lambda
    ConfigParserRegistry::instance().registerParser(
      name, [f = std::move(factory)](const nlohmann::json& config, double sampleRate) -> std::unique_ptr<ModelConfig> {
        return std::make_unique<FactoryConfig>(f, config, sampleRate);
      });
  }
};
} // namespace factory
} // namespace nam
