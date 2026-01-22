#pragma once

// Registry for DSP objects

#include <string>
#include <memory>
#include <unordered_map>
#include <functional>

#include "dsp.h"

namespace nam
{
namespace factory
{
/// \brief Factory function type for creating DSP objects
using FactoryFunction = std::function<std::unique_ptr<DSP>(const nlohmann::json&, std::vector<float>&, const double)>;

/// \brief Registry for factories that instantiate DSP objects
///
/// Singleton registry that maps architecture names to factory functions.
/// Allows dynamic registration of new DSP architectures.
class FactoryRegistry
{
public:
  /// \brief Get the singleton instance
  /// \return Reference to the factory registry instance
  static FactoryRegistry& instance()
  {
    static FactoryRegistry inst;
    return inst;
  }

  /// \brief Register a factory function for an architecture
  /// \param key Architecture name (e.g., "WaveNet", "LSTM")
  /// \param func Factory function that creates DSP instances
  /// \throws std::runtime_error If the key is already registered
  void registerFactory(const std::string& key, FactoryFunction func)
  {
    // Assert that the key is not already registered
    if (factories_.find(key) != factories_.end())
    {
      throw std::runtime_error("Factory already registered for key: " + key);
    }
    factories_[key] = func;
  }

  /// \brief Create a DSP object using a registered factory
  /// \param name Architecture name
  /// \param config JSON configuration object
  /// \param weights Model weights vector
  /// \param expectedSampleRate Expected sample rate in Hz
  /// \return Unique pointer to a DSP object
  /// \throws std::runtime_error If no factory is registered for the given name
  std::unique_ptr<DSP> create(const std::string& name, const nlohmann::json& config, std::vector<float>& weights,
                              const double expectedSampleRate) const
  {
    auto it = factories_.find(name);
    if (it != factories_.end())
    {
      return it->second(config, weights, expectedSampleRate);
    }
    throw std::runtime_error("Factory not found for name: " + name);
  }

private:
  std::unordered_map<std::string, FactoryFunction> factories_;
};

/// \brief Registration helper for factories
///
/// Use this to register your factories. Create a static instance to automatically
/// register a factory when the program starts.
struct Helper
{
  /// \brief Constructor that registers a factory
  /// \param name Architecture name
  /// \param factory Factory function
  Helper(const std::string& name, FactoryFunction factory)
  {
    FactoryRegistry::instance().registerFactory(name, std::move(factory));
  }
};
} // namespace factory
} // namespace nam
