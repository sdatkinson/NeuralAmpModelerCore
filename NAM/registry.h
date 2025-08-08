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
// TODO get rid of weights and expectedSampleRate
using FactoryFunction = std::function<std::unique_ptr<DSP>(const nlohmann::json&, std::vector<float>&, const double)>;

// Register factories for instantiating DSP objects
class FactoryRegistry
{
public:
  static FactoryRegistry& instance()
  {
    static FactoryRegistry inst;
    return inst;
  }

  void registerFactory(const std::string& key, FactoryFunction func)
  {
    // Assert that the key is not already registered
    if (factories_.find(key) != factories_.end())
    {
      throw std::runtime_error("Factory already registered for key: " + key);
    }
    factories_[key] = func;
  }

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

// Registration helper. Use this to register your factories.
struct Helper
{
  Helper(const std::string& name, FactoryFunction factory)
  {
    FactoryRegistry::instance().registerFactory(name, std::move(factory));
  }
};
} // namespace factory
} // namespace nam
