#include "activations.h"

// Global singleton instances (statically allocated, never deleted)
static nam::activations::ActivationTanh _TANH;
static nam::activations::ActivationFastTanh _FAST_TANH;
static nam::activations::ActivationHardTanh _HARD_TANH;
static nam::activations::ActivationReLU _RELU;
static nam::activations::ActivationLeakyReLU _LEAKY_RELU(0.01); // FIXME does not parameterize LeakyReLU
static nam::activations::ActivationPReLU _PRELU(0.01); // Same as leaky ReLU by default
static nam::activations::ActivationSigmoid _SIGMOID;
static nam::activations::ActivationSwish _SWISH;
static nam::activations::ActivationHardSwish _HARD_SWISH;
static nam::activations::ActivationLeakyHardTanh _LEAKY_HARD_TANH;
static nam::activations::ActivationSoftsign _SOFTSIGN;

bool nam::activations::Activation::using_fast_tanh = false;

// Helper to create a non-owning shared_ptr (no-op deleter) for singletons
template <typename T>
nam::activations::Activation::Ptr make_singleton_ptr(T& singleton)
{
  return nam::activations::Activation::Ptr(&singleton, [](nam::activations::Activation*) {});
}

std::unordered_map<std::string, nam::activations::Activation::Ptr> nam::activations::Activation::_activations = {
  {"Tanh", make_singleton_ptr(_TANH)},
  {"Hardtanh", make_singleton_ptr(_HARD_TANH)},
  {"Fasttanh", make_singleton_ptr(_FAST_TANH)},
  {"ReLU", make_singleton_ptr(_RELU)},
  {"LeakyReLU", make_singleton_ptr(_LEAKY_RELU)},
  {"Sigmoid", make_singleton_ptr(_SIGMOID)},
  {"SiLU", make_singleton_ptr(_SWISH)},
  {"Hardswish", make_singleton_ptr(_HARD_SWISH)},
  {"LeakyHardtanh", make_singleton_ptr(_LEAKY_HARD_TANH)},
  {"PReLU", make_singleton_ptr(_PRELU)},
  {"Softsign", make_singleton_ptr(_SOFTSIGN)}};

nam::activations::Activation::Ptr tanh_bak = nullptr;
nam::activations::Activation::Ptr sigmoid_bak = nullptr;

nam::activations::Activation::Ptr nam::activations::Activation::get_activation(const std::string name)
{
  if (_activations.find(name) == _activations.end())
    return nullptr;

  return _activations[name];
}

// ActivationConfig implementation
nam::activations::ActivationConfig nam::activations::ActivationConfig::simple(ActivationType t)
{
  ActivationConfig config;
  config.type = t;
  return config;
}

nam::activations::ActivationConfig nam::activations::ActivationConfig::from_json(const nlohmann::json& j)
{
  ActivationConfig config;

  // Map from string to ActivationType
  static const std::unordered_map<std::string, ActivationType> type_map = {
    {"Tanh", ActivationType::Tanh},
    {"Hardtanh", ActivationType::Hardtanh},
    {"Fasttanh", ActivationType::Fasttanh},
    {"ReLU", ActivationType::ReLU},
    {"LeakyReLU", ActivationType::LeakyReLU},
    {"PReLU", ActivationType::PReLU},
    {"Sigmoid", ActivationType::Sigmoid},
    {"SiLU", ActivationType::SiLU},
    {"Hardswish", ActivationType::Hardswish},
    {"LeakyHardtanh", ActivationType::LeakyHardtanh},
    {"LeakyHardTanh", ActivationType::LeakyHardtanh}, // Support both casings
    {"Softsign", ActivationType::Softsign}
  };

  // If it's a string, simple lookup
  if (j.is_string())
  {
    std::string name = j.get<std::string>();
    auto it = type_map.find(name);
    if (it == type_map.end())
    {
      throw std::runtime_error("Unknown activation type: " + name);
    }
    config.type = it->second;
    return config;
  }

  // If it's an object, parse type and parameters
  if (j.is_object())
  {
    std::string type_str = j["type"].get<std::string>();
    auto it = type_map.find(type_str);
    if (it == type_map.end())
    {
      throw std::runtime_error("Unknown activation type: " + type_str);
    }
    config.type = it->second;

    // Parse optional parameters based on activation type
    if (config.type == ActivationType::PReLU)
    {
      if (j.find("negative_slope") != j.end())
      {
        config.negative_slope = j["negative_slope"].get<float>();
      }
      else if (j.find("negative_slopes") != j.end())
      {
        config.negative_slopes = j["negative_slopes"].get<std::vector<float>>();
      }
    }
    else if (config.type == ActivationType::LeakyReLU)
    {
      config.negative_slope = j.value("negative_slope", 0.01f);
    }
    else if (config.type == ActivationType::LeakyHardtanh)
    {
      config.min_val = j.value("min_val", -1.0f);
      config.max_val = j.value("max_val", 1.0f);
      config.min_slope = j.value("min_slope", 0.01f);
      config.max_slope = j.value("max_slope", 0.01f);
    }
    else if (config.type == ActivationType::Softsign)
    {
      config.alpha = j.value("alpha", 1.0f);
    }

    return config;
  }

  throw std::runtime_error("Invalid activation config: expected string or object");
}

nam::activations::Activation::Ptr nam::activations::Activation::get_activation(const ActivationConfig& config)
{
  switch (config.type)
  {
    case ActivationType::Tanh: return _activations["Tanh"];
    case ActivationType::Hardtanh: return _activations["Hardtanh"];
    case ActivationType::Fasttanh: return _activations["Fasttanh"];
    case ActivationType::ReLU: return _activations["ReLU"];
    case ActivationType::Sigmoid: return _activations["Sigmoid"];
    case ActivationType::SiLU: return _activations["SiLU"];
    case ActivationType::Hardswish: return _activations["Hardswish"];
    case ActivationType::LeakyReLU:
      if (config.negative_slope.has_value())
      {
        return std::make_shared<ActivationLeakyReLU>(config.negative_slope.value());
      }
      return _activations["LeakyReLU"];
    case ActivationType::PReLU:
      if (config.negative_slopes.has_value())
      {
        return std::make_shared<ActivationPReLU>(config.negative_slopes.value());
      }
      else if (config.negative_slope.has_value())
      {
        return std::make_shared<ActivationPReLU>(config.negative_slope.value());
      }
      return std::make_shared<ActivationPReLU>(0.01f);
    case ActivationType::LeakyHardtanh:
      return std::make_shared<ActivationLeakyHardTanh>(config.min_val.value_or(-1.0f), config.max_val.value_or(1.0f),
                                                       config.min_slope.value_or(0.01f),
                                                       config.max_slope.value_or(0.01f));
    case ActivationType::Softsign:
      if (config.alpha.has_value())
      {
        return std::make_shared<ActivationSoftsign>(config.alpha.value());
      }
      return _activations["Softsign"];
    default: return nullptr;
  }
}

void nam::activations::Activation::enable_fast_tanh()
{
  nam::activations::Activation::using_fast_tanh = true;

  if (_activations["Tanh"] != _activations["Fasttanh"])
  {
    tanh_bak = _activations["Tanh"];
    _activations["Tanh"] = _activations["Fasttanh"];
  }
}

void nam::activations::Activation::disable_fast_tanh()
{
  nam::activations::Activation::using_fast_tanh = false;

  if (_activations["Tanh"] == _activations["Fasttanh"])
  {
    _activations["Tanh"] = tanh_bak;
  }
}

void nam::activations::Activation::enable_lut(std::string function_name, float min, float max, std::size_t n_points)
{
  std::function<float(float)> fn;
  if (function_name == "Tanh")
  {
    fn = [](float x) { return std::tanh(x); };
    tanh_bak = _activations["Tanh"];
  }
  else if (function_name == "Sigmoid")
  {
    fn = sigmoid;
    sigmoid_bak = _activations["Sigmoid"];
  }
  else
  {
    throw std::runtime_error("Tried to enable LUT for a function other than Tanh or Sigmoid");
  }
  _activations[function_name] = std::make_shared<FastLUTActivation>(min, max, n_points, fn);
}

void nam::activations::Activation::disable_lut(std::string function_name)
{
  if (function_name == "Tanh")
  {
    _activations["Tanh"] = tanh_bak;
  }
  else if (function_name == "Sigmoid")
  {
    _activations["Sigmoid"] = sigmoid_bak;
  }
  else
  {
    throw std::runtime_error("Tried to disable LUT for a function other than Tanh or Sigmoid");
  }
}
