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

bool nam::activations::Activation::using_fast_tanh = false;

// Helper to create a non-owning shared_ptr (no-op deleter) for singletons
template<typename T>
nam::activations::Activation::Ptr make_singleton_ptr(T& singleton)
{
  return nam::activations::Activation::Ptr(&singleton, [](nam::activations::Activation*){});
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
  {"PReLU", make_singleton_ptr(_PRELU)}
};

nam::activations::Activation::Ptr tanh_bak = nullptr;
nam::activations::Activation::Ptr sigmoid_bak = nullptr;

nam::activations::Activation::Ptr nam::activations::Activation::get_activation(const std::string name)
{
  if (_activations.find(name) == _activations.end())
    return nullptr;

  return _activations[name];
}

nam::activations::Activation::Ptr nam::activations::Activation::get_activation(const nlohmann::json& activation_config)
{
  // If it's a string, use the existing string-based lookup
  if (activation_config.is_string())
  {
    std::string name = activation_config.get<std::string>();
    return get_activation(name);
  }

  // If it's an object, parse the activation type and parameters
  if (activation_config.is_object())
  {
    std::string type = activation_config["type"].get<std::string>();

    // Handle different activation types with parameters
    // These return owning shared_ptr (will be deleted when last reference goes out of scope)
    if (type == "PReLU")
    {
      if (activation_config.find("negative_slope") != activation_config.end())
      {
        float negative_slope = activation_config["negative_slope"].get<float>();
        return std::make_shared<ActivationPReLU>(negative_slope);
      }
      else if (activation_config.find("negative_slopes") != activation_config.end())
      {
        std::vector<float> negative_slopes = activation_config["negative_slopes"].get<std::vector<float>>();
        return std::make_shared<ActivationPReLU>(negative_slopes);
      }
      // If no parameters provided, use default
      return std::make_shared<ActivationPReLU>(0.01f);
    }
    else if (type == "LeakyReLU")
    {
      float negative_slope = activation_config.value("negative_slope", 0.01f);
      return std::make_shared<ActivationLeakyReLU>(negative_slope);
    }
    else if (type == "LeakyHardTanh")
    {
      float min_val = activation_config.value("min_val", -1.0f);
      float max_val = activation_config.value("max_val", 1.0f);
      float min_slope = activation_config.value("min_slope", 0.01f);
      float max_slope = activation_config.value("max_slope", 0.01f);
      return std::make_shared<ActivationLeakyHardTanh>(min_val, max_val, min_slope, max_slope);
    }
    else
    {
      // For other activation types without parameters, use the default string-based lookup
      return get_activation(type);
    }
  }

  return nullptr;
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
