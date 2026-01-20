#include "activations.h"

nam::activations::ActivationTanh _TANH = nam::activations::ActivationTanh();
nam::activations::ActivationFastTanh _FAST_TANH = nam::activations::ActivationFastTanh();
nam::activations::ActivationHardTanh _HARD_TANH = nam::activations::ActivationHardTanh();
nam::activations::ActivationReLU _RELU = nam::activations::ActivationReLU();
nam::activations::ActivationLeakyReLU _LEAKY_RELU =
  nam::activations::ActivationLeakyReLU(0.01); // FIXME does not parameterize LeakyReLU
nam::activations::ActivationPReLU _PRELU = nam::activations::ActivationPReLU(0.01); // Same as leaky ReLU by default
nam::activations::ActivationSigmoid _SIGMOID = nam::activations::ActivationSigmoid();
nam::activations::ActivationSwish _SWISH = nam::activations::ActivationSwish();
nam::activations::ActivationHardSwish _HARD_SWISH = nam::activations::ActivationHardSwish();
nam::activations::ActivationLeakyHardTanh _LEAKY_HARD_TANH = nam::activations::ActivationLeakyHardTanh();

bool nam::activations::Activation::using_fast_tanh = false;

std::unordered_map<std::string, nam::activations::Activation*> nam::activations::Activation::_activations = {
  {"Tanh", &_TANH},  {"Hardtanh", &_HARD_TANH},   {"Fasttanh", &_FAST_TANH},
  {"ReLU", &_RELU},  {"LeakyReLU", &_LEAKY_RELU}, {"Sigmoid", &_SIGMOID},
  {"SiLU", &_SWISH}, {"Hardswish", &_HARD_SWISH}, {"LeakyHardtanh", &_LEAKY_HARD_TANH},
  {"PReLU", &_PRELU}};

nam::activations::Activation* tanh_bak = nullptr;
nam::activations::Activation* sigmoid_bak = nullptr;

nam::activations::Activation* nam::activations::Activation::get_activation(const std::string name)
{
  if (_activations.find(name) == _activations.end())
    return nullptr;

  return _activations[name];
}

nam::activations::Activation* nam::activations::Activation::get_activation(const nlohmann::json& activation_config)
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
    if (type == "PReLU")
    {
      if (activation_config.find("negative_slope") != activation_config.end())
      {
        float negative_slope = activation_config["negative_slope"].get<float>();
        return new ActivationPReLU(negative_slope);
      }
      else if (activation_config.find("negative_slopes") != activation_config.end())
      {
        std::vector<float> negative_slopes = activation_config["negative_slopes"].get<std::vector<float>>();
        return new ActivationPReLU(negative_slopes);
      }
      // If no parameters provided, use default
      return new ActivationPReLU(0.01);
    }
    else if (type == "LeakyReLU")
    {
      float negative_slope = activation_config.value("negative_slope", 0.01f);
      return new ActivationLeakyReLU(negative_slope);
    }
    else if (type == "LeakyHardTanh")
    {
      float min_val = activation_config.value("min_val", -1.0f);
      float max_val = activation_config.value("max_val", 1.0f);
      float min_slope = activation_config.value("min_slope", 0.01f);
      float max_slope = activation_config.value("max_slope", 0.01f);
      return new ActivationLeakyHardTanh(min_val, max_val, min_slope, max_slope);
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
  FastLUTActivation lut_activation(min, max, n_points, fn);
  _activations[function_name] = &lut_activation;
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
