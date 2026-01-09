#include "activations.h"

nam::activations::ActivationTanh _TANH = nam::activations::ActivationTanh();
nam::activations::ActivationFastTanh _FAST_TANH = nam::activations::ActivationFastTanh();
nam::activations::ActivationHardTanh _HARD_TANH = nam::activations::ActivationHardTanh();
nam::activations::ActivationReLU _RELU = nam::activations::ActivationReLU();
nam::activations::ActivationLeakyReLU _LEAKY_RELU = nam::activations::ActivationLeakyReLU(0.01); //FIXME does not parameterize LeakyReLU
nam::activations::ActivationSigmoid _SIGMOID = nam::activations::ActivationSigmoid();
nam::activations::ActivationSwish _SWISH = nam::activations::ActivationSwish();
nam::activations::ActivationHardSwish _HARD_SWISH = nam::activations::ActivationHardSwish();
nam::activations::ActivationLeakyHardTanh _LEAKY_HARD_TANH = nam::activations::ActivationLeakyHardTanh();

bool nam::activations::Activation::using_fast_tanh = false;

std::unordered_map<std::string, nam::activations::Activation*> nam::activations::Activation::_activations = {
  {"Tanh", &_TANH}, {"Hardtanh", &_HARD_TANH},   {"Fasttanh", &_FAST_TANH},
  {"ReLU", &_RELU}, {"LeakyReLU", &_LEAKY_RELU}, {"Sigmoid", &_SIGMOID},
  {"SiLU", &_SWISH}, {"Hardswish", &_HARD_SWISH}, {"LeakyHardtanh", &_LEAKY_HARD_TANH}};

nam::activations::Activation* tanh_bak = nullptr;

nam::activations::Activation* nam::activations::Activation::get_activation(const std::string name)
{
  if (_activations.find(name) == _activations.end())
    return nullptr;

  return _activations[name];
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
