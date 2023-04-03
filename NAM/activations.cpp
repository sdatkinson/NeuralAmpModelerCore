#include "activations.h"

std::unordered_map<std::string, activations::Activation*> activations::Activation::_activations = {
  {"Tanh", new activations::ActivationTanh()},
  {"Hardtanh", new activations::ActivationHardTanh()},
  {"Fasttanh", new activations::ActivationFastTanh()},
  {"ReLU", new activations::ActivationReLU()},
  {"Sigmoid", new activations::ActivationSigmoid()}};

activations::Activation* tanh_bak = nullptr;

activations::Activation* activations::Activation::get_activation(const std::string name)
{
  if (_activations.find(name) == _activations.end())
    return nullptr;

  return _activations[name];
}

void activations::Activation::enable_fast_tanh()
{
  if (_activations["Tanh"] != _activations["Fasttanh"])
  {
    tanh_bak = _activations["Tanh"];
    _activations["Tanh"] = _activations["Fasttanh"];
  }
}

void activations::Activation::disable_fast_tanh()
{
  if (_activations["Tanh"] == _activations["Fasttanh"])
  {
    _activations["Tanh"] = tanh_bak;
  }
}
