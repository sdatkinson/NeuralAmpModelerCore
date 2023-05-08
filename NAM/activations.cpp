#include "activations.h"

activations::ActivationTanh _TANH = activations::ActivationTanh();
activations::ActivationFastTanh _FAST_TANH = activations::ActivationFastTanh();
activations::ActivationHardTanh _HARD_TANH = activations::ActivationHardTanh();
activations::ActivationReLU _RELU = activations::ActivationReLU();
activations::ActivationSigmoid _SIGMOID = activations::ActivationSigmoid();

bool activations::Activation::using_fast_tanh = false;

std::unordered_map<std::string, activations::Activation*> activations::Activation::_activations =
  {{"Tanh", &_TANH}, {"Hardtanh", &_HARD_TANH}, {"Fasttanh", &_FAST_TANH}, {"ReLU", &_RELU}, {"Sigmoid", &_SIGMOID}};

activations::Activation* tanh_bak = nullptr;

activations::Activation* activations::Activation::get_activation(const std::string name)
{
  if (_activations.find(name) == _activations.end())
    return nullptr;

  return _activations[name];
}

void activations::Activation::enable_fast_tanh()
{
  activations::Activation::using_fast_tanh = true;

  if (_activations["Tanh"] != _activations["Fasttanh"])
  {
    tanh_bak = _activations["Tanh"];
    _activations["Tanh"] = _activations["Fasttanh"];
  }
}

void activations::Activation::disable_fast_tanh()
{
  activations::Activation::using_fast_tanh = false;

  if (_activations["Tanh"] == _activations["Fasttanh"])
  {
    _activations["Tanh"] = tanh_bak;
  }
}
