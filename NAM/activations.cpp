#include "activations.h"

nam::activations::ActivationTanh _TANH = nam::activations::ActivationTanh();
nam::activations::ActivationFastTanh _FAST_TANH = nam::activations::ActivationFastTanh();
nam::activations::ActivationHardTanh _HARD_TANH = nam::activations::ActivationHardTanh();
nam::activations::ActivationReLU _RELU = nam::activations::ActivationReLU();
nam::activations::ActivationLeakyReLU _LEAKY_RELU = nam::activations::ActivationLeakyReLU();
nam::activations::ActivationSigmoid _SIGMOID = nam::activations::ActivationSigmoid();

std::atomic<bool> nam::activations::Activation::using_fast_tanh{false};

std::unordered_map<std::string, nam::activations::Activation*> nam::activations::Activation::_activations = {
  {"Tanh", &_TANH}, {"Hardtanh", &_HARD_TANH},   {"Fasttanh", &_FAST_TANH},
  {"ReLU", &_RELU}, {"LeakyReLU", &_LEAKY_RELU}, {"Sigmoid", &_SIGMOID}};

nam::activations::Activation* nam::activations::Activation::get_activation(const std::string name)
{
  // Return FastTanh when Tanh is requested and fast_tanh mode is enabled
  if (name == "Tanh" && using_fast_tanh.load(std::memory_order_relaxed))
  {
    return _activations.at("Fasttanh");
  }

  auto it = _activations.find(name);
  if (it == _activations.end())
    return nullptr;

  return it->second;
}

void nam::activations::Activation::enable_fast_tanh()
{
  using_fast_tanh.store(true, std::memory_order_relaxed);
}

void nam::activations::Activation::disable_fast_tanh()
{
  using_fast_tanh.store(false, std::memory_order_relaxed);
}
