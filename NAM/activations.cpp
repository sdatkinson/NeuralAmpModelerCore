#pragma once

#include "activations.h"

std::unordered_map<std::string, activations::Activation*> activations::Activation::_activations = {
  {"Tanh", new activations::ActivationHardTanh()},
  {"Hardtanh", new activations::ActivationHardTanh()},
  {"Fasttanh", new activations::ActivationFastTanh()},
  {"ReLU", new activations::ActivationReLU()},
  {"Sigmoid", new activations::ActivationSigmoid()}};
