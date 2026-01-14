#pragma once

#include <string>
#include <cmath> // expf
#include <unordered_map>
#include <Eigen/Dense>
#include <functional>
#include <stdexcept>
#include "activations.h"

namespace nam
{
namespace gating_activations
{

class GatingActivation
{
public:
  GatingActivation(activations::Activation* input_act = nullptr, activations::Activation* gating_act = nullptr)
    : input_activation(input_act ? input_act : &default_activation),
      gating_activation(gating_act ? gating_act : &default_activation)
  {
  }
  
  ~GatingActivation() = default;
  
  void apply(Eigen::MatrixXf& input, Eigen::MatrixXf& output)
  {
    // Validate input dimensions
    if (input.rows() < 2) {
      throw std::invalid_argument("GatingActivation: input matrix must have at least 2 rows");
    }
    
    // Ensure output has correct dimensions
    if (output.rows() != 1 || output.cols() != input.cols()) {
      throw std::invalid_argument("GatingActivation: output matrix must have at least 1 row");
    }
    
    // Apply activations to the two rows
    Eigen::MatrixXf input_row = input.row(0);
    Eigen::MatrixXf gating_row = input.row(1);
    
    input_activation->apply(input_row);
    gating_activation->apply(gating_row);
    
    // Element-wise multiplication
    output = input_row.array() * gating_row.array();
  }

private:
  activations::Activation* input_activation;
  activations::Activation* gating_activation;
};

class BlendingActivation
{
public:
  BlendingActivation(activations::Activation* input_act = nullptr, activations::Activation* blend_act = nullptr, float alpha_val = 0.5f)
    : input_activation(input_act ? input_act : &default_activation),
      blending_activation(blend_act ? blend_act : &default_activation),
      alpha(alpha_val)
  {
    // Validate alpha is in valid range
    if (alpha < 0.0f || alpha > 1.0f) {
      throw std::invalid_argument("BlendingActivation: alpha must be between 0.0 and 1.0");
    }
  }
  
  ~BlendingActivation() = default;
  
  void apply(Eigen::MatrixXf& input, Eigen::MatrixXf& output)
  {
    // Validate input dimensions
    if (input.rows() < 2) {
      throw std::invalid_argument("BlendingActivation: input matrix must have at least 2 rows");
    }
    
    // Ensure output has correct dimensions
    if (output.rows() != 1 || output.cols() != input.cols()) {
      throw std::invalid_argument("BlendingActivation: output matrix must have at least 1 row");
    }
    
    // Apply activations to the two rows
    Eigen::MatrixXf input_row = input.row(0);
    Eigen::MatrixXf blend_row = input.row(1);
    
    input_activation->apply(input_row);
    blending_activation->apply(blend_row);
    
    // Weighted blending
    output = alpha * input_row + (1.0f - alpha) * blend_row;
  }

private:
  activations::Activation* input_activation;
  activations::Activation* blending_activation;
  float alpha;
};


}; // namespace gating_activations
}; // namespace nam
