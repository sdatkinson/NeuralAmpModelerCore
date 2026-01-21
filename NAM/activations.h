#pragma once

#include <cassert>
#include <cmath> // expf
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <Eigen/Dense>

#include "json.hpp"

namespace nam
{
namespace activations
{

// Forward declaration
class Activation;

// Strongly-typed activation type enum
enum class ActivationType
{
  Tanh,
  Hardtanh,
  Fasttanh,
  ReLU,
  LeakyReLU,
  PReLU,
  Sigmoid,
  SiLU, // aka Swish
  Hardswish,
  LeakyHardtanh
};

// Strongly-typed activation configuration
struct ActivationConfig
{
  ActivationType type;

  // Optional parameters (used by specific activation types)
  std::optional<float> negative_slope;               // LeakyReLU, PReLU (single)
  std::optional<std::vector<float>> negative_slopes; // PReLU (per-channel)
  std::optional<float> min_val;                      // LeakyHardtanh
  std::optional<float> max_val;                      // LeakyHardtanh
  std::optional<float> min_slope;                    // LeakyHardtanh
  std::optional<float> max_slope;                    // LeakyHardtanh

  // Convenience constructors
  static ActivationConfig simple(ActivationType t);
  static ActivationConfig from_json(const nlohmann::json& j);
};
inline float relu(float x)
{
  return x > 0.0f ? x : 0.0f;
};

inline float sigmoid(float x)
{
  return 1.0f / (1.0f + expf(-x));
};

inline float hard_tanh(float x)
{
  const float t = x < -1 ? -1 : x;
  return t > 1 ? 1 : t;
}

inline float leaky_hardtanh(float x, float min_val, float max_val, float min_slope, float max_slope)
{
  if (x < min_val)
  {
    return (x - min_val) * min_slope + min_val;
  }
  else if (x > max_val)
  {
    return (x - max_val) * max_slope + max_val;
  }
  else
  {
    return x;
  }
}

inline float fast_tanh(const float x)
{
  const float ax = fabsf(x);
  const float x2 = x * x;

  return (x * (2.45550750702956f + 2.45550750702956f * ax + (0.893229853513558f + 0.821226666969744f * ax) * x2)
          / (2.44506634652299f + (2.44506634652299f + x2) * fabsf(x + 0.814642734961073f * x * ax)));
}

inline float fast_sigmoid(const float x)
{
  return 0.5f * (fast_tanh(x * 0.5f) + 1.0f);
}

inline float leaky_relu(float x, float negative_slope)
{
  return x > 0.0f ? x : negative_slope * x;
}
inline float leaky_relu(float x)
{
  return leaky_relu(x, 0.01);
}


inline float swish(float x)
{
  return x * sigmoid(x);
}

inline float hardswish(float x)
{
  if (x <= -3.0)
  {
    return 0;
  }
  else if (x >= 3.0)
  {
    return x;
  }
  else
  {
    return x * (x + 3.0) / 6.0;
  }
}

class Activation
{
public:
  // Type alias for shared pointer to Activation
  using Ptr = std::shared_ptr<Activation>;

  Activation() = default;
  virtual ~Activation() = default;
  virtual void apply(Eigen::MatrixXf& matrix) { apply(matrix.data(), matrix.rows() * matrix.cols()); }
  virtual void apply(Eigen::Block<Eigen::MatrixXf> block) { apply(block.data(), block.rows() * block.cols()); }
  virtual void apply(Eigen::Block<Eigen::MatrixXf, -1, -1, true> block)
  {
    apply(block.data(), block.rows() * block.cols());
  }
  virtual void apply(float* data, long size) {}

  static Ptr get_activation(const std::string name);
  static Ptr get_activation(const ActivationConfig& config);
  static Ptr get_activation(const nlohmann::json& activation_config);
  static void enable_fast_tanh();
  static void disable_fast_tanh();
  static bool using_fast_tanh;
  static void enable_lut(std::string function_name, float min, float max, std::size_t n_points);
  static void disable_lut(std::string function_name);

protected:
  static std::unordered_map<std::string, Ptr> _activations;
};

// identity function activation
class ActivationIdentity : public nam::activations::Activation
{
public:
  ActivationIdentity() = default;
  ~ActivationIdentity() = default;
  // Inherit the default apply methods which do nothing
};

class ActivationTanh : public Activation
{
public:
  void apply(float* data, long size) override
  {
    for (long pos = 0; pos < size; pos++)
    {
      data[pos] = std::tanh(data[pos]);
    }
  }
};

class ActivationHardTanh : public Activation
{
public:
  void apply(float* data, long size) override
  {
    for (long pos = 0; pos < size; pos++)
    {
      data[pos] = hard_tanh(data[pos]);
    }
  }
};

class ActivationLeakyHardTanh : public Activation
{
public:
  ActivationLeakyHardTanh() = default;
  ActivationLeakyHardTanh(float min_val_, float max_val_, float min_slope_, float max_slope_)
  {
    min_val = min_val_;
    max_val = max_val_;
    min_slope = min_slope_;
    max_slope = max_slope_;
  }
  void apply(float* data, long size) override
  {
    for (long pos = 0; pos < size; pos++)
    {
      data[pos] = leaky_hardtanh(data[pos], min_val, max_val, min_slope, max_slope);
    }
  }

private:
  float min_val = -1.0;
  float max_val = 1.0;
  float min_slope = 0.01;
  float max_slope = 0.01;
};

class ActivationFastTanh : public Activation
{
public:
  void apply(float* data, long size) override
  {
    for (long pos = 0; pos < size; pos++)
    {
      data[pos] = fast_tanh(data[pos]);
    }
  }
};

class ActivationReLU : public Activation
{
public:
  void apply(float* data, long size) override
  {
    for (long pos = 0; pos < size; pos++)
    {
      data[pos] = relu(data[pos]);
    }
  }
};

class ActivationLeakyReLU : public Activation
{
public:
  ActivationLeakyReLU() = default;
  ActivationLeakyReLU(float ns) { negative_slope = ns; }
  void apply(float* data, long size) override
  {
    for (long pos = 0; pos < size; pos++)
    {
      data[pos] = leaky_relu(data[pos], negative_slope);
    }
  }

private:
  float negative_slope = 0.01;
};

class ActivationPReLU : public Activation
{
public:
  ActivationPReLU() = default;
  ActivationPReLU(float ns)
  {
    negative_slopes.clear();
    negative_slopes.push_back(ns);
  }
  ActivationPReLU(std::vector<float> ns) { negative_slopes = ns; }

  void apply(Eigen::MatrixXf& matrix) override
  {
    // Matrix is organized as (channels, time_steps)
    unsigned long actual_channels = static_cast<unsigned long>(matrix.rows());
    
    // Prepare the slopes for the current matrix size
    std::vector<float> slopes_for_channels = negative_slopes;

    // Fail loudly if input has more channels than activation
    assert(actual_channels == negative_slopes.size());
    
    // Apply each negative slope to its corresponding channel
    for (unsigned long channel = 0; channel < actual_channels; channel++)
    {
      // Apply the negative slope to all time steps in this channel
      for (int time_step = 0; time_step < matrix.cols(); time_step++)
      {
        matrix(channel, time_step) = leaky_relu(matrix(channel, time_step), slopes_for_channels[channel]);
      }
    }
  }

private:
  std::vector<float> negative_slopes;
};


class ActivationSigmoid : public Activation
{
public:
  void apply(float* data, long size) override
  {
    for (long pos = 0; pos < size; pos++)
    {
      data[pos] = sigmoid(data[pos]);
    }
  }
};

class ActivationSwish : public Activation
{
public:
  void apply(float* data, long size) override
  {
    for (long pos = 0; pos < size; pos++)
    {
      data[pos] = swish(data[pos]);
    }
  }
};

class ActivationHardSwish : public Activation
{
public:
  void apply(float* data, long size) override
  {
    for (long pos = 0; pos < size; pos++)
    {
      data[pos] = hardswish(data[pos]);
    }
  }
};

class FastLUTActivation : public Activation
{
public:
  FastLUTActivation(float min_x, float max_x, std::size_t size, std::function<float(float)> f)
  : min_x_(min_x)
  , max_x_(max_x)
  , size_(size)
  {

    step_ = (max_x - min_x) / (size - 1);
    inv_step_ = 1.0f / step_;
    table_.reserve(size);

    for (std::size_t i = 0; i < size; ++i)
    {
      table_.push_back(f(min_x + i * step_));
    }
  }

  // Fast lookup with linear interpolation
  inline float lookup(float x) const
  {
    // Clamp input to range
    x = std::clamp(x, min_x_, max_x_);

    // Calculate float index
    float f_idx = (x - min_x_) * inv_step_;
    std::size_t i = static_cast<std::size_t>(f_idx);

    // Handle edge case at max_x_
    if (i >= size_ - 1)
      return table_.back();

    // Linear interpolation: y = y0 + (y1 - y0) * fractional_part
    float frac = f_idx - static_cast<float>(i);
    return table_[i] + (table_[i + 1] - table_[i]) * frac;
  }

  // Override base class virtual method to apply LUT lookup to array of floats
  void apply(float* data, long size) override
  {
    for (long i = 0; i < size; i++)
    {
      data[i] = lookup(data[i]);
    }
  }

private:
  float min_x_, max_x_, step_, inv_step_;
  size_t size_;
  std::vector<float> table_;
};

}; // namespace activations
}; // namespace nam
