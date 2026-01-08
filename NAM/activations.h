#pragma once

#include <string>
#include <cmath> // expf
#include <unordered_map>
#include <Eigen/Dense>

namespace nam
{
namespace activations
{
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

inline float leaky_tanh(float x, float min_val, float max_val, float min_slope, float max_slope)
{
  if (x < min_val) {
    return (x - min_val) * min_slope + min_val;
  } else if (x > max_val) {
    return (x - max_val) * max_slope + max_val;
  } else {
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
  
// Assumes PyTorch default of 0.01 for negative slope. This may change to be
// configurable in the future.
inline float leaky_relu(float x, float negative_slope)
{
  //const float negative_slope = 0.01;
  return x > 0.0f ? x : negative_slope * x;
}

inline float swish(float x)
{
  return x * sigmoid(x);
}

inline float hardswish(float x)
{
  if (x <= -3.0) {
    return 0;
  } else if (x >= 3.0) {
    return x;
  } else {
    return x * (x + 3.0)/6.0;
  }
}

class Activation
{
public:
  Activation() = default;
  virtual ~Activation() = default;
  virtual void apply(Eigen::MatrixXf& matrix) { apply(matrix.data(), matrix.rows() * matrix.cols()); }
  virtual void apply(Eigen::Block<Eigen::MatrixXf> block) { apply(block.data(), block.rows() * block.cols()); }
  virtual void apply(Eigen::Block<Eigen::MatrixXf, -1, -1, true> block)
  {
    apply(block.data(), block.rows() * block.cols());
  }
  virtual void apply(float* data, long size) {}

  static Activation* get_activation(const std::string name);
  static void enable_fast_tanh();
  static void disable_fast_tanh();
  static bool using_fast_tanh;

protected:
  static std::unordered_map<std::string, Activation*> _activations;
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
  float negative_slope;
  ActivationLeakyReLU(float ns) {
    negative_slope = ns;
  }
  void apply(float* data, long size) override
  {
    for (long pos = 0; pos < size; pos++)
    {
      data[pos] = leaky_relu(data[pos], negative_slope);
    }
  }
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

}; // namespace activations
}; // namespace nam
