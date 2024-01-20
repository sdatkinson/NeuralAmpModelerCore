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


class Activation
{
public:
  Activation() = default;
  virtual ~Activation() = default;
  virtual void Apply(Eigen::Ref<Eigen::MatrixXf> matrix) { Apply(matrix.data(), matrix.rows() * matrix.cols()); }
  virtual void Apply(Eigen::Block<Eigen::MatrixXf> block) { Apply(block.data(), block.rows() * block.cols()); }
  virtual void Apply(Eigen::Block<Eigen::MatrixXf, -1, -1, true> block)
  {
    Apply(block.data(), block.rows() * block.cols());
  }
  virtual void Apply(float* data, long size) {}

  static Activation* GetActivation(const std::string& name);
  static void EnableFastTanh();
  static void DisableFastTanh();
  static bool sUsingFastTanh;

protected:
  static std::unordered_map<std::string, Activation*> _activations;
};

class ActivationTanh : public Activation
{
public:
  void Apply(float* data, long size) override
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
  void Apply(float* data, long size) override
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
  void Apply(float* data, long size) override
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
  void Apply(float* data, long size) override
  {
    for (long pos = 0; pos < size; pos++)
    {
      data[pos] = relu(data[pos]);
    }
  }
};

class ActivationSigmoid : public Activation
{
public:
  void Apply(float* data, long size) override
  {
    for (long pos = 0; pos < size; pos++)
    {
      data[pos] = sigmoid(data[pos]);
    }
  }
};
}; // namespace activations
}; // namespace nam
