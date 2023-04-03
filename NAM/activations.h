#pragma once

#include <string>
#include <cmath> // expf
#include <unordered_map>
#include <Eigen/Dense>

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

class Activation
{
public:
Activation(){};
  virtual void apply(Eigen::MatrixXf& matrix) { apply(matrix.block(0, 0, matrix.rows(), matrix.cols())); }
  virtual void apply(Eigen::Block<Eigen::MatrixXf> block) {}
  static Activation* get_activation(const std::string name)
  { 
    if (_activations.find(name) == _activations.end())
      return nullptr;

    return _activations[name];
  }

  protected:
    static std::unordered_map<std::string, Activation *> _activations;
};

class ActivationTanh : public Activation
{
  public:
    void apply(Eigen::Block<Eigen::MatrixXf> block) override
    {
      float* ptr = block.data();

      long size = block.rows() * block.cols();

      for (long pos = 0; pos < size; pos++)
      {
        ptr[pos] = std::tanh(ptr[pos]);
      }
    }
};

class ActivationHardTanh : public Activation
{
  public:
    ActivationHardTanh(){};
    void apply(Eigen::Block<Eigen::MatrixXf> block) override
    {
      float* ptr = block.data();

      long size = block.rows() * block.cols();

      for (long pos = 0; pos < size; pos++)
      {
        ptr[pos] = hard_tanh(ptr[pos]);
      }
    }
};

class ActivationFastTanh : public Activation
{
  public:
    ActivationFastTanh(){};
    void apply(Eigen::Block<Eigen::MatrixXf> block) override
    {
      float* ptr = block.data();

      long size = block.rows() * block.cols();

      for (long pos = 0; pos < size; pos++)
      {
        ptr[pos] = fast_tanh(ptr[pos]);
      }
    }
};

class ActivationReLU : public Activation
{
  public:
    ActivationReLU(){};
    void apply(Eigen::Block<Eigen::MatrixXf> block) override
    {
      float* ptr = block.data();

      long size = block.rows() * block.cols();

      for (long pos = 0; pos < size; pos++)
      {
        ptr[pos] = relu(ptr[pos]);
      }
    }
};

class ActivationSigmoid : public Activation
{
  public:
    ActivationSigmoid(){};
    void apply(Eigen::Block<Eigen::MatrixXf> block) override
    {
      float* ptr = block.data();

      long size = block.rows() * block.cols();

      for (long pos = 0; pos < size; pos++)
      {
        ptr[pos] = sigmoid(ptr[pos]);
      }
    }
};

}; // namespace activations