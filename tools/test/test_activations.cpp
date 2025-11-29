#include <string>
#include <vector>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "NAM/activations.h"

TEST_CASE("Test fast_tanh core function", "[activations]")
{
  REQUIRE(nam::activations::fast_tanh(0.0f) == 0.0f);
  // The original test had these commented out.
  // REQUIRE(nam::activations::fast_tanh(1.0f) == Catch::Approx(0.98f));
  // REQUIRE(nam::activations::fast_tanh(-1.0f) == Catch::Approx(-0.98f));
}

void _test_fast_tanh_class(nam::activations::Activation* a)
{
  std::vector<float> inputs, expectedOutputs;

  inputs.push_back(0.0f);
  expectedOutputs.push_back(0.0f);

  // inputs.push_back(1.0f);
  // expectedOutputs.push_back(1.0f);

  // inputs.push_back(-1.0f);
  // expectedOutputs.push_back(-0.01f);

  a->apply(inputs.data(), (long)inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i)
  {
    REQUIRE(inputs[i] == Catch::Approx(expectedOutputs[i]));
  }
}

TEST_CASE("Test FastTanh get by init", "[activations]")
{
  auto a = nam::activations::ActivationFastTanh();
  _test_fast_tanh_class(&a);
}

TEST_CASE("Test FastTanh get by str", "[activations]")
{
  const std::string name = "Fasttanh";
  auto a = nam::activations::Activation::get_activation(name);
  REQUIRE(a != nullptr);
  _test_fast_tanh_class(a);
}

TEST_CASE("Test LeakyReLU core function", "[activations]")
{
  REQUIRE(nam::activations::leaky_relu(0.0f) == 0.0f);
  REQUIRE(nam::activations::leaky_relu(1.0f) == 1.0f);
  REQUIRE(nam::activations::leaky_relu(-1.0f) == -0.01f);
}

void _test_leaky_relu_class(nam::activations::Activation* a)
{
  std::vector<float> inputs, expectedOutputs;

  inputs.push_back(0.0f);
  expectedOutputs.push_back(0.0f);

  inputs.push_back(1.0f);
  expectedOutputs.push_back(1.0f);

  inputs.push_back(-1.0f);
  expectedOutputs.push_back(-0.01f);

  a->apply(inputs.data(), (long)inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i)
  {
    REQUIRE(inputs[i] == Catch::Approx(expectedOutputs[i]));
  }
}

TEST_CASE("Test LeakyReLU get by init", "[activations]")
{
  auto a = nam::activations::ActivationLeakyReLU();
  _test_leaky_relu_class(&a);
}

TEST_CASE("Test LeakyReLU get by str", "[activations]")
{
  const std::string name = "LeakyReLU";
  auto a = nam::activations::Activation::get_activation(name);
  REQUIRE(a != nullptr);
  _test_leaky_relu_class(a);
}