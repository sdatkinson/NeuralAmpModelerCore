// Tests for activation functions
//
// Things you want ot test for:
// 1. That the core elementwise funciton is snapshot-correct.
// 2. The class that wraps the core function for an array of data
// 3. .cpp: that you have the singleton defined, and that it's in the unordered map to get by string

#include <cassert>
#include <string>
#include <vector>

#include "NAM/activations.h"

namespace test_activations
{
// TODO get nonzero cases
class TestFastTanh
{
public:
  static void test_core_function()
  {
    auto TestCase = [](float input, float expectedOutput) {
      float actualOutput = nam::activations::fast_tanh(input);
      assert(actualOutput == expectedOutput);
    };
    // A few snapshot tests
    TestCase(0.0f, 0.0f);
    // TestCase(1.0f, 1.0f);
    // TestCase(-1.0f, -0.01f);
  };

  static void test_get_by_init()
  {
    auto a = nam::activations::ActivationLeakyReLU();
    _test_class(&a);
  }

  // Get the singleton and test it
  static void test_get_by_str()
  {
    const std::string name = "Fasttanh";
    auto a = nam::activations::Activation::get_activation(name);
    _test_class(a);
  }

private:
  // Put the class through its paces
  static void _test_class(nam::activations::Activation* a)
  {
    std::vector<float> inputs, expectedOutputs;

    inputs.push_back(0.0f);
    expectedOutputs.push_back(0.0f);

    // inputs.push_back(1.0f);
    // expectedOutputs.push_back(1.0f);

    // inputs.push_back(-1.0f);
    // expectedOutputs.push_back(-0.01f);

    a->apply(inputs.data(), (long)inputs.size());
    for (auto itActual = inputs.begin(), itExpected = expectedOutputs.begin(); itActual != inputs.end();
         ++itActual, ++itExpected)
    {
      assert(*itActual == *itExpected);
    }
  };
};

class TestLeakyReLU
{
public:
  static void test_core_function()
  {
    auto TestCase = [](float input, float expectedOutput) {
      float actualOutput = nam::activations::leaky_relu(input);
      assert(actualOutput == expectedOutput);
    };
    // A few snapshot tests
    TestCase(0.0f, 0.0f);
    TestCase(1.0f, 1.0f);
    TestCase(-1.0f, -0.01f);
  };

  static void test_get_by_init()
  {
    auto a = nam::activations::ActivationLeakyReLU();
    _test_class(&a);
  }

  // Get the singleton and test it
  static void test_get_by_str()
  {
    const std::string name = "LeakyReLU";
    auto a = nam::activations::Activation::get_activation(name);
    _test_class(a);
  }

private:
  // Put the class through its paces
  static void _test_class(nam::activations::Activation* a)
  {
    std::vector<float> inputs, expectedOutputs;

    inputs.push_back(0.0f);
    expectedOutputs.push_back(0.0f);

    inputs.push_back(1.0f);
    expectedOutputs.push_back(1.0f);

    inputs.push_back(-1.0f);
    expectedOutputs.push_back(-0.01f);

    a->apply(inputs.data(), (long)inputs.size());
    for (auto itActual = inputs.begin(), itExpected = expectedOutputs.begin(); itActual != inputs.end();
         ++itActual, ++itExpected)
    {
      assert(*itActual == *itExpected);
    }
  };
};
}; // namespace test_activations
