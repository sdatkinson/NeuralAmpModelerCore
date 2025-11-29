#include <catch2/catch_test_macros.hpp>
#include "NAM/activations.h"

TEST_CASE("FastTanh core function", "[activations]") {
  REQUIRE(nam::activations::fast_tanh(0.0f) == 0.0f);
}

TEST_CASE("FastTanh get by init", "[activations]") {
  auto a = nam::activations::ActivationLeakyReLU();
  std::vector<float> inputs{0.0f};
  a.apply(inputs.data(), inputs.size());
  REQUIRE(inputs[0] == 0.0f);
}

TEST_CASE("FastTanh get by string", "[activations]") {
  auto a = nam::activations::Activation::get_activation("Fasttanh");
  std::vector<float> inputs{0.0f};
  a->apply(inputs.data(), inputs.size());
  REQUIRE(inputs[0] == 0.0f);
}

TEST_CASE("LeakyReLU core function", "[activations]") {
  REQUIRE(nam::activations::leaky_relu(0.0f) == 0.0f);
  REQUIRE(nam::activations::leaky_relu(1.0f) == 1.0f);
  REQUIRE(nam::activations::leaky_relu(-1.0f) == -0.01f);
}

TEST_CASE("LeakyReLU get by init", "[activations]") {
  auto a = nam::activations::ActivationLeakyReLU();
  std::vector<float> inputs{0.0f, 1.0f, -1.0f};
  std::vector<float> expected{0.0f, 1.0f, -0.01f};
  a.apply(inputs.data(), inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i) {
    REQUIRE(inputs[i] == expected[i]);
  }
}

TEST_CASE("LeakyReLU get by string", "[activations]") {
  auto a = nam::activations::Activation::get_activation("LeakyReLU");
  std::vector<float> inputs{0.0f, 1.0f, -1.0f};
  std::vector<float> expected{0.0f, 1.0f, -0.01f};
  a->apply(inputs.data(), inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i) {
    REQUIRE(inputs[i] == expected[i]);
  }
}
