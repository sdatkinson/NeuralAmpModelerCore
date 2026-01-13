#include <cassert>
#include <string>
#include <vector>
#include <cmath>

#include "NAM/activations.h"

namespace test_lut {

float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

class TestFastLUT
{
  public:
    static void test_sigmoid()
    {
      // create a lut for sigmoid from -8.0 to 8.0 with 1024 samples
      nam::activations::FastLUTActivation lut_sigmoid(-8.0f, 8.0f, 1024, [](float x) {
          return 1.0f / (1.0f + expf(-x));
      });

      float input = 1.25f;
      assert(abs(sigmoid(input) - lut_sigmoid.lookup(input)) < 1e-3);
    }
   static void test_tanh()
    {
      // create a lut for sigmoid from -8.0 to 8.0 with 1024 samples
      nam::activations::FastLUTActivation lut_tanh(-8.0f, 8.0f, 1024, [](float x) {
          return std::tanh(x);
      });

      float input = 1.25f;
      assert(abs(std::tanh(input) - lut_tanh.lookup(input)) < 1e-3);
    }
};
}

