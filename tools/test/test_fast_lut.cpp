#include <cassert>
#include <string>
#include <vector>
#include <cmath>

#include "NAM/fast_lut.h"

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
      FastLUT lut_sigmoid(-8.0f, 8.0f, 1024, [](float x) {
          return 1.0f / (1.0f + expf(-x));
      });

      float input = 1.25f;
      std::cout << "exact sigmoid: " << sigmoid(input) << std::endl;
      std::cout << "lut sigmoid:   " << lut_sigmoid(input) << std::endl;
    }
   static void test_tanh()
    {
      // create a lut for sigmoid from -8.0 to 8.0 with 1024 samples
      FastLUT lut_tanh(-8.0f, 8.0f, 1024, [](float x) {
          return std::tanh(x);
      });

      float input = 1.25f;
      std::cout << "exact tanh: " << std::tanh(input) << std::endl;
      std::cout << "lut tanh:   " << lut_tanh(input) << std::endl;
    }
};
}

