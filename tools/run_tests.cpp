// Entry point for tests
// See the GitHub Action for a demo how to build and run tests.

#include <iostream>
#include "test/test_activations.cpp"
#include "test/test_dsp.cpp"
#include "test/test_get_dsp.cpp"
#include "test/test_wavenet.cpp"
#include "test/test_fast_lut.cpp"

int main()
{
  std::cout << "Running tests..." << std::endl;
  // TODO Automatically loop, catch exceptions, log results

  test_activations::TestFastTanh::test_core_function();
  test_activations::TestFastTanh::test_get_by_init();
  test_activations::TestFastTanh::test_get_by_str();

  test_activations::TestLeakyReLU::test_core_function();
  test_activations::TestLeakyReLU::test_get_by_init();
  test_activations::TestLeakyReLU::test_get_by_str();

  test_activations::TestPReLU::test_core_function();
  test_activations::TestPReLU::test_per_channel_behavior();
  // This is enforced by an assert so it doesn't need to be tested
  //test_activations::TestPReLU::test_wrong_number_of_channels();

  test_dsp::test_construct();
  test_dsp::test_get_input_level();
  test_dsp::test_get_output_level();
  test_dsp::test_has_input_level();
  test_dsp::test_has_output_level();
  test_dsp::test_set_input_level();
  test_dsp::test_set_output_level();

  test_get_dsp::test_gets_input_level();
  test_get_dsp::test_gets_output_level();
  test_get_dsp::test_null_input_level();
  test_get_dsp::test_null_output_level();

  test_wavenet::test_gated();

  test_lut::TestFastLUT::test_sigmoid();
  test_lut::TestFastLUT::test_tanh();

  std::cout << "Success!" << std::endl;
  return 0;
}
