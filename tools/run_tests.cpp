// Entry point for tests
// See the GitHub Action for a demo how to build and run tests.

#include <iostream>
#include "test/test_activations.cpp"
#include "test/test_dsp.cpp"
#include "test/test_get_dsp.cpp"
#include "test/test_wavenet.cpp"
#include "test/test_fast_lut.cpp"
#include "test/test_gating_activations.cpp"
#include "test/test_wavenet_gating_compatibility.cpp"

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

  // Gating activations tests
  test_gating_activations::TestGatingActivation::test_basic_functionality();
  test_gating_activations::TestGatingActivation::test_with_custom_activations();
  test_gating_activations::TestGatingActivation::test_error_handling();
  
  test_gating_activations::TestBlendingActivation::test_basic_functionality();
  test_gating_activations::TestBlendingActivation::test_different_alpha_values();
  test_gating_activations::TestBlendingActivation::test_with_custom_activations();
  test_gating_activations::TestBlendingActivation::test_error_handling();
  test_gating_activations::TestBlendingActivation::test_edge_cases();

  // Wavenet gating compatibility tests
  test_wavenet_gating_compatibility::TestWavenetGatingCompatibility::test_wavenet_style_gating();
  test_wavenet_gating_compatibility::TestWavenetGatingCompatibility::test_column_by_column_processing();
  test_wavenet_gating_compatibility::TestWavenetGatingCompatibility::test_memory_contiguity();
  test_wavenet_gating_compatibility::TestWavenetGatingCompatibility::test_multiple_channels();

  std::cout << "Success!" << std::endl;
  return 0;
}
