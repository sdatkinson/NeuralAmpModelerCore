// Entry point for tests
// See the GitHub Action for a demo how to build and run tests.

#include <iostream>
#include "test/test_activations.cpp"
#include "test/test_dsp.cpp"
#include "test/test_get_dsp.cpp"
#include "test/test_sequential.cpp"
#include "test/test_wavenet.cpp"

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

  test_sequential::test_construct_with_models();
  test_sequential::test_construct_empty();
  test_sequential::test_construct_mismatched_sample_rates_throws();
  test_sequential::test_construct_with_unknown_sample_rates();
  test_sequential::test_input_level_from_first_model();
  test_sequential::test_output_level_from_last_model();
  test_sequential::test_loudness();
  test_sequential::test_no_levels_when_models_dont_have_them();
  test_sequential::test_process_calls_all_models();
  test_sequential::test_process_chains_models();
  test_sequential::test_prewarm_calls_all_models();
  test_sequential::test_reset_calls_all_models();
  test_sequential::test_single_model();

  test_wavenet::test_gated();

  std::cout << "Success!" << std::endl;
  return 0;
}