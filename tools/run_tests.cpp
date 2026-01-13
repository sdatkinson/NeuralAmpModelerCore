// Entry point for tests
// See the GitHub Action for a demo how to build and run tests.

#include <iostream>
#include "test/test_activations.cpp"
#include "test/test_conv1d.cpp"
#include "test/test_convnet.cpp"
#include "test/test_dsp.cpp"
#include "test/test_get_dsp.cpp"
#include "test/test_ring_buffer.cpp"
#include "test/test_wavenet/test_layer.cpp"
#include "test/test_wavenet/test_layer_array.cpp"
#include "test/test_wavenet/test_full.cpp"

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

  test_ring_buffer::test_construct();
  test_ring_buffer::test_reset();
  test_ring_buffer::test_reset_with_receptive_field();
  test_ring_buffer::test_write();
  test_ring_buffer::test_read_with_lookback();
  test_ring_buffer::test_advance();
  test_ring_buffer::test_rewind();
  test_ring_buffer::test_multiple_writes_reads();
  test_ring_buffer::test_reset_zeros_history_area();
  test_ring_buffer::test_rewind_preserves_history();

  test_conv1d::test_construct();
  test_conv1d::test_set_size();
  test_conv1d::test_reset();
  test_conv1d::test_process_basic();
  test_conv1d::test_process_with_bias();
  test_conv1d::test_process_multichannel();
  test_conv1d::test_process_dilation();
  test_conv1d::test_process_multiple_calls();
  test_conv1d::test_get_output_different_sizes();
  test_conv1d::test_set_size_and_weights();
  test_conv1d::test_get_num_weights();
  test_conv1d::test_reset_multiple();

  test_wavenet::test_layer::test_gated();
  test_wavenet::test_layer::test_layer_getters();
  test_wavenet::test_layer::test_non_gated_layer();
  test_wavenet::test_layer::test_layer_activations();
  test_wavenet::test_layer::test_layer_multichannel();
  test_wavenet::test_layer_array::test_layer_array_basic();
  test_wavenet::test_layer_array::test_layer_array_receptive_field();
  test_wavenet::test_layer_array::test_layer_array_with_head_input();
  test_wavenet::test_full::test_wavenet_model();
  test_wavenet::test_full::test_wavenet_multiple_arrays();
  test_wavenet::test_full::test_wavenet_zero_input();
  test_wavenet::test_full::test_wavenet_different_buffer_sizes();
  test_wavenet::test_full::test_wavenet_prewarm();

  test_convnet::test_convnet_basic();
  test_convnet::test_convnet_batchnorm();
  test_convnet::test_convnet_multiple_blocks();
  test_convnet::test_convnet_zero_input();
  test_convnet::test_convnet_different_buffer_sizes();
  test_convnet::test_convnet_prewarm();
  test_convnet::test_convnet_multiple_calls();

  std::cout << "Success!" << std::endl;
  return 0;
}