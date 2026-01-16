// Entry point for tests
// See the GitHub Action for a demo how to build and run tests.

#include <iostream>
#include "test/test_activations.cpp"
#include "test/test_conv1d.cpp"
#include "test/test_conv_1x1.cpp"
#include "test/test_convnet.cpp"
#include "test/test_dsp.cpp"
#include "test/test_fast_lut.cpp"
#include "test/test_get_dsp.cpp"
#include "test/test_ring_buffer.cpp"
#include "test/test_wavenet/test_layer.cpp"
#include "test/test_wavenet/test_layer_array.cpp"
#include "test/test_wavenet/test_full.cpp"
#include "test/test_wavenet/test_real_time_safe.cpp"
#include "test/test_gating_activations.cpp"
#include "test/test_wavenet_gating_compatibility.cpp"
#include "test/test_blending_detailed.cpp"
#include "test/test_input_buffer_verification.cpp"

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

  test_lut::TestFastLUT::test_sigmoid();
  test_lut::TestFastLUT::test_tanh();

  test_activations::TestPReLU::test_core_function();
  test_activations::TestPReLU::test_per_channel_behavior();
  // This is enforced by an assert so it doesn't need to be tested
  // test_activations::TestPReLU::test_wrong_number_of_channels();

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
  test_conv1d::test_process_grouped_basic();
  test_conv1d::test_process_grouped_with_bias();
  test_conv1d::test_process_grouped_multiple_groups();
  test_conv1d::test_process_grouped_kernel_size();
  test_conv1d::test_process_grouped_dilation();
  test_conv1d::test_process_grouped_channel_isolation();
  test_conv1d::test_get_num_weights_grouped();

  test_conv_1x1::test_construct();
  test_conv_1x1::test_construct_with_groups();
  test_conv_1x1::test_construct_validation_in_channels();
  test_conv_1x1::test_construct_validation_out_channels();
  test_conv_1x1::test_process_basic();
  test_conv_1x1::test_process_with_bias();
  test_conv_1x1::test_process_underscore();
  test_conv_1x1::test_process_grouped_basic();
  test_conv_1x1::test_process_grouped_with_bias();
  test_conv_1x1::test_process_grouped_multiple_groups();
  test_conv_1x1::test_process_grouped_channel_isolation();
  test_conv_1x1::test_process_underscore_grouped();
  test_conv_1x1::test_set_max_buffer_size();
  test_conv_1x1::test_process_multiple_calls();

  test_wavenet::test_layer::test_gated();
  test_wavenet::test_layer::test_layer_getters();
  test_wavenet::test_layer::test_non_gated_layer();
  test_wavenet::test_layer::test_layer_activations();
  test_wavenet::test_layer::test_layer_multichannel();
  test_wavenet::test_layer::test_layer_bottleneck();
  test_wavenet::test_layer::test_layer_bottleneck_gated();
  test_wavenet::test_layer_array::test_layer_array_basic();
  test_wavenet::test_layer_array::test_layer_array_receptive_field();
  test_wavenet::test_layer_array::test_layer_array_with_head_input();
  test_wavenet::test_full::test_wavenet_model();
  test_wavenet::test_full::test_wavenet_multiple_arrays();
  test_wavenet::test_full::test_wavenet_zero_input();
  test_wavenet::test_full::test_wavenet_different_buffer_sizes();
  test_wavenet::test_full::test_wavenet_prewarm();
  test_wavenet::test_allocation_tracking_pass();
  test_wavenet::test_allocation_tracking_fail();
  test_wavenet::test_conv1d_process_realtime_safe();
  test_wavenet::test_conv1d_grouped_process_realtime_safe();
  test_wavenet::test_conv1d_grouped_dilated_process_realtime_safe();
  test_wavenet::test_layer_process_realtime_safe();
  test_wavenet::test_layer_bottleneck_process_realtime_safe();
  test_wavenet::test_layer_grouped_process_realtime_safe();
  test_wavenet::test_layer_array_process_realtime_safe();
  test_wavenet::test_process_realtime_safe();

  test_convnet::test_convnet_basic();
  test_convnet::test_convnet_batchnorm();
  test_convnet::test_convnet_multiple_blocks();
  test_convnet::test_convnet_zero_input();
  test_convnet::test_convnet_different_buffer_sizes();
  test_convnet::test_convnet_prewarm();
  test_convnet::test_convnet_multiple_calls();

  // Gating activations tests
  test_gating_activations::TestGatingActivation::test_basic_functionality();
  test_gating_activations::TestGatingActivation::test_with_custom_activations();
  //  test_gating_activations::TestGatingActivation::test_error_handling();

  // Wavenet gating compatibility tests
  test_wavenet_gating_compatibility::TestWavenetGatingCompatibility::test_wavenet_style_gating();
  test_wavenet_gating_compatibility::TestWavenetGatingCompatibility::test_column_by_column_processing();
  test_wavenet_gating_compatibility::TestWavenetGatingCompatibility::test_memory_contiguity();
  test_wavenet_gating_compatibility::TestWavenetGatingCompatibility::test_multiple_channels();

  test_gating_activations::TestBlendingActivation::test_basic_functionality();
  test_gating_activations::TestBlendingActivation::test_blending_behavior();
  test_gating_activations::TestBlendingActivation::test_with_custom_activations();
  //  test_gating_activations::TestBlendingActivation::test_error_handling();
  test_gating_activations::TestBlendingActivation::test_edge_cases();

  // Detailed blending tests
  test_blending_detailed::TestBlendingDetailed::test_blending_with_different_activations();
  test_blending_detailed::TestBlendingDetailed::test_input_buffer_usage();

  // Input buffer verification tests
  test_input_buffer_verification::TestInputBufferVerification::test_buffer_stores_pre_activation_values();
  test_input_buffer_verification::TestInputBufferVerification::test_buffer_with_different_activations();

  std::cout << "Success!" << std::endl;
  return 0;
}
