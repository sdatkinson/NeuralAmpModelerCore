#include <algorithm>
#include <cstdio>
#include <vector>

#include "NAM/nam_c_api.h"

int main(int argc, char* argv[])
{
  if (argc < 2)
  {
    std::fprintf(stderr, "Usage: c_api_example <model.nam>\n");
    return 1;
  }

  nam_model_t* model = nullptr;
  nam_status_t status = nam_create_model_from_file(argv[1], &model);
  if (status != NAM_STATUS_OK || model == nullptr)
  {
    std::fprintf(stderr, "Failed to load model: %s\n", nam_get_last_error());
    return 1;
  }

  const double sample_rate = 48000.0;
  const int block_size = 64;
  status = nam_reset(model, sample_rate, block_size);
  if (status != NAM_STATUS_OK)
  {
    std::fprintf(stderr, "nam_reset failed: %s\n", nam_get_last_error());
    nam_destroy_model(model);
    return 1;
  }

  const int in_channels = std::max(1, nam_num_input_channels(model));
  const int out_channels = std::max(1, nam_num_output_channels(model));

  std::vector<std::vector<nam_sample_t>> input_storage(in_channels, std::vector<nam_sample_t>(block_size, 0.0));
  std::vector<std::vector<nam_sample_t>> output_storage(out_channels, std::vector<nam_sample_t>(block_size, 0.0));

  std::vector<nam_sample_t*> input(in_channels);
  std::vector<nam_sample_t*> output(out_channels);
  for (int ch = 0; ch < in_channels; ++ch)
    input[ch] = input_storage[ch].data();
  for (int ch = 0; ch < out_channels; ++ch)
    output[ch] = output_storage[ch].data();

  status = nam_process(model, input.data(), output.data(), block_size);
  if (status != NAM_STATUS_OK)
  {
    std::fprintf(stderr, "nam_process failed: %s\n", nam_get_last_error());
    nam_destroy_model(model);
    return 1;
  }

  std::printf("Model loaded with %d input(s), %d output(s), expected SR: %.1f\n", in_channels, out_channels,
              nam_expected_sample_rate(model));
  std::printf("Processed %d frames. First output sample: %f\n", block_size, static_cast<double>(output[0][0]));

  nam_destroy_model(model);
  return 0;
}
