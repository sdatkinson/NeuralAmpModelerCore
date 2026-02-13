#include "nam_c_api.h"

#include <exception>
#include <filesystem>
#include <memory>
#include <string>
#include <utility>

#include "activations.h"
#include "get_dsp.h"

struct nam_model
{
  std::unique_ptr<nam::DSP> dsp;
};

namespace
{
thread_local std::string g_last_error;

void set_error(std::string message)
{
  g_last_error = std::move(message);
}

nam_status_t handle_exception()
{
  try
  {
    throw;
  }
  catch (const std::exception& e)
  {
    set_error(e.what());
  }
  catch (...)
  {
    set_error("Unknown exception");
  }
  return NAM_STATUS_EXCEPTION;
}
} // namespace

extern "C"
{

nam_status_t nam_create_model_from_file(const char* model_path, nam_model_t** out_model)
{
  if (model_path == nullptr || out_model == nullptr)
  {
    set_error("Invalid argument: model_path and out_model must be non-null.");
    return NAM_STATUS_INVALID_ARGUMENT;
  }

  *out_model = nullptr;
  try
  {
    auto model = std::make_unique<nam_model>();
    model->dsp = nam::get_dsp(std::filesystem::path(model_path));
    *out_model = model.release();
    g_last_error.clear();
    return NAM_STATUS_OK;
  }
  catch (...)
  {
    return handle_exception();
  }
}

void nam_destroy_model(nam_model_t* model)
{
  delete model;
}

nam_status_t nam_reset(nam_model_t* model, double sample_rate, int max_buffer_size)
{
  if (model == nullptr || model->dsp == nullptr || max_buffer_size < 0)
  {
    set_error("Invalid argument: model must be valid and max_buffer_size must be >= 0.");
    return NAM_STATUS_INVALID_ARGUMENT;
  }

  try
  {
    model->dsp->Reset(sample_rate, max_buffer_size);
    g_last_error.clear();
    return NAM_STATUS_OK;
  }
  catch (...)
  {
    return handle_exception();
  }
}

nam_status_t nam_process(nam_model_t* model, nam_sample_t** input, nam_sample_t** output, int num_frames)
{
  if (model == nullptr || model->dsp == nullptr || input == nullptr || output == nullptr || num_frames < 0)
  {
    set_error("Invalid argument: model/input/output must be valid and num_frames must be >= 0.");
    return NAM_STATUS_INVALID_ARGUMENT;
  }

  try
  {
    model->dsp->process(input, output, num_frames);
    g_last_error.clear();
    return NAM_STATUS_OK;
  }
  catch (...)
  {
    return handle_exception();
  }
}

int nam_num_input_channels(const nam_model_t* model)
{
  if (model == nullptr || model->dsp == nullptr)
    return 0;
  return model->dsp->NumInputChannels();
}

int nam_num_output_channels(const nam_model_t* model)
{
  if (model == nullptr || model->dsp == nullptr)
    return 0;
  return model->dsp->NumOutputChannels();
}

double nam_expected_sample_rate(const nam_model_t* model)
{
  if (model == nullptr || model->dsp == nullptr)
    return -1.0;
  return model->dsp->GetExpectedSampleRate();
}

nam_status_t nam_enable_fast_tanh(void)
{
  try
  {
    nam::activations::Activation::enable_fast_tanh();
    g_last_error.clear();
    return NAM_STATUS_OK;
  }
  catch (...)
  {
    return handle_exception();
  }
}

nam_status_t nam_disable_fast_tanh(void)
{
  try
  {
    nam::activations::Activation::disable_fast_tanh();
    g_last_error.clear();
    return NAM_STATUS_OK;
  }
  catch (...)
  {
    return handle_exception();
  }
}

int nam_is_fast_tanh_enabled(void)
{
  return nam::activations::Activation::using_fast_tanh ? 1 : 0;
}

nam_status_t nam_enable_lut(const char* function_name, float min, float max, int n_points)
{
  if (function_name == nullptr || n_points <= 1 || !(min < max))
  {
    set_error("Invalid argument: function_name must be non-null, min < max, and n_points > 1.");
    return NAM_STATUS_INVALID_ARGUMENT;
  }

  try
  {
    nam::activations::Activation::enable_lut(function_name, min, max, static_cast<std::size_t>(n_points));
    g_last_error.clear();
    return NAM_STATUS_OK;
  }
  catch (...)
  {
    return handle_exception();
  }
}

nam_status_t nam_disable_lut(const char* function_name)
{
  if (function_name == nullptr)
  {
    set_error("Invalid argument: function_name must be non-null.");
    return NAM_STATUS_INVALID_ARGUMENT;
  }

  try
  {
    nam::activations::Activation::disable_lut(function_name);
    g_last_error.clear();
    return NAM_STATUS_OK;
  }
  catch (...)
  {
    return handle_exception();
  }
}

const char* nam_get_last_error(void)
{
  return g_last_error.c_str();
}

} // extern "C"
