#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
  #if defined(NAM_BUILD_SHARED)
    #define NAM_API __declspec(dllexport)
  #elif defined(NAM_USE_SHARED)
    #define NAM_API __declspec(dllimport)
  #else
    #define NAM_API
  #endif
#elif defined(__GNUC__) || defined(__clang__)
  #define NAM_API __attribute__((visibility("default")))
#else
  #define NAM_API
#endif

#ifdef NAM_SAMPLE_FLOAT
typedef float nam_sample_t;
#else
typedef double nam_sample_t;
#endif

typedef struct nam_model nam_model_t;

typedef enum nam_status
{
  NAM_STATUS_OK = 0,
  NAM_STATUS_INVALID_ARGUMENT = 1,
  NAM_STATUS_EXCEPTION = 2
} nam_status_t;

// Returns a model handle loaded from a .nam file path.
NAM_API nam_status_t nam_create_model_from_file(const char* model_path, nam_model_t** out_model);

// Releases a model handle created by nam_create_model_from_file.
NAM_API void nam_destroy_model(nam_model_t* model);

// Resets DSP state for the given runtime sample rate and max block size.
NAM_API nam_status_t nam_reset(nam_model_t* model, double sample_rate, int max_buffer_size);

// Processes deinterleaved channels: input[channel][frame], output[channel][frame].
NAM_API nam_status_t nam_process(nam_model_t* model, nam_sample_t** input, nam_sample_t** output,
                                 int num_frames);

NAM_API int nam_num_input_channels(const nam_model_t* model);
NAM_API int nam_num_output_channels(const nam_model_t* model);
NAM_API double nam_expected_sample_rate(const nam_model_t* model);

// Enables/disables fast tanh approximation globally for NAM activations.
NAM_API nam_status_t nam_enable_fast_tanh(void);
NAM_API nam_status_t nam_disable_fast_tanh(void);
NAM_API int nam_is_fast_tanh_enabled(void);
NAM_API nam_status_t nam_enable_lut(const char* function_name, float min, float max, int n_points);
NAM_API nam_status_t nam_disable_lut(const char* function_name);

// Thread-local error message set when a NAM_STATUS_EXCEPTION is returned.
NAM_API const char* nam_get_last_error(void);

#ifdef __cplusplus
}
#endif
