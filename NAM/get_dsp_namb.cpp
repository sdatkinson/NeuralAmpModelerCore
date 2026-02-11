// Binary .namb loader for NAM models
// No dependency on nlohmann/json

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include "get_dsp_namb.h"

// Architecture headers (no json.hpp dependency needed - we only call constructors)
#include "activations.h"
#include "convnet.h"
#include "lstm.h"
#include "namb_format.h"
#include "wavenet.h"

using namespace nam::namb;

namespace
{

// =============================================================================
// Activation config reading
// =============================================================================

nam::activations::ActivationConfig read_activation_config(BinaryReader& r)
{
  nam::activations::ActivationConfig config;
  config.type = static_cast<nam::activations::ActivationType>(r.read_u8());
  uint8_t param_count = r.read_u8();

  switch (config.type)
  {
    case nam::activations::ActivationType::LeakyReLU:
      if (param_count >= 1)
      {
        config.negative_slope = r.read_f32();
        for (uint8_t i = 1; i < param_count; i++)
          r.read_f32(); // skip extra
      }
      break;

    case nam::activations::ActivationType::PReLU:
      if (param_count == 1)
      {
        config.negative_slope = r.read_f32();
      }
      else if (param_count > 1)
      {
        std::vector<float> slopes;
        slopes.reserve(param_count);
        for (uint8_t i = 0; i < param_count; i++)
          slopes.push_back(r.read_f32());
        config.negative_slopes = std::move(slopes);
      }
      break;

    case nam::activations::ActivationType::LeakyHardtanh:
      if (param_count >= 4)
      {
        config.min_val = r.read_f32();
        config.max_val = r.read_f32();
        config.min_slope = r.read_f32();
        config.max_slope = r.read_f32();
        for (uint8_t i = 4; i < param_count; i++)
          r.read_f32(); // skip extra
      }
      else
      {
        for (uint8_t i = 0; i < param_count; i++)
          r.read_f32(); // skip
      }
      break;

    default:
      // Simple activation - skip any params
      for (uint8_t i = 0; i < param_count; i++)
        r.read_f32();
      break;
  }

  return config;
}

// =============================================================================
// FiLM params reading (4 bytes)
// =============================================================================

nam::wavenet::_FiLMParams read_film_params(BinaryReader& r)
{
  uint8_t flags = r.read_u8();
  r.read_u8(); // reserved
  uint16_t groups = r.read_u16();

  bool active = (flags & 0x01) != 0;
  bool shift = (flags & 0x02) != 0;

  return nam::wavenet::_FiLMParams(active, shift, groups);
}

// =============================================================================
// Metadata parsing
// =============================================================================

struct ParsedMetadata
{
  uint8_t version_major = 0;
  uint8_t version_minor = 0;
  uint8_t version_patch = 0;
  uint8_t meta_flags = 0;
  double sample_rate = -1.0;
  double loudness = 0.0;
  double input_level = 0.0;
  double output_level = 0.0;
};

ParsedMetadata read_metadata_block(BinaryReader& r)
{
  ParsedMetadata m;
  m.version_major = r.read_u8();
  m.version_minor = r.read_u8();
  m.version_patch = r.read_u8();
  m.meta_flags = r.read_u8();
  m.sample_rate = r.read_f64();
  m.loudness = r.read_f64();
  m.input_level = r.read_f64();
  m.output_level = r.read_f64();
  r.skip(12); // reserved
  return m;
}

// =============================================================================
// Model construction (recursive for condition_dsp)
// =============================================================================

// Result of loading a model: the DSP object and how many weights were consumed
struct LoadResult
{
  std::unique_ptr<nam::DSP> dsp;
  size_t weights_consumed;
};

// Forward declaration
LoadResult load_model(BinaryReader& r, const float* weights, size_t weight_count, const ParsedMetadata& meta);

// --- Linear ---

LoadResult load_linear(BinaryReader& r, const float* weights, size_t weight_count, double sample_rate)
{
  int32_t receptive_field = r.read_i32();
  bool bias = r.read_u8() != 0;
  int in_channels = r.read_u8();
  int out_channels = r.read_u8();
  r.read_u8(); // reserved

  std::vector<float> w(weights, weights + weight_count);
  std::unique_ptr<nam::DSP> dsp =
    std::make_unique<nam::Linear>(in_channels, out_channels, receptive_field, bias, w, sample_rate);
  LoadResult result;
  result.dsp = std::move(dsp);
  result.weights_consumed = weight_count;
  return result;
}

// --- LSTM ---

LoadResult load_lstm(BinaryReader& r, const float* weights, size_t weight_count, double sample_rate)
{
  uint16_t num_layers = r.read_u16();
  uint16_t input_size = r.read_u16();
  uint16_t hidden_size = r.read_u16();
  uint8_t in_channels = r.read_u8();
  uint8_t out_channels = r.read_u8();
  r.skip(2); // reserved

  std::vector<float> w(weights, weights + weight_count);
  std::unique_ptr<nam::DSP> dsp = std::make_unique<nam::lstm::LSTM>(in_channels, out_channels, num_layers, input_size,
                                                                    hidden_size, w, sample_rate);
  LoadResult result;
  result.dsp = std::move(dsp);
  result.weights_consumed = weight_count;
  return result;
}

// --- ConvNet ---

LoadResult load_convnet(BinaryReader& r, const float* weights, size_t weight_count, double sample_rate)
{
  uint16_t channels = r.read_u16();
  bool batchnorm = r.read_u8() != 0;
  uint8_t num_dilations = r.read_u8();
  uint16_t groups = r.read_u16();
  uint8_t in_channels = r.read_u8();
  uint8_t out_channels = r.read_u8();

  nam::activations::ActivationConfig activation_config = read_activation_config(r);

  std::vector<int> dilations;
  dilations.reserve(num_dilations);
  for (int i = 0; i < num_dilations; i++)
    dilations.push_back(r.read_i32());

  std::vector<float> w(weights, weights + weight_count);
  std::unique_ptr<nam::DSP> dsp = std::make_unique<nam::convnet::ConvNet>(
    in_channels, out_channels, channels, dilations, batchnorm, activation_config, w, sample_rate, groups);
  LoadResult result;
  result.dsp = std::move(dsp);
  result.weights_consumed = weight_count;
  return result;
}

// --- WaveNet ---

LoadResult load_wavenet(BinaryReader& r, const float* weights, size_t weight_count, const ParsedMetadata& meta)
{
  uint8_t in_channels = r.read_u8();
  uint8_t has_head = r.read_u8();
  uint8_t num_layer_arrays = r.read_u8();
  uint8_t has_condition_dsp = r.read_u8();

  bool with_head = (has_head != 0);
  double sample_rate = meta.sample_rate;

  // Condition DSP
  std::unique_ptr<nam::DSP> condition_dsp;
  size_t cdsp_weights_consumed = 0;

  if (has_condition_dsp)
  {
    uint32_t cdsp_weight_count = r.read_u32();

    // Read condition DSP metadata (48 bytes)
    ParsedMetadata cdsp_meta = read_metadata_block(r);

    // Load condition DSP model recursively
    LoadResult cdsp_result = load_model(r, weights, cdsp_weight_count, cdsp_meta);
    condition_dsp = std::move(cdsp_result.dsp);
    cdsp_weights_consumed = cdsp_result.weights_consumed;

    // Apply metadata to condition DSP
    if (cdsp_meta.meta_flags & META_HAS_LOUDNESS)
      condition_dsp->SetLoudness(cdsp_meta.loudness);
    if (cdsp_meta.meta_flags & META_HAS_INPUT_LEVEL)
      condition_dsp->SetInputLevel(cdsp_meta.input_level);
    if (cdsp_meta.meta_flags & META_HAS_OUTPUT_LEVEL)
      condition_dsp->SetOutputLevel(cdsp_meta.output_level);
    condition_dsp->prewarm();
  }

  // Parse layer array params
  std::vector<nam::wavenet::LayerArrayParams> layer_array_params;
  for (int la = 0; la < num_layer_arrays; la++)
  {
    uint16_t input_size = r.read_u16();
    uint16_t condition_size = r.read_u16();
    uint16_t head_size = r.read_u16();
    uint16_t la_channels = r.read_u16();
    uint16_t bottleneck = r.read_u16();
    uint16_t kernel_size = r.read_u16();

    bool head_bias = r.read_u8() != 0;
    uint8_t num_dilations = r.read_u8();
    uint16_t groups_input = r.read_u16();
    uint16_t groups_input_mixin = r.read_u16();

    // layer1x1 (4 bytes)
    bool layer1x1_active = r.read_u8() != 0;
    uint16_t layer1x1_groups = r.read_u16();
    r.read_u8(); // reserved

    // head1x1 (6 bytes)
    bool head1x1_active = r.read_u8() != 0;
    uint16_t head1x1_out_channels = r.read_u16();
    uint16_t head1x1_groups = r.read_u16();
    r.read_u8(); // reserved

    // 8 FiLM params (32 bytes)
    nam::wavenet::_FiLMParams conv_pre_film = read_film_params(r);
    nam::wavenet::_FiLMParams conv_post_film = read_film_params(r);
    nam::wavenet::_FiLMParams input_mixin_pre_film = read_film_params(r);
    nam::wavenet::_FiLMParams input_mixin_post_film = read_film_params(r);
    nam::wavenet::_FiLMParams activation_pre_film = read_film_params(r);
    nam::wavenet::_FiLMParams activation_post_film = read_film_params(r);
    nam::wavenet::_FiLMParams layer1x1_post_film = read_film_params(r);
    nam::wavenet::_FiLMParams head1x1_post_film = read_film_params(r);

    // Dilations [N * int32]
    std::vector<int> dilations;
    dilations.reserve(num_dilations);
    for (int i = 0; i < num_dilations; i++)
      dilations.push_back(r.read_i32());

    // Activation configs [N * variable]
    std::vector<nam::activations::ActivationConfig> activation_configs;
    activation_configs.reserve(num_dilations);
    for (int i = 0; i < num_dilations; i++)
      activation_configs.push_back(read_activation_config(r));

    // Gating modes [N * uint8]
    std::vector<nam::wavenet::GatingMode> gating_modes;
    gating_modes.reserve(num_dilations);
    for (int i = 0; i < num_dilations; i++)
    {
      uint8_t gm = r.read_u8();
      switch (gm)
      {
        case GATING_GATED: gating_modes.push_back(nam::wavenet::GatingMode::GATED); break;
        case GATING_BLENDED: gating_modes.push_back(nam::wavenet::GatingMode::BLENDED); break;
        default: gating_modes.push_back(nam::wavenet::GatingMode::NONE); break;
      }
    }

    // Secondary activation configs [N * variable]
    std::vector<nam::activations::ActivationConfig> secondary_activation_configs;
    secondary_activation_configs.reserve(num_dilations);
    for (int i = 0; i < num_dilations; i++)
      secondary_activation_configs.push_back(read_activation_config(r));

    nam::wavenet::Layer1x1Params layer1x1_params(layer1x1_active, layer1x1_groups);
    nam::wavenet::Head1x1Params head1x1_params(head1x1_active, head1x1_out_channels, head1x1_groups);

    layer_array_params.emplace_back(input_size, condition_size, head_size, la_channels, bottleneck, kernel_size,
                                    std::move(dilations), std::move(activation_configs), std::move(gating_modes),
                                    head_bias, groups_input, groups_input_mixin, layer1x1_params, head1x1_params,
                                    std::move(secondary_activation_configs), conv_pre_film, conv_post_film,
                                    input_mixin_pre_film, input_mixin_post_film, activation_pre_film,
                                    activation_post_film, layer1x1_post_film, head1x1_post_film);
  }

  // Build wavenet weights (excluding condition DSP weights which were consumed earlier)
  const float* wavenet_weights_ptr = weights + cdsp_weights_consumed;
  size_t wavenet_weight_count = weight_count - cdsp_weights_consumed;
  std::vector<float> wavenet_weights(wavenet_weights_ptr, wavenet_weights_ptr + wavenet_weight_count);

  // head_scale is the last weight value, but the constructor takes it as a param too
  // (it gets overridden by set_weights_). Pass 0.0f; set_weights_ will set the correct value.
  std::unique_ptr<nam::DSP> dsp = std::make_unique<nam::wavenet::WaveNet>(
    in_channels, layer_array_params, 0.0f, with_head, std::move(wavenet_weights), std::move(condition_dsp),
    sample_rate);

  LoadResult result;
  result.dsp = std::move(dsp);
  result.weights_consumed = weight_count;
  return result;
}

// =============================================================================
// Dispatch to architecture-specific loader
// =============================================================================

LoadResult load_model(BinaryReader& r, const float* weights, size_t weight_count, const ParsedMetadata& meta)
{
  uint8_t arch = r.read_u8();
  r.read_u8(); // reserved
  r.read_u16(); // config_size (not needed - we parse sequentially)

  switch (arch)
  {
    case ARCH_LINEAR: return load_linear(r, weights, weight_count, meta.sample_rate);
    case ARCH_LSTM: return load_lstm(r, weights, weight_count, meta.sample_rate);
    case ARCH_CONVNET: return load_convnet(r, weights, weight_count, meta.sample_rate);
    case ARCH_WAVENET: return load_wavenet(r, weights, weight_count, meta);
    default: throw std::runtime_error("NAMB: unknown architecture ID " + std::to_string(arch));
  }
}

} // anonymous namespace

// =============================================================================
// Public API
// =============================================================================

std::unique_ptr<nam::DSP> nam::get_dsp_namb(const uint8_t* data, size_t size)
{
  if (size < FILE_HEADER_SIZE + METADATA_BLOCK_SIZE)
    throw std::runtime_error("NAMB: file too small");

  BinaryReader header_reader(data, FILE_HEADER_SIZE);

  // Validate magic
  uint32_t magic = header_reader.read_u32();
  if (magic != MAGIC)
    throw std::runtime_error("NAMB: invalid magic number");

  // Validate format version
  uint16_t version = header_reader.read_u16();
  if (version != FORMAT_VERSION)
    throw std::runtime_error("NAMB: unsupported format version " + std::to_string(version));

  header_reader.read_u16(); // flags
  uint32_t total_file_size = header_reader.read_u32();
  uint32_t weights_offset = header_reader.read_u32();
  uint32_t total_weight_count = header_reader.read_u32();
  header_reader.read_u32(); // model_block_size
  uint32_t stored_checksum = header_reader.read_u32();

  // Validate file size
  if (size < total_file_size)
    throw std::runtime_error("NAMB: file truncated (expected " + std::to_string(total_file_size) + " bytes, got "
                             + std::to_string(size) + ")");

  // Validate CRC32
  uint32_t computed_checksum = compute_file_crc32(data, total_file_size);
  if (computed_checksum != stored_checksum)
    throw std::runtime_error("NAMB: checksum mismatch");

  // Validate weights section
  size_t expected_weights_end = weights_offset + total_weight_count * sizeof(float);
  if (expected_weights_end > total_file_size)
    throw std::runtime_error("NAMB: weights extend beyond file");

  // Read metadata block (at offset 32)
  BinaryReader meta_reader(data + FILE_HEADER_SIZE, METADATA_BLOCK_SIZE);
  ParsedMetadata meta = read_metadata_block(meta_reader);

  // Verify config version
  std::string version_str = std::to_string(meta.version_major) + "." + std::to_string(meta.version_minor) + "."
                            + std::to_string(meta.version_patch);
  nam::verify_config_version(version_str);

  // Get weight data pointer
  const float* weights = reinterpret_cast<const float*>(data + weights_offset);

  // Read model block (at offset 80)
  size_t model_data_size = weights_offset - MODEL_BLOCK_OFFSET;
  BinaryReader model_reader(data + MODEL_BLOCK_OFFSET, model_data_size);

  // Load the model
  LoadResult result = load_model(model_reader, weights, total_weight_count, meta);

  // Apply metadata
  if (meta.meta_flags & META_HAS_LOUDNESS)
    result.dsp->SetLoudness(meta.loudness);
  if (meta.meta_flags & META_HAS_INPUT_LEVEL)
    result.dsp->SetInputLevel(meta.input_level);
  if (meta.meta_flags & META_HAS_OUTPUT_LEVEL)
    result.dsp->SetOutputLevel(meta.output_level);

  result.dsp->prewarm();

  return std::move(result.dsp);
}

std::unique_ptr<nam::DSP> nam::get_dsp_namb(const std::filesystem::path& filename)
{
  if (!std::filesystem::exists(filename))
    throw std::runtime_error("NAMB file doesn't exist: " + filename.string());

  // Read entire file into memory
  std::ifstream file(filename, std::ios::binary | std::ios::ate);
  if (!file.is_open())
    throw std::runtime_error("Cannot open NAMB file: " + filename.string());

  size_t file_size = file.tellg();
  file.seekg(0, std::ios::beg);

  std::vector<uint8_t> data(file_size);
  file.read(reinterpret_cast<char*>(data.data()), file_size);
  file.close();

  return get_dsp_namb(data.data(), data.size());
}
