// nam2namb: Convert .nam (JSON) models to .namb (compact binary) format
//
// Usage: nam2namb input.nam [output.namb]
// If output is not specified, replaces .nam extension with .namb

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "NAM/activations.h"
#include "NAM/namb_format.h"
#include "json.hpp"

using json = nlohmann::json;
using namespace nam::namb;

// =============================================================================
// Architecture name to ID mapping
// =============================================================================

static uint8_t architecture_id(const std::string& name)
{
  if (name == "Linear")
    return ARCH_LINEAR;
  if (name == "ConvNet")
    return ARCH_CONVNET;
  if (name == "LSTM")
    return ARCH_LSTM;
  if (name == "WaveNet")
    return ARCH_WAVENET;
  throw std::runtime_error("Unknown architecture: " + name);
}

// =============================================================================
// Activation config serialization
// =============================================================================

static void write_activation_config(BinaryWriter& w, const json& activation_json)
{
  auto config = nam::activations::ActivationConfig::from_json(activation_json);
  w.write_u8(static_cast<uint8_t>(config.type));

  // Collect parameters
  std::vector<float> params;
  if (config.type == nam::activations::ActivationType::LeakyReLU)
  {
    if (config.negative_slope.has_value())
      params.push_back(config.negative_slope.value());
  }
  else if (config.type == nam::activations::ActivationType::PReLU)
  {
    if (config.negative_slopes.has_value())
    {
      params = config.negative_slopes.value();
    }
    else if (config.negative_slope.has_value())
    {
      params.push_back(config.negative_slope.value());
    }
  }
  else if (config.type == nam::activations::ActivationType::LeakyHardtanh)
  {
    params.push_back(config.min_val.value_or(-1.0f));
    params.push_back(config.max_val.value_or(1.0f));
    params.push_back(config.min_slope.value_or(0.01f));
    params.push_back(config.max_slope.value_or(0.01f));
  }

  if (params.size() > 255)
    throw std::runtime_error("Activation has too many parameters (max 255)");

  w.write_u8(static_cast<uint8_t>(params.size()));
  for (float p : params)
    w.write_f32(p);
}

// =============================================================================
// FiLM params serialization (4 bytes)
// =============================================================================

static void write_film_params(BinaryWriter& w, const json& layer_config, const std::string& key)
{
  if (layer_config.find(key) == layer_config.end() || layer_config[key] == false)
  {
    // Inactive FiLM
    w.write_u8(0); // flags: not active
    w.write_u8(0); // reserved
    w.write_u16(1); // groups (default)
    return;
  }

  const json& film = layer_config[key];
  bool active = film.value("active", true);
  bool shift = film.value("shift", true);
  int groups = film.value("groups", 1);

  uint8_t flags = 0;
  if (active)
    flags |= 0x01;
  if (shift)
    flags |= 0x02;

  w.write_u8(flags);
  w.write_u8(0); // reserved
  w.write_u16(static_cast<uint16_t>(groups));
}

// =============================================================================
// Gating mode parsing (from JSON, same logic as wavenet::Factory)
// =============================================================================

static uint8_t gating_mode_from_string(const std::string& s)
{
  if (s == "gated")
    return GATING_GATED;
  if (s == "blended")
    return GATING_BLENDED;
  if (s == "none")
    return GATING_NONE;
  throw std::runtime_error("Invalid gating_mode: " + s);
}

// =============================================================================
// Metadata block serialization (48 bytes)
// =============================================================================

static void write_metadata_block(BinaryWriter& w, const json& model_json)
{
  // Parse version
  std::string version_str = model_json["version"].get<std::string>();
  int major = 0, minor = 0, patch = 0;
  sscanf(version_str.c_str(), "%d.%d.%d", &major, &minor, &patch);

  w.write_u8(static_cast<uint8_t>(major));
  w.write_u8(static_cast<uint8_t>(minor));
  w.write_u8(static_cast<uint8_t>(patch));

  // Meta flags and values
  uint8_t meta_flags = 0;
  double loudness = 0.0, input_level = 0.0, output_level = 0.0;

  if (model_json.find("metadata") != model_json.end() && !model_json["metadata"].is_null())
  {
    const json& meta = model_json["metadata"];
    if (meta.find("loudness") != meta.end() && !meta["loudness"].is_null())
    {
      meta_flags |= META_HAS_LOUDNESS;
      loudness = meta["loudness"].get<double>();
    }
    if (meta.find("input_level_dbu") != meta.end() && !meta["input_level_dbu"].is_null())
    {
      meta_flags |= META_HAS_INPUT_LEVEL;
      input_level = meta["input_level_dbu"].get<double>();
    }
    if (meta.find("output_level_dbu") != meta.end() && !meta["output_level_dbu"].is_null())
    {
      meta_flags |= META_HAS_OUTPUT_LEVEL;
      output_level = meta["output_level_dbu"].get<double>();
    }
  }
  w.write_u8(meta_flags);

  // Sample rate
  double sample_rate = -1.0;
  if (model_json.find("sample_rate") != model_json.end())
    sample_rate = model_json["sample_rate"].get<double>();
  w.write_f64(sample_rate);

  w.write_f64(loudness);
  w.write_f64(input_level);
  w.write_f64(output_level);

  // Reserved (12 bytes)
  w.write_zeros(12);
}

// =============================================================================
// Collect all weights recursively (condition_dsp weights first)
// =============================================================================

static void collect_weights(const json& model_json, std::vector<float>& all_weights)
{
  // If this is a WaveNet with condition_dsp, collect condition_dsp weights first
  const std::string arch = model_json["architecture"].get<std::string>();
  if (arch == "WaveNet")
  {
    const json& config = model_json["config"];
    if (config.find("condition_dsp") != config.end())
    {
      const json& cdsp = config["condition_dsp"];
      collect_weights(cdsp, all_weights);
    }
  }

  // Then add this model's weights
  if (model_json.find("weights") != model_json.end())
  {
    const auto& weights = model_json["weights"];
    for (const auto& w : weights)
      all_weights.push_back(w.get<float>());
  }
}

// =============================================================================
// Model block serialization (recursive for condition_dsp)
// =============================================================================

// Forward declaration
static void write_model_block(BinaryWriter& w, const json& model_json);

static void write_linear_config(BinaryWriter& w, const json& config)
{
  int32_t receptive_field = config["receptive_field"].get<int>();
  bool bias = config["bias"].get<bool>();
  int in_channels = config.value("in_channels", 1);
  int out_channels = config.value("out_channels", 1);

  w.write_i32(receptive_field);
  w.write_u8(bias ? 1 : 0);
  w.write_u8(static_cast<uint8_t>(in_channels));
  w.write_u8(static_cast<uint8_t>(out_channels));
  w.write_u8(0); // reserved
}

static void write_lstm_config(BinaryWriter& w, const json& config)
{
  w.write_u16(static_cast<uint16_t>(config["num_layers"].get<int>()));
  w.write_u16(static_cast<uint16_t>(config["input_size"].get<int>()));
  w.write_u16(static_cast<uint16_t>(config["hidden_size"].get<int>()));
  w.write_u8(static_cast<uint8_t>(config.value("in_channels", 1)));
  w.write_u8(static_cast<uint8_t>(config.value("out_channels", 1)));
  w.write_u16(0); // reserved
}

static void write_convnet_config(BinaryWriter& w, const json& config)
{
  int channels = config["channels"].get<int>();
  bool batchnorm = config["batchnorm"].get<bool>();
  const auto& dilations = config["dilations"];
  int groups = config.value("groups", 1);
  int in_channels = config.value("in_channels", 1);
  int out_channels = config.value("out_channels", 1);

  w.write_u16(static_cast<uint16_t>(channels));
  w.write_u8(batchnorm ? 1 : 0);
  w.write_u8(static_cast<uint8_t>(dilations.size()));
  w.write_u16(static_cast<uint16_t>(groups));
  w.write_u8(static_cast<uint8_t>(in_channels));
  w.write_u8(static_cast<uint8_t>(out_channels));

  // Activation config
  write_activation_config(w, config["activation"]);

  // Dilations
  for (const auto& d : dilations)
    w.write_i32(d.get<int>());
}

static void write_wavenet_config(BinaryWriter& w, const json& model_json)
{
  const json& config = model_json["config"];

  int in_channels = config.value("in_channels", 1);
  bool with_head = config.find("head") != config.end() && !config["head"].is_null();
  size_t num_layer_arrays = config["layers"].size();
  bool has_condition_dsp = config.find("condition_dsp") != config.end();

  w.write_u8(static_cast<uint8_t>(in_channels));
  w.write_u8(with_head ? 1 : 0);
  w.write_u8(static_cast<uint8_t>(num_layer_arrays));
  w.write_u8(has_condition_dsp ? 1 : 0);

  // Condition DSP (if present)
  if (has_condition_dsp)
  {
    const json& cdsp_json = config["condition_dsp"];

    // Count condition DSP weights (recursively)
    std::vector<float> cdsp_weights;
    collect_weights(cdsp_json, cdsp_weights);
    w.write_u32(static_cast<uint32_t>(cdsp_weights.size()));

    // Condition DSP metadata (48 bytes)
    write_metadata_block(w, cdsp_json);

    // Condition DSP model block (recursive)
    write_model_block(w, cdsp_json);
  }

  // Layer array params
  for (size_t la = 0; la < num_layer_arrays; la++)
  {
    const json& layer = config["layers"][la];

    int layer_channels = layer["channels"].get<int>();
    int bottleneck = layer.value("bottleneck", layer_channels);
    const auto& dilations = layer["dilations"];
    size_t num_dilations = dilations.size();

    w.write_u16(static_cast<uint16_t>(layer["input_size"].get<int>()));
    w.write_u16(static_cast<uint16_t>(layer["condition_size"].get<int>()));
    w.write_u16(static_cast<uint16_t>(layer["head_size"].get<int>()));
    w.write_u16(static_cast<uint16_t>(layer_channels));
    w.write_u16(static_cast<uint16_t>(bottleneck));
    w.write_u16(static_cast<uint16_t>(layer["kernel_size"].get<int>()));

    w.write_u8(layer["head_bias"].get<bool>() ? 1 : 0);
    w.write_u8(static_cast<uint8_t>(num_dilations));

    int groups_input = layer.value("groups_input", 1);
    int groups_input_mixin = layer.value("groups_input_mixin", 1);
    w.write_u16(static_cast<uint16_t>(groups_input));
    w.write_u16(static_cast<uint16_t>(groups_input_mixin));

    // layer1x1 params (4 bytes)
    bool layer1x1_active = true;
    int layer1x1_groups = 1;
    if (layer.find("layer1x1") != layer.end())
    {
      layer1x1_active = layer["layer1x1"]["active"].get<bool>();
      layer1x1_groups = layer["layer1x1"]["groups"].get<int>();
    }
    w.write_u8(layer1x1_active ? 1 : 0);
    w.write_u16(static_cast<uint16_t>(layer1x1_groups));
    w.write_u8(0); // reserved

    // head1x1 params (6 bytes)
    bool head1x1_active = false;
    int head1x1_out_channels = layer_channels;
    int head1x1_groups = 1;
    if (layer.find("head1x1") != layer.end())
    {
      head1x1_active = layer["head1x1"]["active"].get<bool>();
      head1x1_out_channels = layer["head1x1"]["out_channels"].get<int>();
      head1x1_groups = layer["head1x1"]["groups"].get<int>();
    }
    w.write_u8(head1x1_active ? 1 : 0);
    w.write_u16(static_cast<uint16_t>(head1x1_out_channels));
    w.write_u16(static_cast<uint16_t>(head1x1_groups));
    w.write_u8(0); // reserved

    // 8 FiLM params (32 bytes)
    write_film_params(w, layer, "conv_pre_film");
    write_film_params(w, layer, "conv_post_film");
    write_film_params(w, layer, "input_mixin_pre_film");
    write_film_params(w, layer, "input_mixin_post_film");
    write_film_params(w, layer, "activation_pre_film");
    write_film_params(w, layer, "activation_post_film");
    write_film_params(w, layer, "layer1x1_post_film");
    write_film_params(w, layer, "head1x1_post_film");

    // Dilations [num_dilations * int32]
    for (const auto& d : dilations)
      w.write_i32(d.get<int>());

    // Activation configs [num_dilations * variable]
    if (layer["activation"].is_array())
    {
      for (const auto& act : layer["activation"])
        write_activation_config(w, act);
    }
    else
    {
      // Single activation - write it N times
      for (size_t i = 0; i < num_dilations; i++)
        write_activation_config(w, layer["activation"]);
    }

    // Gating modes [num_dilations * uint8]
    if (layer.find("gating_mode") != layer.end())
    {
      if (layer["gating_mode"].is_array())
      {
        for (const auto& gm : layer["gating_mode"])
          w.write_u8(gating_mode_from_string(gm.get<std::string>()));
      }
      else
      {
        uint8_t mode = gating_mode_from_string(layer["gating_mode"].get<std::string>());
        for (size_t i = 0; i < num_dilations; i++)
          w.write_u8(mode);
      }
    }
    else if (layer.find("gated") != layer.end())
    {
      // Backward compatibility
      uint8_t mode = layer["gated"].get<bool>() ? GATING_GATED : GATING_NONE;
      for (size_t i = 0; i < num_dilations; i++)
        w.write_u8(mode);
    }
    else
    {
      for (size_t i = 0; i < num_dilations; i++)
        w.write_u8(GATING_NONE);
    }

    // Secondary activation configs [num_dilations * variable]
    // Parse gating modes to determine which layers need secondary activations
    std::vector<uint8_t> gating_modes;
    if (layer.find("gating_mode") != layer.end())
    {
      if (layer["gating_mode"].is_array())
      {
        for (const auto& gm : layer["gating_mode"])
          gating_modes.push_back(gating_mode_from_string(gm.get<std::string>()));
      }
      else
      {
        uint8_t mode = gating_mode_from_string(layer["gating_mode"].get<std::string>());
        gating_modes.resize(num_dilations, mode);
      }
    }
    else if (layer.find("gated") != layer.end())
    {
      uint8_t mode = layer["gated"].get<bool>() ? GATING_GATED : GATING_NONE;
      gating_modes.resize(num_dilations, mode);
    }
    else
    {
      gating_modes.resize(num_dilations, GATING_NONE);
    }

    for (size_t i = 0; i < num_dilations; i++)
    {
      if (gating_modes[i] != GATING_NONE)
      {
        // Need a secondary activation
        if (layer.find("secondary_activation") != layer.end())
        {
          if (layer["secondary_activation"].is_array())
          {
            write_activation_config(w, layer["secondary_activation"][i]);
          }
          else
          {
            write_activation_config(w, layer["secondary_activation"]);
          }
        }
        else
        {
          // Default: Sigmoid
          w.write_u8(static_cast<uint8_t>(nam::activations::ActivationType::Sigmoid));
          w.write_u8(0); // no params
        }
      }
      else
      {
        // NONE mode - write an empty activation config (identity)
        w.write_u8(static_cast<uint8_t>(nam::activations::ActivationType::Tanh)); // type doesn't matter
        w.write_u8(0); // no params
      }
    }
  }
}

static void write_model_block(BinaryWriter& w, const json& model_json)
{
  std::string arch_name = model_json["architecture"].get<std::string>();
  uint8_t arch = architecture_id(arch_name);

  // Write model block header
  w.write_u8(arch);
  w.write_u8(0); // reserved

  // Placeholder for config_size (will backpatch)
  size_t config_size_offset = w.position();
  w.write_u16(0);

  size_t config_start = w.position();

  const json& config = model_json["config"];

  switch (arch)
  {
    case ARCH_LINEAR: write_linear_config(w, config); break;
    case ARCH_CONVNET: write_convnet_config(w, config); break;
    case ARCH_LSTM: write_lstm_config(w, config); break;
    case ARCH_WAVENET: write_wavenet_config(w, model_json); break;
    default: throw std::runtime_error("Unknown architecture ID");
  }

  // Backpatch config_size (uint16 at config_size_offset)
  size_t config_size = w.position() - config_start;
  if (config_size > 65535)
    throw std::runtime_error("Config too large for uint16");
  uint16_t cs = static_cast<uint16_t>(config_size);
  std::memcpy(w.data() + config_size_offset, &cs, 2);
}

// =============================================================================
// Main conversion function
// =============================================================================

static std::vector<uint8_t> convert_nam_to_namb(const json& model_json)
{
  BinaryWriter w;

  // ---- File Header (32 bytes) ----
  w.write_u32(MAGIC);
  w.write_u16(FORMAT_VERSION);
  w.write_u16(0); // flags

  size_t total_file_size_offset = w.position();
  w.write_u32(0); // total_file_size (backpatch)

  size_t weights_offset_pos = w.position();
  w.write_u32(0); // weights_offset (backpatch)

  size_t total_weight_count_offset = w.position();
  w.write_u32(0); // total_weight_count (backpatch)

  size_t model_block_size_offset = w.position();
  w.write_u32(0); // model_block_size (backpatch)

  w.write_u32(0); // checksum (backpatch)
  w.write_u32(0); // reserved

  // ---- Metadata Block (48 bytes at offset 32) ----
  write_metadata_block(w, model_json);

  // ---- Model Block (variable, at offset 80) ----
  size_t model_block_start = w.position();
  write_model_block(w, model_json);
  size_t model_block_size = w.position() - model_block_start;

  // ---- Padding to align weights to 4 bytes ----
  while (w.position() % 4 != 0)
    w.write_u8(0);

  size_t weights_offset = w.position();

  // ---- Weight Data ----
  std::vector<float> all_weights;
  collect_weights(model_json, all_weights);

  for (float wt : all_weights)
    w.write_f32(wt);

  // ---- Backpatch header fields ----
  uint32_t total_file_size = static_cast<uint32_t>(w.size());
  w.set_u32(total_file_size_offset, total_file_size);
  w.set_u32(weights_offset_pos, static_cast<uint32_t>(weights_offset));
  w.set_u32(total_weight_count_offset, static_cast<uint32_t>(all_weights.size()));
  w.set_u32(model_block_size_offset, static_cast<uint32_t>(model_block_size));

  // ---- Compute and write CRC32 ----
  uint32_t checksum = compute_file_crc32(w.data(), w.size());
  w.set_u32(24, checksum); // checksum at offset 24

  return w.buffer();
}

// =============================================================================
// Entry point
// =============================================================================

int main(int argc, char* argv[])
{
  if (argc < 2)
  {
    std::cerr << "Usage: nam2namb input.nam [output.namb]" << std::endl;
    return 1;
  }

  std::filesystem::path input_path(argv[1]);
  std::filesystem::path output_path;

  if (argc >= 3)
  {
    output_path = argv[2];
  }
  else
  {
    output_path = input_path;
    output_path.replace_extension(".namb");
  }

  // Read input JSON
  std::ifstream input_file(input_path);
  if (!input_file.is_open())
  {
    std::cerr << "Error: cannot open " << input_path << std::endl;
    return 1;
  }

  json model_json;
  try
  {
    input_file >> model_json;
  }
  catch (const std::exception& e)
  {
    std::cerr << "Error parsing JSON: " << e.what() << std::endl;
    return 1;
  }
  input_file.close();

  // Convert
  std::vector<uint8_t> namb_data;
  try
  {
    namb_data = convert_nam_to_namb(model_json);
  }
  catch (const std::exception& e)
  {
    std::cerr << "Error converting: " << e.what() << std::endl;
    return 1;
  }

  // Write output
  std::ofstream output_file(output_path, std::ios::binary);
  if (!output_file.is_open())
  {
    std::cerr << "Error: cannot create " << output_path << std::endl;
    return 1;
  }
  output_file.write(reinterpret_cast<const char*>(namb_data.data()), namb_data.size());
  output_file.close();

  // Report
  size_t json_size = std::filesystem::file_size(input_path);
  size_t namb_size = namb_data.size();
  double reduction = 100.0 * (1.0 - (double)namb_size / (double)json_size);

  std::cout << input_path.filename().string() << " -> " << output_path.filename().string() << std::endl;
  std::cout << "  JSON: " << json_size << " bytes" << std::endl;
  std::cout << "  NAMB: " << namb_size << " bytes" << std::endl;
  std::cout << "  Reduction: " << std::fixed << std::setprecision(1) << reduction << "%" << std::endl;

  return 0;
}
