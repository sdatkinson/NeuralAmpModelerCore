// memory_usage.cpp — Report total memory required to host a NAM model at runtime.
//
// Usage: memory_usage <model_path> [--buffer-size N]
//
// Parses the .nam JSON config and computes weight memory (learned parameters stored
// in Eigen matrices/vectors) and buffer memory (intermediate computation/state that
// depends on maxBufferSize) without instantiating the model.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "json.hpp"

using json = nlohmann::json;

static constexpr int DEFAULT_BUFFER_SIZE = 2048;
static constexpr long INPUT_BUFFER_SAFETY_FACTOR = 32;

// ─── Result accumulator ─────────────────────────────────────────────────────

struct MemoryResult
{
  size_t weight_bytes = 0;
  size_t buffer_bytes = 0;

  void add_weights(size_t floats) { weight_bytes += floats * sizeof(float); }
  void add_buffers(size_t floats) { buffer_bytes += floats * sizeof(float); }

  MemoryResult& operator+=(const MemoryResult& o)
  {
    weight_bytes += o.weight_bytes;
    buffer_bytes += o.buffer_bytes;
    return *this;
  }
};

// ─── Conv1x1 ────────────────────────────────────────────────────────────────

// Conv1x1 stores either a full (out_channels x in_channels) matrix (possibly
// block-diagonal when grouped), or a depthwise weight vector when groups ==
// in_channels == out_channels.
static MemoryResult conv1x1_memory(int in_ch, int out_ch, bool bias, int groups, int M)
{
  MemoryResult r;
  bool depthwise = (groups == in_ch && in_ch == out_ch);
  if (depthwise)
    r.add_weights(in_ch); // _depthwise_weight(in_ch)
  else
    r.add_weights((size_t)out_ch * in_ch); // _weight(out_ch, in_ch)
  if (bias)
    r.add_weights(out_ch); // _bias(out_ch)
  r.add_buffers((size_t)out_ch * M); // _output(out_ch, M)
  return r;
}

// ─── Conv1D ─────────────────────────────────────────────────────────────────

// Conv1D stores kernel_size weight matrices (each out_ch x in_ch) or depthwise
// vectors, plus a bias vector, a ring buffer, and an output buffer.
static MemoryResult conv1d_memory(int in_ch, int out_ch, int kernel_size, bool bias, int dilation, int groups, int M)
{
  MemoryResult r;
  bool depthwise = (groups == in_ch && in_ch == out_ch);
  if (depthwise)
    r.add_weights((size_t)kernel_size * in_ch); // _depthwise_weight[k](in_ch)
  else
    r.add_weights((size_t)kernel_size * out_ch * in_ch); // _weight[k](out_ch, in_ch)
  if (bias)
    r.add_weights(out_ch); // _bias(out_ch)

  // Ring buffer: storage = (in_ch, 2 * max_lookback + M)
  // max_lookback = (kernel_size - 1) * dilation
  long max_lookback = (kernel_size > 0) ? (long)(kernel_size - 1) * dilation : 0;
  long ring_storage = 2 * max_lookback + M;
  r.add_buffers((size_t)in_ch * ring_storage); // _input_buffer._storage

  // Output buffer: (out_ch, M)
  r.add_buffers((size_t)out_ch * M); // _output

  return r;
}

// ─── FiLM ───────────────────────────────────────────────────────────────────

struct FiLMParams
{
  bool active = false;
  bool shift = true;
  int groups = 1;
};

static MemoryResult film_memory(int condition_dim, int input_dim, const FiLMParams& fp, int M)
{
  if (!fp.active)
    return {};
  MemoryResult r;
  int scale_shift_dim = fp.shift ? 2 * input_dim : input_dim;
  // _cond_to_scale_shift is a Conv1x1(condition_dim -> scale_shift_dim, bias=true, groups)
  r += conv1x1_memory(condition_dim, scale_shift_dim, true, fp.groups, M);
  // _output(input_dim, M)
  r.add_buffers((size_t)input_dim * M);
  return r;
}

// ─── BatchNorm ──────────────────────────────────────────────────────────────

static MemoryResult batchnorm_memory(int dim)
{
  MemoryResult r;
  // Stores scale(dim) + loc(dim) derived from running_mean, running_var, weight, bias, eps
  // The source values are consumed from weights array; only scale + loc are stored at runtime.
  r.add_weights(2 * (size_t)dim);
  return r;
}

// ─── LSTM ───────────────────────────────────────────────────────────────────

static MemoryResult lstm_memory(const json& config)
{
  MemoryResult r;
  int num_layers = config["num_layers"];
  int input_size = config["input_size"];
  int hidden_size = config["hidden_size"];
  int in_channels = config.value("in_channels", 1);
  int out_channels = config.value("out_channels", 1);

  for (int i = 0; i < num_layers; i++)
  {
    int cell_input = (i == 0) ? input_size : hidden_size;
    // _w(4*H, I+H)
    r.add_weights((size_t)4 * hidden_size * (cell_input + hidden_size));
    // _b(4*H)
    r.add_weights(4 * (size_t)hidden_size);
    // _xh(I+H) — stores initial hidden state in the hidden portion
    r.add_weights((size_t)(cell_input + hidden_size));
    // _c(H) — initial cell state
    r.add_weights((size_t)hidden_size);

    // Buffers: _ifgo(4*H)
    r.add_buffers(4 * (size_t)hidden_size);
    // Note: _xh and _c are also modified during inference but they are
    // loaded from weights (initial state), so counted as weights above.
  }

  // _head_weight(out_channels, hidden_size)
  r.add_weights((size_t)out_channels * hidden_size);
  // _head_bias(out_channels)
  r.add_weights(out_channels);

  // Top-level buffers: _input(input_size), _output(out_channels)
  r.add_buffers(input_size);
  r.add_buffers(out_channels);

  return r;
}

// ─── Linear ─────────────────────────────────────────────────────────────────

static MemoryResult linear_memory(const json& config)
{
  MemoryResult r;
  int receptive_field = config["receptive_field"];
  bool bias = config["bias"];
  int in_channels = config.value("in_channels", 1);
  int out_channels = config.value("out_channels", 1);

  // _weight(receptive_field)
  r.add_weights(receptive_field);
  // _bias (scalar float)
  if (bias)
    r.add_weights(1);

  // Buffer base: _input_buffers = in_channels vectors of (32 * receptive_field)
  r.add_buffers((size_t)in_channels * INPUT_BUFFER_SAFETY_FACTOR * receptive_field);
  // _output_buffers: resized per-call, not pre-allocated to a fixed size
  // (depends on num_frames, not maxBufferSize)

  return r;
}

// ─── ConvNet ────────────────────────────────────────────────────────────────

static MemoryResult convnet_memory(const json& config, int M)
{
  MemoryResult r;
  int channels = config["channels"];
  std::vector<int> dilations = config["dilations"];
  bool batchnorm = config["batchnorm"];
  int groups = config.value("groups", 1);
  int in_channels = config.value("in_channels", 1);
  int out_channels = config.value("out_channels", 1);

  int max_dilation = *std::max_element(dilations.begin(), dilations.end());

  // Buffer base class: _input_buffers = in_channels * (32 * max_dilation)
  int receptive_field = max_dilation; // passed to Buffer as receptive_field
  r.add_buffers((size_t)in_channels * INPUT_BUFFER_SAFETY_FACTOR * receptive_field);

  // ConvNet blocks
  for (size_t i = 0; i < dilations.size(); i++)
  {
    int block_in = (i == 0) ? in_channels : channels;
    int block_out = channels;
    // Conv1D with kernel_size=2, bias=!batchnorm
    r += conv1d_memory(block_in, block_out, 2, !batchnorm, dilations[i], groups, M);
    // Optional batchnorm
    if (batchnorm)
      r += batchnorm_memory(block_out);
    // _output(out_channels, M) per block
    r.add_buffers((size_t)block_out * M);
  }

  // _block_vals: 1 entry of (channels, buffer_size)
  // buffer_size = input_buffers[0].size() = 32 * receptive_field
  long buffer_size = INPUT_BUFFER_SAFETY_FACTOR * receptive_field;
  r.add_buffers((size_t)channels * buffer_size);

  // _head: weight(out_channels, channels) + bias(out_channels)
  r.add_weights((size_t)out_channels * channels);
  r.add_weights(out_channels);

  // _head_output is resized per-call, not a fixed pre-allocation

  return r;
}

// ─── WaveNet helpers ────────────────────────────────────────────────────────

static FiLMParams parse_film_params(const json& layer_config, const std::string& key)
{
  FiLMParams fp;
  if (layer_config.find(key) == layer_config.end() || layer_config[key] == false)
    return fp; // inactive
  const json& fc = layer_config[key];
  fp.active = fc.value("active", true);
  fp.shift = fc.value("shift", true);
  fp.groups = fc.value("groups", 1);
  return fp;
}

enum class GatingMode
{
  NONE,
  GATED,
  BLENDED
};

static std::vector<GatingMode> parse_gating_modes(const json& layer_config, size_t num_layers)
{
  std::vector<GatingMode> modes;

  auto parse_str = [](const std::string& s) -> GatingMode {
    if (s == "gated")
      return GatingMode::GATED;
    if (s == "blended")
      return GatingMode::BLENDED;
    return GatingMode::NONE;
  };

  if (layer_config.find("gating_mode") != layer_config.end())
  {
    if (layer_config["gating_mode"].is_array())
    {
      for (const auto& gm : layer_config["gating_mode"])
        modes.push_back(parse_str(gm.get<std::string>()));
    }
    else
    {
      GatingMode mode = parse_str(layer_config["gating_mode"].get<std::string>());
      modes.resize(num_layers, mode);
    }
  }
  else if (layer_config.find("gated") != layer_config.end())
  {
    bool gated = layer_config["gated"];
    modes.resize(num_layers, gated ? GatingMode::GATED : GatingMode::NONE);
  }
  else
  {
    modes.resize(num_layers, GatingMode::NONE);
  }
  return modes;
}

// WaveNet _Layer memory
static MemoryResult wavenet_layer_memory(int condition_size, int channels, int bottleneck, int kernel_size, int dilation,
                                         GatingMode gating_mode, int groups_input, int groups_input_mixin,
                                         bool layer1x1_active, int layer1x1_groups, bool head1x1_active,
                                         int head1x1_out_channels, int head1x1_groups, const FiLMParams& conv_pre_film,
                                         const FiLMParams& conv_post_film, const FiLMParams& input_mixin_pre_film,
                                         const FiLMParams& input_mixin_post_film,
                                         const FiLMParams& activation_pre_film,
                                         const FiLMParams& activation_post_film,
                                         const FiLMParams& layer1x1_post_film, const FiLMParams& head1x1_post_film,
                                         int M)
{
  MemoryResult r;
  bool gated = (gating_mode != GatingMode::NONE);
  int conv_out = gated ? 2 * bottleneck : bottleneck;

  // _conv: Conv1D(channels -> conv_out, kernel_size, bias=true, dilation, groups_input)
  r += conv1d_memory(channels, conv_out, kernel_size, true, dilation, groups_input, M);

  // _input_mixin: Conv1x1(condition_size -> conv_out, bias=false, groups_input_mixin)
  r += conv1x1_memory(condition_size, conv_out, false, groups_input_mixin, M);

  // _layer1x1 (optional): Conv1x1(bottleneck -> channels, bias=true, layer1x1_groups)
  if (layer1x1_active)
    r += conv1x1_memory(bottleneck, channels, true, layer1x1_groups, M);

  // _head1x1 (optional): Conv1x1(bottleneck -> head1x1_out_channels, bias=true, head1x1_groups)
  if (head1x1_active)
    r += conv1x1_memory(bottleneck, head1x1_out_channels, true, head1x1_groups, M);

  // Buffers: _z(conv_out, M)
  r.add_buffers((size_t)conv_out * M);
  // _output_next_layer(channels, M)
  r.add_buffers((size_t)channels * M);
  // _output_head: if head1x1 active -> (head1x1_out_channels, M), else (bottleneck, M)
  int head_out = head1x1_active ? head1x1_out_channels : bottleneck;
  r.add_buffers((size_t)head_out * M);

  // FiLM modules (up to 8)
  r += film_memory(condition_size, channels, conv_pre_film, M);
  r += film_memory(condition_size, conv_out, conv_post_film, M);
  r += film_memory(condition_size, condition_size, input_mixin_pre_film, M);
  r += film_memory(condition_size, conv_out, input_mixin_post_film, M);
  r += film_memory(condition_size, conv_out, activation_pre_film, M);
  r += film_memory(condition_size, bottleneck, activation_post_film, M);
  if (layer1x1_active)
    r += film_memory(condition_size, channels, layer1x1_post_film, M);
  if (head1x1_active)
    r += film_memory(condition_size, head1x1_out_channels, head1x1_post_film, M);

  return r;
}

// WaveNet _LayerArray memory
static MemoryResult wavenet_layer_array_memory(const json& layer_config, int M)
{
  MemoryResult r;
  int input_size = layer_config["input_size"];
  int condition_size = layer_config["condition_size"];
  int head_size = layer_config["head_size"];
  int channels = layer_config["channels"];
  int bottleneck = layer_config.value("bottleneck", channels);
  int kernel_size = layer_config["kernel_size"];
  std::vector<int> dilations = layer_config["dilations"];
  size_t num_layers = dilations.size();
  bool head_bias = layer_config["head_bias"];

  int groups_input = layer_config.value("groups_input", 1);
  int groups_input_mixin = layer_config.value("groups_input_mixin", 1);

  // layer1x1 params
  bool layer1x1_active = true;
  int layer1x1_groups = 1;
  if (layer_config.find("layer1x1") != layer_config.end())
  {
    layer1x1_active = layer_config["layer1x1"]["active"];
    layer1x1_groups = layer_config["layer1x1"]["groups"];
  }

  // head1x1 params
  bool head1x1_active = false;
  int head1x1_out_channels = channels;
  int head1x1_groups = 1;
  if (layer_config.find("head1x1") != layer_config.end())
  {
    head1x1_active = layer_config["head1x1"]["active"];
    head1x1_out_channels = layer_config["head1x1"]["out_channels"];
    head1x1_groups = layer_config["head1x1"]["groups"];
  }

  // Gating modes
  std::vector<GatingMode> gating_modes = parse_gating_modes(layer_config, num_layers);

  // FiLM params
  FiLMParams conv_pre = parse_film_params(layer_config, "conv_pre_film");
  FiLMParams conv_post = parse_film_params(layer_config, "conv_post_film");
  FiLMParams input_mixin_pre = parse_film_params(layer_config, "input_mixin_pre_film");
  FiLMParams input_mixin_post = parse_film_params(layer_config, "input_mixin_post_film");
  FiLMParams activation_pre = parse_film_params(layer_config, "activation_pre_film");
  FiLMParams activation_post = parse_film_params(layer_config, "activation_post_film");
  FiLMParams layer1x1_post = parse_film_params(layer_config, "layer1x1_post_film");
  FiLMParams head1x1_post = parse_film_params(layer_config, "head1x1_post_film");

  // _rechannel: Conv1x1(input_size -> channels, bias=false)
  r += conv1x1_memory(input_size, channels, false, 1, M);

  // Per-layer
  for (size_t i = 0; i < num_layers; i++)
  {
    r += wavenet_layer_memory(condition_size, channels, bottleneck, kernel_size, dilations[i], gating_modes[i],
                              groups_input, groups_input_mixin, layer1x1_active, layer1x1_groups, head1x1_active,
                              head1x1_out_channels, head1x1_groups, conv_pre, conv_post, input_mixin_pre,
                              input_mixin_post, activation_pre, activation_post, layer1x1_post, head1x1_post, M);
  }

  // _head_rechannel: Conv1x1(head_output_size -> head_size, bias=head_bias)
  int head_output_size = head1x1_active ? head1x1_out_channels : bottleneck;
  r += conv1x1_memory(head_output_size, head_size, head_bias, 1, M);

  // Buffers: _layer_outputs(channels, M)
  r.add_buffers((size_t)channels * M);
  // _head_inputs(head_output_size, M)
  r.add_buffers((size_t)head_output_size * M);

  return r;
}

// Forward declaration for recursive condition_dsp
static MemoryResult compute_memory(const std::string& architecture, const json& config, int M);

// WaveNet top-level memory
static MemoryResult wavenet_memory(const json& config, int M)
{
  MemoryResult r;
  int in_channels = config.value("in_channels", 1);

  // condition_dim = in_channels (from _get_condition_dim())
  int condition_dim = in_channels;

  // Recursive condition_dsp
  bool has_condition_dsp = false;
  int condition_output_channels = condition_dim;
  if (config.find("condition_dsp") != config.end())
  {
    has_condition_dsp = true;
    const json& cdsp = config["condition_dsp"];
    std::string cdsp_arch = cdsp["architecture"];
    json cdsp_config = cdsp["config"];
    r += compute_memory(cdsp_arch, cdsp_config, M);
    // condition_output_channels comes from the condition_dsp's output
    // For now, we use condition_size from first layer as a proxy
    // (the actual model validates this match)
    if (config.find("layers") != config.end() && config["layers"].size() > 0)
      condition_output_channels = config["layers"][0]["condition_size"];
  }

  // _condition_input(condition_dim, M)
  r.add_buffers((size_t)condition_dim * M);

  // _condition_output
  if (!has_condition_dsp)
  {
    // _condition_output(condition_dim, M)
    r.add_buffers((size_t)condition_dim * M);
  }
  else
  {
    // _condition_output(condition_output_channels, M)
    r.add_buffers((size_t)condition_output_channels * M);
    // _condition_dsp_input_buffers: condition_dim vectors of M doubles/floats
    // These are std::vector<std::vector<NAM_SAMPLE>> where NAM_SAMPLE is double
    r.add_buffers((size_t)condition_dim * M * (sizeof(double) / sizeof(float)));
    // _condition_dsp_output_buffers: condition_output_channels vectors of M doubles
    r.add_buffers((size_t)condition_output_channels * M * (sizeof(double) / sizeof(float)));
    // Pointer arrays are negligible
  }

  // Layer arrays
  for (const auto& layer_config : config["layers"])
    r += wavenet_layer_array_memory(layer_config, M);

  // _head_scale (1 float) — it's a weight
  r.add_weights(1);

  return r;
}

// ─── Dispatch ───────────────────────────────────────────────────────────────

static MemoryResult compute_memory(const std::string& architecture, const json& config, int M)
{
  if (architecture == "WaveNet")
    return wavenet_memory(config, M);
  if (architecture == "LSTM")
    return lstm_memory(config);
  if (architecture == "ConvNet")
    return convnet_memory(config, M);
  if (architecture == "Linear")
    return linear_memory(config);
  throw std::runtime_error("Unknown architecture: " + architecture);
}

// ─── Formatting helpers ─────────────────────────────────────────────────────

static std::string format_bytes(size_t bytes)
{
  char buf[64];
  if (bytes < 1024)
    snprintf(buf, sizeof(buf), "%zu bytes", bytes);
  else if (bytes < 1024 * 1024)
    snprintf(buf, sizeof(buf), "%.2f KB", bytes / 1024.0);
  else
    snprintf(buf, sizeof(buf), "%.2f MB", bytes / (1024.0 * 1024.0));
  return buf;
}

static std::string format_with_commas(size_t n)
{
  std::string s = std::to_string(n);
  int insert_pos = (int)s.length() - 3;
  while (insert_pos > 0)
  {
    s.insert(insert_pos, ",");
    insert_pos -= 3;
  }
  return s;
}

// ─── Main ───────────────────────────────────────────────────────────────────

int main(int argc, char* argv[])
{
  if (argc < 2)
  {
    fprintf(stderr, "Usage: memory_usage <model_path> [--buffer-size N]\n");
    return 1;
  }

  const char* model_path = argv[1];
  int buffer_size = DEFAULT_BUFFER_SIZE;

  for (int i = 2; i < argc; i++)
  {
    if (strcmp(argv[i], "--buffer-size") == 0 && i + 1 < argc)
    {
      buffer_size = atoi(argv[++i]);
      if (buffer_size <= 0)
      {
        fprintf(stderr, "Error: buffer size must be positive\n");
        return 1;
      }
    }
    else
    {
      fprintf(stderr, "Unknown option: %s\n", argv[i]);
      return 1;
    }
  }

  // Read and parse JSON
  std::ifstream file(model_path);
  if (!file.is_open())
  {
    fprintf(stderr, "Error: cannot open %s\n", model_path);
    return 1;
  }

  json j;
  try
  {
    file >> j;
  }
  catch (const std::exception& e)
  {
    fprintf(stderr, "Error parsing JSON: %s\n", e.what());
    return 1;
  }

  std::string architecture = j["architecture"];
  json config = j["config"];

  // Cross-check: count weights in JSON
  size_t json_weight_count = 0;
  if (j.find("weights") != j.end())
    json_weight_count = j["weights"].size();

  double sample_rate = -1.0;
  if (j.find("sample_rate") != j.end())
    sample_rate = j["sample_rate"];

  try
  {
    MemoryResult result = compute_memory(architecture, config, buffer_size);
    size_t total = result.weight_bytes + result.buffer_bytes;

    printf("Model: %s\n", model_path);
    printf("Architecture: %s\n", architecture.c_str());
    if (sample_rate > 0)
      printf("Sample rate: %.0f Hz\n", sample_rate);
    printf("\n");
    printf("Weights:  %s bytes (%s)\n", format_with_commas(result.weight_bytes).c_str(),
           format_bytes(result.weight_bytes).c_str());
    printf("Buffers:  %s bytes (%s)  [buffer size: %d]\n", format_with_commas(result.buffer_bytes).c_str(),
           format_bytes(result.buffer_bytes).c_str(), buffer_size);
    printf("Total:    %s bytes (%s)\n", format_with_commas(total).c_str(), format_bytes(total).c_str());

    if (json_weight_count > 0)
    {
      printf("\nJSON weights: %zu values (%s bytes)\n", json_weight_count,
             format_with_commas(json_weight_count * sizeof(float)).c_str());
    }
  }
  catch (const std::exception& e)
  {
    fprintf(stderr, "Error computing memory: %s\n", e.what());
    return 1;
  }

  return 0;
}
