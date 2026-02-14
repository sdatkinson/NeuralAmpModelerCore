#pragma once
// Unified model configuration types for both JSON and binary loaders.
// No circular dependencies: architecture headers define config structs,
// this header combines them into a variant.

#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "convnet.h"
#include "dsp.h"
#include "lstm.h"
#include "wavenet.h"

namespace nam
{

/// \brief Metadata common to all model formats
struct ModelMetadata
{
  std::string version;
  double sample_rate = -1.0;
  std::optional<double> loudness;
  std::optional<double> input_level;
  std::optional<double> output_level;
};

/// \brief Variant of all architecture configs
using ModelConfig = std::variant<linear::LinearConfig, lstm::LSTMConfig, convnet::ConvNetConfig, wavenet::WaveNetConfig>;

/// \brief Construct a DSP object from a typed config, weights, and metadata
///
/// This is the single construction path used by both JSON and binary loaders.
/// Handles construction, metadata application, and prewarm.
/// \param config Architecture-specific configuration (variant)
/// \param weights Model weights (taken by value to allow move for WaveNet)
/// \param metadata Model metadata (version, sample rate, loudness, levels)
/// \return Unique pointer to a DSP object
std::unique_ptr<DSP> create_dsp(ModelConfig config, std::vector<float> weights, const ModelMetadata& metadata);

/// \brief Parse a ModelConfig from a JSON architecture name and config block
/// \param architecture Architecture name string (e.g., "WaveNet", "LSTM")
/// \param config JSON config block for this architecture
/// \param sample_rate Expected sample rate from metadata
/// \return ModelConfig variant
ModelConfig parse_model_config_json(const std::string& architecture, const nlohmann::json& config,
                                    double sample_rate);

} // namespace nam
