#pragma once

// Specialized WaveNet fast path for the A2 standard (Channels=8) and
// A2 nano (Channels=3) models. Shares the same architecture shape; only
// the channel count differs.
//
// When NAM_ENABLE_A2_FAST is defined at build time, wavenet::create_config
// consults is_a2_shape() on every incoming WaveNet config and, on match,
// instantiates an A2FastModel<Channels> instead of the generic WaveNet.
//
// The baseline here is correct-but-unoptimized (plain column-major loops).
// Follow-up optimizations (unrolled GEMV, tap-major nest, factored
// per-kernel-size helpers) plug into the same class.

#if defined(NAM_ENABLE_A2_FAST)

  #include <array>
  #include <memory>

  #include "../model_config.h"
  #include "json.hpp"

namespace nam
{
namespace wavenet
{
namespace a2_fast
{

/// \brief Number of layers in an A2 layer array.
constexpr int kNumLayers = 23;
/// \brief Kernel size of the layer-array head rechannel convolution.
constexpr int kHeadKernelSize = 16;
/// \brief Head scale factor used by every A2 model.
constexpr float kHeadScale = 0.01f;
/// \brief LeakyReLU negative-slope used by every layer.
constexpr float kLeakySlope = 0.01f;

/// \brief Per-layer kernel sizes (fixed pattern shared by A2 standard + nano).
inline constexpr std::array<int, kNumLayers> kKernelSizes = {
  6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 15, 15, 6, 6, 6, 6, 6, 6, 6};

/// \brief Per-layer dilations (fixed pattern shared by A2 standard + nano).
inline constexpr std::array<int, kNumLayers> kDilations = {
  1, 3, 7, 17, 41, 101, 239, 1, 3, 7, 17, 41, 101, 239, 1, 13, 1, 3, 7, 17, 41, 101, 239};

/// \brief Strict detector: returns true iff config matches the A2 shape.
/// \param config   The "config" sub-object from a .nam WaveNet entry.
/// \param channels Out-param set to 3 (A2 nano) or 8 (A2 standard) on match.
/// \return true if every architectural knob matches the A2 signature exactly.
bool is_a2_shape(const nlohmann::json& config, int* channels);

/// \brief Build a ModelConfig that instantiates the A2 fast path.
/// \pre is_a2_shape(config, ...) returned true.
std::unique_ptr<ModelConfig> create_a2_fast_config(const nlohmann::json& config, double sampleRate);

} // namespace a2_fast
} // namespace wavenet
} // namespace nam

#endif // NAM_ENABLE_A2_FAST
