#pragma once

// This header defines the WaveNet end-user model: ``WaveNet`` (DSP), ``WaveNetConfig``, and JSON helpers
// ``parse_config_json`` / ``create_config``. Lower-level building blocks live in ``params.h`` and ``detail.h``.

#include <memory>
#include <optional>
#include <vector>

#include <Eigen/Dense>

#include "../dsp.h"
#include "json.hpp"

#include "detail.h"

namespace nam
{
namespace wavenet
{

/// \brief The main WaveNet model
///
/// WaveNet is a dilated convolutional neural network architecture for audio processing.
/// It consists of multiple LayerArrays, each containing multiple layers with increasing
/// dilation factors. The model processes audio through:
///
/// 1. Condition DSP (optional) - processes input to generate conditioning signal
/// 2. LayerArrays - sequential processing with residual and skip connections
/// 3. Head scaling - final output scaling
///
/// The model supports real-time audio processing with pre-allocated buffers.
class WaveNet : public DSP
{
public:
  /// \brief Constructor
  /// \param in_channels Number of input channels
  /// \param layer_array_params Parameters for each layer array
  /// \param head_scale Scaling factor applied to the final head output
  /// \param with_head Whether to apply the optional post-stack head (Conv1D stack after layer arrays)
  /// \param head_params Configuration for the post-stack head when ``with_head`` is true
  /// \param weights Model weights (will be consumed during construction)
  /// \param condition_dsp Optional DSP module for processing the conditioning input
  /// \param expected_sample_rate Expected sample rate in Hz (-1.0 if unknown)
  WaveNet(const int in_channels, const std::vector<LayerArrayParams>& layer_array_params, const float head_scale,
          const bool with_head, std::optional<HeadParams> head_params, std::vector<float> weights,
          std::unique_ptr<DSP> condition_dsp, const double expected_sample_rate = -1.0);

  /// \brief Destructor
  ~WaveNet() = default;

  /// \brief Process audio frames
  ///
  /// Implements the DSP::process() interface. Processes input audio through the
  /// complete WaveNet pipeline and writes to output.
  /// \param input Input audio buffers (in_channels x frames)
  /// \param output Output audio buffers (out_channels x frames)
  /// \param num_frames Number of frames to process
  void process(NAM_SAMPLE** input, NAM_SAMPLE** output, const int num_frames) override;

  /// \brief Set model weights from a vector
  /// \param weights Vector containing all model weights
  void set_weights_(std::vector<float>& weights);

  /// \brief Set model weights from an iterator
  /// \param weights Iterator to the weights vector. Will be advanced as weights are consumed.
  void set_weights_(std::vector<float>::iterator& weights);

protected:
  // Element-wise arrays:
  Eigen::MatrixXf _condition_input;
  Eigen::MatrixXf _condition_output;
  std::unique_ptr<DSP> _condition_dsp;
  // Temporary buffers for condition DSP processing (to avoid allocations in _process_condition)
  std::vector<std::vector<NAM_SAMPLE>> _condition_dsp_input_buffers;
  std::vector<std::vector<NAM_SAMPLE>> _condition_dsp_output_buffers;
  std::vector<NAM_SAMPLE*> _condition_dsp_input_ptrs;
  std::vector<NAM_SAMPLE*> _condition_dsp_output_ptrs;

  /// \brief Resize all buffers to handle maxBufferSize frames
  /// \param maxBufferSize Maximum number of frames to process in a single call
  void SetMaxBufferSize(const int maxBufferSize) override;

  /// \brief Compute the conditioning array to be given to the layer arrays
  ///
  /// Processes the condition input through the condition DSP (if present) or
  /// passes it through directly.
  /// \param num_frames Number of frames to process
  virtual void _process_condition(const int num_frames);

  /// \brief Fill in the "condition" array that's fed into the various parts of the net
  ///
  /// Copies input audio into the condition buffer for processing.
  /// \param input Input audio buffers
  /// \param num_frames Number of frames to process
  virtual void _set_condition_array(NAM_SAMPLE** input, const int num_frames);

  /// \brief Get the number of conditioning inputs
  ///
  /// For standard WaveNet, this is just the audio input (same as input channels).
  /// \return Number of conditioning input channels
  virtual int _get_condition_dim() const { return NumInputChannels(); };

private:
  std::vector<detail::LayerArray> _layer_arrays;

  float _head_scale;

  std::unique_ptr<detail::Head> _post_stack_head;
  /// Scratch (in_channels × maxBufferSize) for scaled head input when ``_post_stack_head`` is used
  Eigen::MatrixXf _scaled_head_scratch;

  int mPrewarmSamples = 0; // Pre-compute during initialization
  int PrewarmSamples() override { return mPrewarmSamples; };
};

/// \brief Configuration for a WaveNet model
struct WaveNetConfig : public ModelConfig
{
  int in_channels;
  std::vector<LayerArrayParams> layer_array_params;
  float head_scale;
  bool with_head;
  std::optional<HeadParams> head_params;
  std::unique_ptr<DSP> condition_dsp;

  // Move-only due to unique_ptr
  WaveNetConfig() = default;
  WaveNetConfig(WaveNetConfig&&) = default;
  WaveNetConfig& operator=(WaveNetConfig&&) = default;
  WaveNetConfig(const WaveNetConfig&) = delete;
  WaveNetConfig& operator=(const WaveNetConfig&) = delete;

  std::unique_ptr<DSP> create(std::vector<float> weights, double sampleRate) override;
};

/// \brief Parse WaveNet configuration from JSON
/// \param config JSON configuration object
/// \param expectedSampleRate Expected sample rate in Hz (-1.0 if unknown)
/// \return WaveNetConfig
WaveNetConfig parse_config_json(const nlohmann::json& config, const double expectedSampleRate);

/// \brief Config parser for ConfigParserRegistry
std::unique_ptr<ModelConfig> create_config(const nlohmann::json& config, double sampleRate);

} // namespace wavenet
} // namespace nam
