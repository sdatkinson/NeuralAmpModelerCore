#pragma once
// LSTM implementation

#include <map>
#include <vector>
#include <memory>

#include <Eigen/Dense>

#include "dsp.h"

namespace nam
{
namespace lstm
{
/// \brief A single LSTM cell
class LSTMCell
{
public:
  /// \brief Constructor
  /// \param input_size Size of the input vector
  /// \param hidden_size Size of the hidden state
  /// \param weights Iterator to the weights vector. Will be advanced as weights are consumed.
  LSTMCell(const int input_size, const int hidden_size, std::vector<float>::iterator& weights);

  /// \brief Get the current hidden state
  /// \return Hidden state vector
  Eigen::VectorXf get_hidden_state() const { return this->_xh(Eigen::placeholders::lastN(this->_get_hidden_size())); };

  /// \brief Process a single input vector
  /// \param x Input vector
  void process_(const Eigen::VectorXf& x);

private:
  // Parameters
  // xh -> ifgo
  // (dx+dh) -> (4*dh)
  Eigen::MatrixXf _w;
  Eigen::VectorXf _b;

  // State
  // Concatenated input and hidden state
  Eigen::VectorXf _xh;
  // Input, Forget, Cell, Output gates
  Eigen::VectorXf _ifgo;

  // Cell state
  Eigen::VectorXf _c;

  long _get_hidden_size() const { return this->_b.size() / 4; };
  long _get_input_size() const { return this->_xh.size() - this->_get_hidden_size(); };
};

/// \brief A multi-layer LSTM model
///
/// A multi-layer LSTM processes audio frame-by-frame, maintaining hidden states
/// across layers. Each layer processes the hidden state from the previous layer as input.
class LSTM : public DSP
{
public:
  /// \brief Constructor
  /// \param in_channels Number of input channels
  /// \param out_channels Number of output channels
  /// \param num_layers Number of LSTM layers
  /// \param input_size Size of the input to each LSTM cell
  /// \param hidden_size Size of the hidden state in each LSTM cell
  /// \param weights Model weights vector
  /// \param expected_sample_rate Expected sample rate in Hz (-1.0 if unknown)
  LSTM(const int in_channels, const int out_channels, const int num_layers, const int input_size, const int hidden_size,
       std::vector<float>& weights, const double expected_sample_rate = -1.0);

  /// \brief Destructor
  ~LSTM() = default;

  /// \brief Process audio frames
  /// \param input Input audio buffers
  /// \param output Output audio buffers
  /// \param num_frames Number of frames to process
  void process(NAM_SAMPLE** input, NAM_SAMPLE** output, const int num_frames) override;

protected:
  // Hacky, but a half-second seems to work for most models.
  int PrewarmSamples() override;

  Eigen::MatrixXf _head_weight; // (out_channels x hidden_size)
  Eigen::VectorXf _head_bias; // (out_channels)
  std::vector<LSTMCell> _layers;

  void _process_sample();

  // Input to the LSTM.
  // Since this is assumed to not be a parametric model, its shape should be (in_channels,)
  Eigen::VectorXf _input;
  // Output from _process_sample - multi-channel output vector (size out_channels)
  Eigen::VectorXf _output;
};

/// \brief Configuration for an LSTM model
struct LSTMConfig
{
  int num_layers;
  int input_size;
  int hidden_size;
  int in_channels;
  int out_channels;
};

/// \brief Parse LSTM configuration from JSON
/// \param config JSON configuration object
/// \return LSTMConfig
LSTMConfig parse_config_json(const nlohmann::json& config);

/// \brief Factory function to instantiate LSTM from JSON
/// \param config JSON configuration object
/// \param weights Model weights vector
/// \param expectedSampleRate Expected sample rate in Hz (-1.0 if unknown)
/// \return Unique pointer to a DSP object (LSTM instance)
std::unique_ptr<DSP> Factory(const nlohmann::json& config, std::vector<float>& weights,
                             const double expectedSampleRate);

}; // namespace lstm
}; // namespace nam
