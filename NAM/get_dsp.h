#pragma once

#include <fstream>

#include "dsp.h"

namespace nam
{
// Get NAM from a .nam file at the provided location
std::unique_ptr<DSP> get_dsp(const std::filesystem::path config_filename);

// Get NAM from a provided configuration struct
std::unique_ptr<DSP> get_dsp(dspData& conf);

// Get NAM from a provided .nam file path and store its configuration in the provided conf
std::unique_ptr<DSP> get_dsp(const std::filesystem::path config_filename, dspData& returnedConfig);

// Get NAM from a provided configuration JSON object
std::unique_ptr<DSP> get_dsp(const nlohmann::json& config, dspData& returnedConfig);

// Get NAM from a provided configuration JSON object (convenience overload)
std::unique_ptr<DSP> get_dsp(const nlohmann::json& config);

// Get sample rate from a .nam file
// Returns -1 if not known (Really old .nam files)
double get_sample_rate_from_nam_file(const nlohmann::json& j);
}; // namespace nam
