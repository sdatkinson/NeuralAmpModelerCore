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

// Get sample rate from a config json:
double get_sample_rate(const nlohmann::json& j)
{
  if (j.find("sample_rate") != j.end())
    return j["sample_rate"];
  else
    return -1.0;
};
}; // namespace nam
