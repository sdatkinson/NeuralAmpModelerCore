#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

#include "dsp.h"
#include "json.hpp"
#include "lstm.h"
#include "convnet.h"
#include "wavenet.h"

namespace nam
{
struct Version
{
  int major;
  int minor;
  int patch;
};

Version ParseVersion(const std::string& versionStr)
{
  Version version;

  // Split the version string into major, minor, and patch components
  std::stringstream ss(versionStr);
  std::string majorStr, minorStr, patchStr;
  std::getline(ss, majorStr, '.');
  std::getline(ss, minorStr, '.');
  std::getline(ss, patchStr);

  // Parse the components as integers and assign them to the version struct
  try
  {
    version.major = std::stoi(majorStr);
    version.minor = std::stoi(minorStr);
    version.patch = std::stoi(patchStr);
  }
  catch (const std::invalid_argument&)
  {
    throw std::invalid_argument("Invalid version string: " + versionStr);
  }
  catch (const std::out_of_range&)
  {
    throw std::out_of_range("Version string out of range: " + versionStr);
  }

  // Validate the semver components
  if (version.major < 0 || version.minor < 0 || version.patch < 0)
  {
    throw std::invalid_argument("Negative version component: " + versionStr);
  }
  return version;
}

void verify_config_version(const std::string versionStr)
{
  Version version = ParseVersion(versionStr);
  if (version.major != 0 || version.minor != 5)
  {
    std::stringstream ss;
    ss << "Model config is an unsupported version " << versionStr
       << ". Try either converting the model to a more recent version, or "
          "update your version of the NAM plugin.";
    throw std::runtime_error(ss.str());
  }
}

std::vector<float> GetWeights(nlohmann::json const& j)
{
  auto it = j.find("weights");
  if (it != j.end())
  {
    return *it;
  }
  else
    throw std::runtime_error("Corrupted model file is missing weights.");
}

std::unique_ptr<DSP> get_dsp(const std::filesystem::path config_filename)
{
  dspData temp;
  return get_dsp(config_filename, temp);
}

std::unique_ptr<DSP> get_dsp(const std::filesystem::path config_filename, dspData& returnedConfig)
{
  if (!std::filesystem::exists(config_filename))
    throw std::runtime_error("Config file doesn't exist!\n");
  std::ifstream i(config_filename);
  nlohmann::json j;
  i >> j;
  verify_config_version(j["version"]);

  auto architecture = j["architecture"];
  nlohmann::json config = j["config"];
  std::vector<float> weights = GetWeights(j);

  // Assign values to returnedConfig
  returnedConfig.version = j["version"];
  returnedConfig.architecture = j["architecture"];
  returnedConfig.config = j["config"];
  returnedConfig.metadata = j["metadata"];
  returnedConfig.weights = weights;
  if (j.find("sample_rate") != j.end())
    returnedConfig.expected_sample_rate = j["sample_rate"];
  else
  {
    returnedConfig.expected_sample_rate = -1.0;
  }


  /*Copy to a new dsp_config object for get_dsp below,
   since not sure if weights actually get modified as being non-const references on some
   model constructors inside get_dsp(dsp_config& conf).
   We need to return unmodified version of dsp_config via returnedConfig.*/
  dspData conf = returnedConfig;

  return get_dsp(conf);
}

struct OptionalValue
{
  bool have = false;
  double value = 0.0;
};

std::unique_ptr<DSP> get_dsp(dspData& conf)
{
  verify_config_version(conf.version);

  auto& architecture = conf.architecture;
  nlohmann::json& config = conf.config;
  std::vector<float>& weights = conf.weights;
  OptionalValue loudness, inputLevel, outputLevel;

  auto AssignOptional = [&conf](const std::string key, OptionalValue& v) {
    if (conf.metadata.find(key) != conf.metadata.end())
    {
      if (!conf.metadata[key].is_null())
      {
        v.value = conf.metadata[key];
        v.have = true;
      }
    }
  };

  if (!conf.metadata.is_null())
  {
    AssignOptional("loudness", loudness);
    AssignOptional("input_level_dbu", inputLevel);
    AssignOptional("output_level_dbu", outputLevel);
  }
  const double expectedSampleRate = conf.expected_sample_rate;

  std::unique_ptr<DSP> out = nullptr;
  if (architecture == "Linear")
  {
    const int receptive_field = config["receptive_field"];
    const bool _bias = config["bias"];
    out = std::make_unique<Linear>(receptive_field, _bias, weights, expectedSampleRate);
  }
  else if (architecture == "ConvNet")
  {
    const int channels = config["channels"];
    const bool batchnorm = config["batchnorm"];
    std::vector<int> dilations = config["dilations"];
    const std::string activation = config["activation"];
    out = std::make_unique<convnet::ConvNet>(channels, dilations, batchnorm, activation, weights, expectedSampleRate);
  }
  else if (architecture == "LSTM")
  {
    const int num_layers = config["num_layers"];
    const int input_size = config["input_size"];
    const int hidden_size = config["hidden_size"];
    out = std::make_unique<lstm::LSTM>(num_layers, input_size, hidden_size, weights, expectedSampleRate);
  }
  else if (architecture == "WaveNet")
  {
    std::vector<wavenet::LayerArrayParams> layer_array_params;
    for (size_t i = 0; i < config["layers"].size(); i++)
    {
      nlohmann::json layer_config = config["layers"][i];
      layer_array_params.push_back(
        wavenet::LayerArrayParams(layer_config["input_size"], layer_config["condition_size"], layer_config["head_size"],
                                  layer_config["channels"], layer_config["kernel_size"], layer_config["dilations"],
                                  layer_config["activation"], layer_config["gated"], layer_config["head_bias"]));
    }
    const bool with_head = !config["head"].is_null();
    const float head_scale = config["head_scale"];
    out = std::make_unique<wavenet::WaveNet>(layer_array_params, head_scale, with_head, weights, expectedSampleRate);
  }
  else
  {
    throw std::runtime_error("Unrecognized architecture");
  }
  if (loudness.have)
  {
    out->SetLoudness(loudness.value);
  }
  if (inputLevel.have)
  {
    out->SetInputLevel(inputLevel.value);
  }
  if (outputLevel.have)
  {
    out->SetOutputLevel(outputLevel.value);
  }

  // "pre-warm" the model to settle initial conditions
  // Can this be removed now that it's part of Reset()?
  out->prewarm();

  return out;
}
}; // namespace nam
