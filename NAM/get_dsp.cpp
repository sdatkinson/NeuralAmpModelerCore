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

void VerifyConfigVersion(const std::string& versionStr)
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
  if (j.find("weights") != j.end())
  {
    auto weight_list = j["weights"];
    std::vector<float> weights;
    for (auto it = weight_list.begin(); it != weight_list.end(); ++it)
      weights.push_back(*it);
    return weights;
  }
  else
    throw std::runtime_error("Corrupted model is missing weights.");
}

std::unique_ptr<DSP> GetDSP(const std::filesystem::path& configFileName)
{
  dspData temp;
  return GetDSP(configFileName, temp);
}

std::unique_ptr<DSP> GetDSP(const std::filesystem::path& configFileName, dspData& config)
{
  if (!std::filesystem::exists(configFileName))
    throw std::runtime_error("Config JSON doesn't exist!\n");
  std::ifstream i(configFileName);
  nlohmann::json j;
  i >> j;
  VerifyConfigVersion(j["version"]);

  auto architecture = j["architecture"];
  std::vector<float> weights = GetWeights(j);

  // Assign values to config
  config.version = j["version"];
  config.architecture = j["architecture"];
  config.config = j["config"];
  config.metadata = j["metadata"];
  config.weights = weights;
  if (j.find("sample_rate") != j.end())
    config.expectedSampleRate = j["sample_rate"];
  else
  {
    config.expectedSampleRate = -1.0;
  }

  /*Copy to a new dsp_config object for GetDSP below,
   since not sure if weights actually get modified as being non-const references on some
   model constructors inside GetDSP(dsp_config& conf).
   We need to return unmodified version of dsp_config via returnedConfig.*/
  dspData conf = config;

  return GetDSP(conf);
}

std::unique_ptr<DSP> GetDSP(const char* jsonStr, dspData& config)
{
  nlohmann::json j = nlohmann::json::parse(jsonStr);
  VerifyConfigVersion(j["version"]);

  auto architecture = j["architecture"];
  std::vector<float> weights = GetWeights(j);

  // Assign values to config
  config.version = j["version"];
  config.architecture = j["architecture"];
  config.config = j["config"];
  config.metadata = j["metadata"];
  config.weights = weights;
  if (j.find("sample_rate") != j.end())
    config.expectedSampleRate = j["sample_rate"];
  else
  {
    config.expectedSampleRate = -1.0;
  }

  /*Copy to a new dsp_config object for GetDSP below,
   since not sure if weights actually get modified as being non-const references on some
   model constructors inside GetDSP(dsp_config& conf).
   We need to return unmodified version of dsp_config via returnedConfig.*/
  dspData conf = config;

  return GetDSP(conf);
}

std::unique_ptr<DSP> GetDSP(dspData& conf)
{
  VerifyConfigVersion(conf.version);

  auto& architecture = conf.architecture;
  nlohmann::json& config = conf.config;
  std::vector<float>& weights = conf.weights;
  bool haveLoudness = false;
  double loudness = 0.0;

  if (!conf.metadata.is_null())
  {
    if (conf.metadata.find("loudness") != conf.metadata.end())
    {
      loudness = conf.metadata["loudness"];
      haveLoudness = true;
    }
  }
  const double expectedSampleRate = conf.expectedSampleRate;

  std::unique_ptr<DSP> out = nullptr;
  if (architecture == "Linear")
  {
    const int receptiveField = config["receptiveField"];
    const bool bias = config["bias"];
    out = std::make_unique<Linear>(receptiveField, bias, weights, expectedSampleRate);
  }
  else if (architecture == "ConvNet")
  {
    const int channels = config["channels"];
    const bool batchnorm = config["batchnorm"];
    std::vector<int> dilations;
    for (size_t i = 0; i < config["dilations"].size(); i++)
      dilations.push_back(config["dilations"][i]);
    auto activation = config["activation"];
    out = std::make_unique<convnet::ConvNet>(channels, dilations, batchnorm, activation, weights, expectedSampleRate);
  }
  else if (architecture == "LSTM")
  {
    const int numLayers = config["num_layers"];
    const int inputSize = config["input_size"];
    const int hiddenSize = config["hidden_size"];
    out = std::make_unique<lstm::LSTM>(numLayers, inputSize, hiddenSize, weights, expectedSampleRate);
  }
  else if (architecture == "WaveNet")
  {
    std::vector<wavenet::LayerArrayParams> layerArrayParams;
    for (size_t i = 0; i < config["layers"].size(); i++)
    {
      nlohmann::json layerConfig = config["layers"][i];
      std::vector<int> dilations;
      for (size_t j = 0; j < layerConfig["dilations"].size(); j++)
        dilations.push_back(layerConfig["dilations"][j]);
      layerArrayParams.push_back(
        wavenet::LayerArrayParams(layerConfig["input_size"], layerConfig["condition_size"], layerConfig["head_size"],
                                  layerConfig["channels"], layerConfig["kernel_size"], dilations,
                                  layerConfig["activation"], layerConfig["gated"], layerConfig["head_bias"]));
    }
    const bool withHead = config["head"] == NULL;
    const float headScale = config["head_scale"];
    out = std::make_unique<wavenet::WaveNet>(layerArrayParams, headScale, withHead, weights, expectedSampleRate);
  }
  else
  {
    throw std::runtime_error("Unrecognized architecture");
  }
  if (haveLoudness)
  {
    out->SetLoudness(loudness);
  }

  // "pre-warm" the model to settle initial conditions
  out->Prewarm();

  return out;
}
}; // namespace nam
