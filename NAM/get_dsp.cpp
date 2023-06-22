#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

#include "dsp.h"
#include "json.hpp"
#include "lstm.h"
#include "convnet.h"
#include "wavenet.h"

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

std::vector<float> GetWeights(nlohmann::json const& j, const std::filesystem::path config_path)
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
    throw std::runtime_error("Corrupted model file is missing weights.");
}

std::unique_ptr<DSP> get_dsp_legacy(const std::filesystem::path model_dir)
{
  auto config_filename = model_dir / std::filesystem::path("config.json");
  dspData temp;
  return get_dsp(config_filename, temp);
}

std::unique_ptr<DSP> get_dsp(const std::filesystem::path config_filename)
{
  dspData temp;
  return get_dsp(config_filename, temp);
}

std::unique_ptr<DSP> get_dsp(const std::filesystem::path config_filename, dspData& returnedConfig)
{
  if (!std::filesystem::exists(config_filename))
    throw std::runtime_error("Config JSON doesn't exist!\n");
  std::ifstream i(config_filename);
  nlohmann::json j;
  i >> j;
  verify_config_version(j["version"]);

  auto architecture = j["architecture"];
  nlohmann::json config = j["config"];
  std::vector<float> params = GetWeights(j, config_filename);

  // Assign values to returnedConfig
  returnedConfig.version = j["version"];
  returnedConfig.architecture = j["architecture"];
  returnedConfig.config = j["config"];
  returnedConfig.metadata = j["metadata"];
  returnedConfig.params = params;
  if (j.find("sample_rate") != j.end())
    returnedConfig.expected_sample_rate = j["sample_rate"];
  else
  {
    returnedConfig.expected_sample_rate = -1.0;
  }


  /*Copy to a new dsp_config object for get_dsp below,
  since not sure if params actually get modified as being non-const references on some
  model constructors inside get_dsp(dsp_config& conf).
  We need to return unmodified version of dsp_config via returnedConfig.*/
  dspData conf = returnedConfig;

  return get_dsp(conf);
}

std::unique_ptr<DSP> get_dsp(dspData& conf)
{
  verify_config_version(conf.version);

  auto& architecture = conf.architecture;
  nlohmann::json& config = conf.config;
  std::vector<float>& params = conf.params;
  bool haveLoudness = false;
  double loudness = TARGET_DSP_LOUDNESS;

  if (!conf.metadata.is_null())
  {
    if (conf.metadata.find("loudness") != conf.metadata.end())
    {
      loudness = conf.metadata["loudness"];
      haveLoudness = true;
    }
  }
  const double expected_sample_rate = conf.expected_sample_rate;

  if (architecture == "Linear")
  {
    const int receptive_field = config["receptive_field"];
    const bool _bias = config["bias"];
    return std::make_unique<Linear>(loudness, receptive_field, _bias, params, expected_sample_rate);
  }
  else if (architecture == "ConvNet")
  {
    const int channels = config["channels"];
    const bool batchnorm = config["batchnorm"];
    std::vector<int> dilations;
    for (size_t i = 0; i < config["dilations"].size(); i++)
      dilations.push_back(config["dilations"][i]);
    const std::string activation = config["activation"];
    return std::make_unique<convnet::ConvNet>(
      loudness, channels, dilations, batchnorm, activation, params, expected_sample_rate);
  }
  else if (architecture == "LSTM")
  {
    const int num_layers = config["num_layers"];
    const int input_size = config["input_size"];
    const int hidden_size = config["hidden_size"];
    auto empty_json = nlohmann::json{};
    return std::make_unique<lstm::LSTM>(
      loudness, num_layers, input_size, hidden_size, params, empty_json, expected_sample_rate);
  }
  else if (architecture == "CatLSTM")
  {
    const int num_layers = config["num_layers"];
    const int input_size = config["input_size"];
    const int hidden_size = config["hidden_size"];
    return std::make_unique<lstm::LSTM>(
      loudness, num_layers, input_size, hidden_size, params, config["parametric"], expected_sample_rate);
  }
  else if (architecture == "WaveNet" || architecture == "CatWaveNet")
  {
    std::vector<wavenet::LayerArrayParams> layer_array_params;
    for (size_t i = 0; i < config["layers"].size(); i++)
    {
      nlohmann::json layer_config = config["layers"][i];
      std::vector<int> dilations;
      for (size_t j = 0; j < layer_config["dilations"].size(); j++)
        dilations.push_back(layer_config["dilations"][j]);
      layer_array_params.push_back(
        wavenet::LayerArrayParams(layer_config["input_size"], layer_config["condition_size"], layer_config["head_size"],
                                  layer_config["channels"], layer_config["kernel_size"], dilations,
                                  layer_config["activation"], layer_config["gated"], layer_config["head_bias"]));
    }
    const bool with_head = config["head"] == NULL;
    const float head_scale = config["head_scale"];
    // Solves compilation issue on macOS Error: No matching constructor for
    // initialization of 'wavenet::WaveNet' Solution from
    // https://stackoverflow.com/a/73956681/3768284
    auto parametric_json = architecture == "CatWaveNet" ? config["parametric"] : nlohmann::json{};
    return std::make_unique<wavenet::WaveNet>(
      loudness, layer_array_params, head_scale, with_head, parametric_json, params, expected_sample_rate);
  }
  else
  {
    throw std::runtime_error("Unrecognized architecture");
  }
}
