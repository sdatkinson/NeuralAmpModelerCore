#include "nam_file.h"

#include <array>
#include <fstream>
#include <string>

namespace nam
{
void validate_nam_json(const nlohmann::json& config, const std::string& source)
{
  if (!config.is_object())
    throw NamFileValidationError("Invalid " + source + ": root JSON value must be an object.");

  constexpr std::array<const char*, 4> required_keys{"version", "architecture", "config", "weights"};
  for (const auto* key : required_keys)
  {
    if (config.find(key) == config.end())
      throw NamFileValidationError("Invalid " + source + ": missing required key \"" + key + "\".");
  }

  // Checked here rather than left to .get<std::string>() at the point of use, which would throw
  // nlohmann::json::type_error -- a different exception type from every other failure on this path.
  constexpr std::array<const char*, 2> string_keys{"version", "architecture"};
  for (const auto* key : string_keys)
  {
    if (!config.at(key).is_string())
      throw NamFileValidationError("Invalid " + source + ": \"" + key + "\" must be a string.");
  }
}

nlohmann::json validate_nam_file(const std::filesystem::path& filename)
{
  if (!std::filesystem::exists(filename))
    throw NamFileValidationError("Could not validate .nam file [" + filename.string() + "]: file does not exist.");

  std::ifstream input(filename);
  if (!input.is_open())
    throw NamFileValidationError("Could not validate .nam file [" + filename.string() + "]: file could not be read.");

  nlohmann::json config;
  try
  {
    input >> config;
  }
  catch (const nlohmann::json::parse_error& error)
  {
    throw NamFileValidationError("Could not parse .nam file [" + filename.string() + "]: " + error.what());
  }

  validate_nam_json(config, ".nam file [" + filename.string() + "]");

  return config;
}
} // namespace nam
