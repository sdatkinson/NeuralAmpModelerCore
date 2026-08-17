#include <cassert>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "json.hpp"

#include "NAM/nam_file.h"

namespace test_nam_file
{
namespace
{

class TemporaryNamFile
{
public:
  TemporaryNamFile(const std::string& filename, const std::string& contents)
  : path(std::filesystem::temp_directory_path() / filename)
  {
    std::ofstream output(path);
    output << contents;
  }

  ~TemporaryNamFile() { std::filesystem::remove(path); }

  const std::filesystem::path path;
};

nlohmann::json minimum_valid_nam_file()
{
  return {{"version", "0.7.0"},
          {"architecture", "Test"},
          {"config", nlohmann::json::object()},
          {"weights", nlohmann::json::array()}};
}

void assert_validation_error(const TemporaryNamFile& file, const std::string& expected_detail)
{
  try
  {
    nam::validate_nam_file(file.path);
    assert(false);
  }
  catch (const nam::NamFileValidationError& error)
  {
    const std::string message = error.what();
    assert(message.find(file.path.string()) != std::string::npos);
    assert(message.find(expected_detail) != std::string::npos);
  }
}

} // namespace

void test_accepts_minimum_valid_file()
{
  const TemporaryNamFile file("nam_core_minimum_valid_file_test.nam", minimum_valid_nam_file().dump());
  const auto config = nam::validate_nam_file(file.path);

  assert(config.is_object());
}

void test_rejects_non_object_json()
{
  const TemporaryNamFile file("nam_core_non_object_file_test.nam", "[]");
  assert_validation_error(file, "object");
}

void test_rejects_missing_required_keys()
{
  const std::vector<std::string> required_keys{"version", "architecture", "config", "weights"};
  for (const auto& key : required_keys)
  {
    auto config = minimum_valid_nam_file();
    config.erase(key);
    const TemporaryNamFile file("nam_core_missing_" + key + "_test.nam", config.dump());
    assert_validation_error(file, key);
  }
}

} // namespace test_nam_file
