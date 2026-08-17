#pragma once

#include <filesystem>
#include <stdexcept>

#include "json.hpp"

namespace nam
{
/// \brief Indicates that a .nam file failed validation while loading
class NamFileValidationError : public std::runtime_error
{
public:
  using std::runtime_error::runtime_error;
};

/// \brief Parse and validate a .nam file
/// \param filename Path to the .nam file
/// \return The parsed model configuration
/// \throws NamFileValidationError If the file cannot be read, parsed, or does not contain the minimum required fields
nlohmann::json validate_nam_file(const std::filesystem::path& filename);
} // namespace nam
