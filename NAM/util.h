#pragma once

// Utilities

#include <string>
#include <Eigen/Dense> // Eigen::MatrixXf

namespace nam
{
namespace util
{
/// \brief Convert a string to lowercase
/// \param s Input string
/// \return Lowercase version of the input string
std::string lowercase(const std::string& s);
}; // namespace util
}; // namespace nam
