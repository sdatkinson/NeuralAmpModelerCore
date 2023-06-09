#pragma once

// Utilities

#include <string>
#include <Eigen/Dense>  // Eigen::MatrixXf

namespace util
{
std::string lowercase(const std::string& s);
void init_matrix(Eigen::MatrixXf& matrix);
}; // namespace util
