#include <algorithm>
#include <cctype>

#include "util.h"

std::string util::lowercase(const std::string& s)
{
  std::string out(s);
  std::transform(s.begin(), s.end(), out.begin(), [](unsigned char c) { return std::tolower(c); });
  return out;
}

void util::init_matrix(Eigen::MatrixXf& matrix)
{
  for(auto i = 0; i < matrix.rows(); ++i)
    for(auto j = 0; j < matrix.cols(); ++j)
      matrix(i, j) = 0.0f;
}
