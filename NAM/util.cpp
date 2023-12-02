#include <algorithm>
#include <cctype>

#include "util.h"

std::string nam::util::lowercase(const std::string& s)
{
  std::string out(s);
  std::transform(s.begin(), s.end(), out.begin(), [](unsigned char c) { return std::tolower(c); });
  return out;
}
