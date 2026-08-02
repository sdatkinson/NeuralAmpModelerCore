#include "json_util.h"

#include <sstream>
#include <stdexcept>

namespace nam
{
namespace util
{
const nlohmann::json& RequireField(const nlohmann::json& j, const char* key, const char* context)
{
  if (!j.is_object())
  {
    throw std::runtime_error(std::string(context) + ": expected a JSON object containing '" + key + "'");
  }
  const auto it = j.find(key);
  if (it == j.end())
  {
    throw std::runtime_error(std::string(context) + ": missing required field '" + key + "'");
  }
  return *it;
}

namespace
{
// nlohmann::json's `.get<int>()` silently truncates a stored JSON float (e.g. `4.9` -> `4`)
// rather than rejecting it, which would let a hostile file smuggle a non-integral value
// through a dimension check. Require the underlying value to actually be a JSON integer.
int RequireIntegralValue(const nlohmann::json& value, const char* key, const char* context)
{
  if (!value.is_number_integer())
  {
    throw std::runtime_error(std::string(context) + ": field '" + key + "' must be an integer");
  }
  return value.get<int>();
}
} // namespace

int RequireDimension(const nlohmann::json& j, const char* key, const char* context, int maxValue)
{
  const nlohmann::json& value_json = RequireField(j, key, context);
  const int value = RequireIntegralValue(value_json, key, context);
  if (value < 1 || value > maxValue)
  {
    std::stringstream ss;
    ss << context << ": field '" << key << "' (" << value << ") must be between 1 and " << maxValue;
    throw std::runtime_error(ss.str());
  }
  return value;
}

int OptionalDimension(const nlohmann::json& j, const char* key, const char* context, int defaultValue, int maxValue)
{
  if (!j.is_object())
  {
    throw std::runtime_error(std::string(context) + ": expected a JSON object containing '" + key + "'");
  }
  const auto it = j.find(key);
  if (it == j.end() || it->is_null())
  {
    return defaultValue;
  }
  const int value = RequireIntegralValue(*it, key, context);
  if (value < 1 || value > maxValue)
  {
    std::stringstream ss;
    ss << context << ": field '" << key << "' (" << value << ") must be between 1 and " << maxValue;
    throw std::runtime_error(ss.str());
  }
  return value;
}

std::vector<int> RequireIntArray(const nlohmann::json& j, const char* key, const char* context, int minValue,
                                 int maxValue, bool allowEmpty, int maxLength)
{
  const nlohmann::json& arr = RequireField(j, key, context);
  if (!arr.is_array())
  {
    throw std::runtime_error(std::string(context) + ": field '" + key + "' must be an array");
  }
  if (!allowEmpty && arr.empty())
  {
    throw std::runtime_error(std::string(context) + ": field '" + key + "' must not be empty");
  }
  if (arr.size() > static_cast<size_t>(maxLength))
  {
    std::stringstream ss;
    ss << context << ": field '" << key << "' has " << arr.size() << " elements, which exceeds the limit of "
       << maxLength;
    throw std::runtime_error(ss.str());
  }

  std::vector<int> values;
  values.reserve(arr.size());
  for (const auto& element : arr)
  {
    const int value = RequireIntegralValue(element, key, context);
    if (value < minValue || value > maxValue)
    {
      std::stringstream ss;
      ss << context << ": field '" << key << "' contains " << value << ", which must be between " << minValue << " and "
         << maxValue;
      throw std::runtime_error(ss.str());
    }
    values.push_back(value);
  }
  return values;
}
}; // namespace util
}; // namespace nam
