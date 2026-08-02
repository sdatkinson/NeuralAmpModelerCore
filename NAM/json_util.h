#pragma once

// Helpers for safely reading required fields out of an untrusted .nam model-file JSON
// document.
//
// nlohmann::json's `operator[]` on a `const` object asserts (via `JSON_ASSERT`, which is
// plain `assert()`) that the requested key exists before dereferencing it. Under
// `-DNDEBUG` (a typical plugin host Release build) a missing key is therefore undefined
// behavior rather than a thrown exception. `.nam` files are downloaded from the internet,
// so any field the loader treats as required must be looked up through the helpers below
// instead of `operator[]`.
//
// Deliberately kept free of Eigen so that consumers of this header don't need to pull it
// in (see NAM/util.h, which does include Eigen).

#include <stdexcept>
#include <string>
#include <vector>

#include "json.hpp"

namespace nam
{
namespace util
{
/// \brief Memory-safety bound for integer dimensions read from a model file (e.g. channel
/// counts, layer counts). This is not a claim about the size of legitimate models--it only
/// exists to stop a hostile value from driving an unbounded allocation.
constexpr int kMaxModelDimension = 1 << 16;

/// \brief Memory-safety bound for the LENGTH of arrays read from a model file (e.g. the
/// "layers" array, or a per-layer "dilations"/"kernel_sizes" array). Each element of such an
/// array typically drives construction of a heap-allocated object (a `Layer`, a `LayerArray`),
/// so an unbounded array length lets a tiny, highly-compressible file request an unbounded
/// number of allocations. This is not a claim about the size of legitimate models--real
/// WaveNets have tens of layers, not thousands--it only exists to bound the cost of parsing a
/// hostile file.
constexpr int kMaxModelArrayLength = 4096;

/// \brief Look up a required key in a JSON object, throwing if it is absent.
/// \param j The JSON object to search
/// \param key The required key
/// \param context Human-readable description of the enclosing object, used in the error
///        message (e.g. "WaveNet layer array 2")
/// \return Reference to the value at `key`
/// \throws std::runtime_error If `j` is not an object or `key` is not present
const nlohmann::json& RequireField(const nlohmann::json& j, const char* key, const char* context);

/// \brief Look up a required key and convert its value to `T`, throwing if the key is
/// absent or the value can't be converted.
/// \param j The JSON object to search
/// \param key The required key
/// \param context Human-readable description of the enclosing object, used in the error
///        message
/// \return The value at `key`, converted to `T`
/// \throws std::runtime_error If `j` is not an object, `key` is not present, or the value
///         can't be converted to `T`
template <typename T>
T RequireValue(const nlohmann::json& j, const char* key, const char* context)
{
  const nlohmann::json& value = RequireField(j, key, context);
  try
  {
    return value.get<T>();
  }
  catch (const nlohmann::json::exception& e)
  {
    throw std::runtime_error(std::string(context) + ": field '" + key + "' has the wrong type (" + e.what() + ")");
  }
}

/// \brief Look up a required integer dimension (e.g. a channel count), enforcing that it
/// falls within `[1, maxValue]`.
/// \param j The JSON object to search
/// \param key The required key
/// \param context Human-readable description of the enclosing object, used in the error
///        message
/// \param maxValue Inclusive upper bound on the returned value
/// \return The validated dimension
/// \throws std::runtime_error If the key is absent, isn't an integer (a JSON float such as
///         `4.9` is rejected rather than silently truncated), or is outside `[1, maxValue]`
int RequireDimension(const nlohmann::json& j, const char* key, const char* context, int maxValue = kMaxModelDimension);

/// \brief Look up an OPTIONAL integer dimension, enforcing that it falls within
/// `[1, maxValue]` when present. Use this for fields that default to a fixed value when
/// absent (e.g. `in_channels` defaulting to 1)--absence is fine, but a present-and-hostile
/// value (e.g. 0, negative, non-integral, or absurdly large) is not.
/// \param j The JSON object to search
/// \param key The optional key
/// \param context Human-readable description of the enclosing object, used in the error
///        message
/// \param defaultValue Value to return if `key` is absent
/// \param maxValue Inclusive upper bound on the returned value
/// \return `defaultValue` if `key` is absent, otherwise the validated value
/// \throws std::runtime_error If `j` is not an object, or `key` is present but isn't an
///         integer or is outside `[1, maxValue]`
int OptionalDimension(const nlohmann::json& j, const char* key, const char* context, int defaultValue,
                      int maxValue = kMaxModelDimension);

/// \brief Look up a required array of integers, enforcing that every element falls within
/// `[minValue, maxValue]` and that the array itself isn't longer than `maxLength`.
/// \param j The JSON object to search
/// \param key The required key
/// \param context Human-readable description of the enclosing object, used in the error
///        message
/// \param minValue Inclusive lower bound on every element
/// \param maxValue Inclusive upper bound on every element
/// \param allowEmpty Whether an empty array is acceptable
/// \param maxLength Inclusive upper bound on the array's length (see `kMaxModelArrayLength`)
/// \return The validated array
/// \throws std::runtime_error If the key is absent, isn't an array of integers (a JSON float
///         element such as `4.9` is rejected rather than silently truncated), is empty when
///         `allowEmpty` is false, is longer than `maxLength`, or contains an out-of-range
///         element
std::vector<int> RequireIntArray(const nlohmann::json& j, const char* key, const char* context, int minValue,
                                 int maxValue, bool allowEmpty = false, int maxLength = kMaxModelArrayLength);
}; // namespace util
}; // namespace nam
