#pragma once

#include <filesystem>
#include <iterator>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

// Version 2 DSP abstraction ==================================================

namespace dsp
{
class Params
{
};

class DSP
{
public:
  DSP();
  ~DSP();
  // The main interface for processing audio.
  // The incoming audio is given as a raw pointer-to-pointers.
  // The indexing is [channel][frame].
  // The output shall be a pointer-to-pointers of matching size.
  // This object instance will own the data referenced by the pointers and be
  // responsible for its allocation and deallocation.
  virtual double** Process(double** inputs, const size_t numChannels, const size_t numFrames) = 0;
  // Update the parameters of the DSP object according to the provided params.
  // Not declaring a pure virtual bc there's no concrete definition that can
  // use Params.
  // But, use this name :)
  // virtual void SetParams(Params* params) = 0;

protected:
  // Methods

  // Allocate mOutputPointers.
  // Assumes it's already null (Use _DeallocateOutputPointers()).
  void _AllocateOutputPointers(const size_t numChannels);
  // Ensure mOutputPointers is freed.
  void _DeallocateOutputPointers();

  size_t _GetNumChannels() const { return this->mOutputs.size(); };
  size_t _GetNumFrames() const { return this->_GetNumChannels() > 0 ? this->mOutputs[0].size() : 0; }
  // Return a pointer-to-pointers for the DSP's output buffers (all channels)
  // Assumes that ._PrepareBuffers()  was called recently enough.
  double** _GetPointers();
  // Resize mOutputs to (numChannels, numFrames) and ensure that the raw
  // pointers are also keeping up.
  virtual void _PrepareBuffers(const size_t numChannels, const size_t numFrames);
  // Resize the pointer-to-pointers for the vector-of-vectors.
  void _ResizePointers(const size_t numChannels);

  // Attributes

  // The output array into which the DSP module's calculations will be written.
  // Pointers to this member's data will be returned by .Process(), and std
  // Will ensure proper allocation.
  std::vector<std::vector<double>> mOutputs;
  // A pointer to pointers of which copies will be given out as the output of
  // .Process(). This object will ensure proper allocation and deallocation of
  // the first level; The second level points to .data() from mOutputs.
  double** mOutputPointers;
  size_t mOutputPointersSize;
};

// A class where a longer buffer of history is needed to correctly calculate
// the DSP algorithm (e.g. algorithms involving convolution).
//
// Hacky stuff:
// * Mono
// * Single-precision floats.
class History : public DSP
{
public:
  History();

protected:
  // Called at the end of the DSP, advance the hsitory index to the next open
  // spot.  Does not ensure that it's at a valid address.
  void _AdvanceHistoryIndex(const size_t bufferSize);
  // Drop the new samples into the history array.
  // Manages history array size
  void _UpdateHistory(double** inputs, const size_t numChannels, const size_t numFrames);

  // The history array that's used for DSP calculations.
  std::vector<float> mHistory;
  // How many samples previous are required.
  // Zero means that no history is required--only the current sample.
  size_t mHistoryRequired;
  // Location of the first sample in the current buffer.
  // Shall always be in the range [mHistoryRequired, mHistory.size()).
  size_t mHistoryIndex;

private:
  // Make sure that the history array is long enough.
  void _EnsureHistorySize(const size_t bufferSize);
  // Copy the end of the history back to the fron and reset mHistoryIndex
  void _RewindHistory();
};
}; // namespace dsp
