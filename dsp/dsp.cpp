#include <algorithm> // std::max_element
#include <algorithm>
#include <cmath> // pow, tanh, expf
#include <filesystem>
#include <fstream>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "dsp.h"

// ============================================================================
// Implementation of Version 2 interface

dsp::DSP::DSP()
: mOutputPointers(nullptr)
, mOutputPointersSize(0)
{
}

dsp::DSP::~DSP()
{
  this->_DeallocateOutputPointers();
};

void dsp::DSP::_AllocateOutputPointers(const size_t numChannels)
{
  if (this->mOutputPointers != nullptr)
    throw std::runtime_error("Tried to re-allocate over non-null mOutputPointers");
  this->mOutputPointers = new DSP_SAMPLE*[numChannels];
  if (this->mOutputPointers == nullptr)
    throw std::runtime_error("Failed to allocate pointer to output buffer!\n");
  this->mOutputPointersSize = numChannels;
}

void dsp::DSP::_DeallocateOutputPointers()
{
  if (this->mOutputPointers != nullptr)
  {
    delete[] this->mOutputPointers;
    this->mOutputPointers = nullptr;
  }
  if (this->mOutputPointers != nullptr)
    throw std::runtime_error("Failed to deallocate output pointer!");
  this->mOutputPointersSize = 0;
}

DSP_SAMPLE** dsp::DSP::_GetPointers()
{
  for (auto c = 0; c < this->_GetNumChannels(); c++)
    this->mOutputPointers[c] = this->mOutputs[c].data();
  return this->mOutputPointers;
}

void dsp::DSP::_PrepareBuffers(const size_t numChannels, const size_t numFrames)
{
  const size_t oldFrames = this->_GetNumFrames();
  const size_t oldChannels = this->_GetNumChannels();

  const bool resizeChannels = oldChannels != numChannels;
  const bool resizeFrames = resizeChannels || (oldFrames != numFrames);
  if (resizeChannels)
  {
    this->mOutputs.resize(numChannels);
    this->_ResizePointers(numChannels);
  }
  if (resizeFrames)
    for (auto c = 0; c < numChannels; c++)
      this->mOutputs[c].resize(numFrames);
}

void dsp::DSP::_ResizePointers(const size_t numChannels)
{
  if (this->mOutputPointersSize == numChannels)
    return;
  this->_DeallocateOutputPointers();
  this->_AllocateOutputPointers(numChannels);
}

dsp::History::History()
: DSP()
, mHistoryRequired(0)
, mHistoryIndex(0)
{
}

void dsp::History::_AdvanceHistoryIndex(const size_t bufferSize)
{
  this->mHistoryIndex += bufferSize;
}

void dsp::History::_EnsureHistorySize(const size_t bufferSize)
{
  const size_t repeatSize = std::max(bufferSize, this->mHistoryRequired);
  const size_t requiredHistoryArraySize = 10 * repeatSize; // Just so we don't spend too much time copying back.
  if (this->mHistory.size() < requiredHistoryArraySize)
  {
    this->mHistory.resize(requiredHistoryArraySize);
    std::fill(this->mHistory.begin(), this->mHistory.end(), 0.0f);
    this->mHistoryIndex = this->mHistoryRequired; // Guaranteed to be less than
                                                  // requiredHistoryArraySize
  }
}

void dsp::History::_RewindHistory()
{
  // TODO memcpy?  Should be fine w/ history array being >2x the history length.
  for (size_t i = 0, j = this->mHistoryIndex - this->mHistoryRequired; i < this->mHistoryRequired; i++, j++)
    this->mHistory[i] = this->mHistory[j];
  this->mHistoryIndex = this->mHistoryRequired;
}

void dsp::History::_UpdateHistory(DSP_SAMPLE** inputs, const size_t numChannels, const size_t numFrames)
{
  this->_EnsureHistorySize(numFrames);
  if (numChannels < 1)
    throw std::runtime_error("Zero channels?");
  if (this->mHistoryIndex + numFrames >= this->mHistory.size())
    this->_RewindHistory();
  // Grabs channel 1, drops hannel 2.
  for (size_t i = 0, j = this->mHistoryIndex; i < numFrames; i++, j++)
    // Convert down to float here.
    this->mHistory[j] = (float)inputs[0][i];
}
