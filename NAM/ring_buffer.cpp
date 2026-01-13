#include "ring_buffer.h"

namespace nam
{

void RingBuffer::Reset(const int channels, const int max_buffer_size)
{
  // Store the max buffer size for external queries
  _max_buffer_size = max_buffer_size;
  
  // Calculate storage size: 2 * max_lookback + max_buffer_size
  // This ensures we have enough room for:
  // - max_lookback at the start (for history after rewind)
  // - max_buffer_size in the middle (for writes/reads)
  // - no aliasing when rewinding
  const long storage_size = 2 * _max_lookback + max_buffer_size;
  _storage.resize(channels, storage_size);
  _storage.setZero();
  // Initialize write position to max_lookback to leave room for history
  // Zero the storage behind the starting write position (for lookback)
  if (_max_lookback > 0)
  {
    _storage.leftCols(_max_lookback).setZero();
  }
  _write_pos = _max_lookback;
}

void RingBuffer::Write(const Eigen::MatrixXf& input, const int num_frames)
{
  // Check if we need to rewind
  if (NeedsRewind(num_frames))
    Rewind();

  // Write the input data at the write position
  _storage.middleCols(_write_pos, num_frames) = input.leftCols(num_frames);
}

Eigen::Block<Eigen::MatrixXf> RingBuffer::Read(const int num_frames, const long lookback)
{
  long read_pos = GetReadPos(lookback);

  // Handle wrapping if read_pos is negative or beyond storage bounds
  if (read_pos < 0)
  {
    // Wrap around to the end of the storage
    read_pos = _storage.cols() + read_pos;
  }

  // Ensure we don't read beyond storage bounds
  // If read_pos + num_frames would exceed storage, we need to wrap or clamp
  if (read_pos + num_frames > (long)_storage.cols())
  {
    // For now, clamp to available space
    // This shouldn't happen if storage is sized correctly, but handle gracefully
    long available = _storage.cols() - read_pos;
    if (available < num_frames)
    {
      // This is an error condition - storage not sized correctly
      // Return what we can (shouldn't happen in normal usage)
      return _storage.block(0, read_pos, _storage.rows(), available);
    }
  }

  return _storage.block(0, read_pos, _storage.rows(), num_frames);
}

void RingBuffer::Advance(const int num_frames)
{
  _write_pos += num_frames;
}

bool RingBuffer::NeedsRewind(const int num_frames) const
{
  return _write_pos + num_frames > (long)_storage.cols();
}

void RingBuffer::Rewind()
{
  if (_max_lookback == 0)
  {
    // No history to preserve, just reset
    _write_pos = 0;
    return;
  }

  // Copy the max lookback amount of history back to the beginning
  // This is the history that will be needed for lookback reads
  const long copy_start = _write_pos - _max_lookback;
  if (copy_start >= 0 && copy_start < (long)_storage.cols() && _max_lookback > 0)
  {
    // Copy _max_lookback samples from before the write position to the start
    _storage.leftCols(_max_lookback) = _storage.middleCols(copy_start, _max_lookback);
  }
  // Reset write position to just after the copied history
  _write_pos = _max_lookback;
}

long RingBuffer::GetReadPos(const long lookback) const
{
  return _write_pos - lookback;
}
} // namespace nam
