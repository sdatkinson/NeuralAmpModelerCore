#include "ring_buffer.h"
#include <cassert>

namespace nam
{

void RingBuffer::Reset(const long channels, const int max_buffer_size)
{
  // Store the max buffer size for external queries
  _max_buffer_size = max_buffer_size;

  // Calculate storage size: 2 * max_lookback + max_buffer_size
  // This ensures we have enough room for:
  // - max_lookback at the start (for history after rewind)
  // - max_buffer_size in the middle (for writes/reads)
  // - no aliasing when rewinding
  const long storage_size = 2 * _max_lookback + max_buffer_size;
  _storage.resize((nam::Index)channels, (nam::Index)storage_size);
  _storage.setZero();
  // Initialize write position to max_lookback to leave room for history
  // Zero the storage behind the starting write position (for lookback)
  if (_max_lookback > 0)
  {
    _storage.leftCols((nam::Index)_max_lookback).setZero();
  }
  _write_pos = _max_lookback;
}

void RingBuffer::Write(const Eigen::MatrixXf& input, const int num_frames)
{
  // Assert that num_frames doesn't exceed max buffer size
  assert(num_frames <= _max_buffer_size && "Write: num_frames must not exceed max_buffer_size");

  // Check if we need to rewind
  if (NeedsRewind(num_frames))
    Rewind();

  // Write the input data at the write position
  // NOTE: This function assumes that `input` is a full, pre-allocated MatrixXf
  //       covering the entire valid buffer range. Callers should not pass Block
  //       expressions across the API boundary; instead, pass the full buffer and
  //       slice inside the callee. This avoids Eigen evaluating Blocks into
  //       temporaries (which would allocate) when binding to MatrixXf.
  const long channels = _storage.rows();
  const int copy_cols = num_frames;

  for (int col = 0; col < copy_cols; ++col)
  {
    for (long row = 0; row < channels; ++row)
    {
      _storage(row, _write_pos + col) = input(row, col);
    }
  }
}

Eigen::Block<Eigen::MatrixXf> RingBuffer::Read(const int num_frames, const long lookback)
{
  // Assert that lookback doesn't exceed max_lookback
  assert(lookback <= _max_lookback && "Read: lookback must not exceed max_lookback");

  // Assert that num_frames doesn't exceed max buffer size
  assert(num_frames <= _max_buffer_size && "Read: num_frames must not exceed max_buffer_size");

  long read_pos = _write_pos - lookback;

  // Assert that read_pos is non-negative
  // (Asserted by the access to _storage.block())
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

  // Assert that write pointer is far enough along to avoid aliasing when copying
  // We copy from position (_write_pos - _max_lookback) to position 0
  // For no aliasing, we need: _write_pos - _max_lookback >= _max_lookback
  // Which simplifies to: _write_pos >= 2 * _max_lookback
  assert(_write_pos >= 2 * _max_lookback
         && "Write pointer must be at least 2 * max_lookback to avoid aliasing during rewind");

  // Copy the max lookback amount of history back to the beginning
  // This is the history that will be needed for lookback reads
  const long copy_start = _write_pos - _max_lookback;
  assert(copy_start >= 0 && copy_start < (long)_storage.cols() && "Copy start position must be within storage bounds");

  // Copy _max_lookback samples from before the write position to the start
  _storage.leftCols(_max_lookback) = _storage.middleCols(copy_start, _max_lookback);

  // Reset write position to just after the copied history
  _write_pos = _max_lookback;
}
} // namespace nam
