#pragma once

#include <Eigen/Dense>

namespace nam
{
// Ring buffer for managing Eigen::MatrixXf buffers with write/read pointers
class RingBuffer
{
public:
  RingBuffer() {};
  // Initialize/resize storage
  // :param channels: Number of channels (rows in the storage matrix)
  // :param max_buffer_size: Maximum amount that will be written or read at once
  void Reset(const int channels, const int max_buffer_size);
  // Write new data at write pointer
  // :param input: Input matrix (channels x num_frames)
  // :param num_frames: Number of frames to write
  // NOTE: This function expects a full, pre-allocated, column-major MatrixXf
  //       covering the entire valid buffer range. Callers should not pass
  //       Block expressions (e.g. .leftCols()) across the API boundary; instead,
  //       pass the full buffer and slice inside the callee. This avoids Eigen
  //       evaluating Blocks into temporaries (which would allocate) when
  //       binding to MatrixXf.
  void Write(const Eigen::MatrixXf& input, const int num_frames);
  // Read data with optional lookback
  // :param num_frames: Number of frames to read
  // :param lookback: Number of frames to look back from write pointer (default 0)
  // :return: Block reference to the storage data
  Eigen::Block<Eigen::MatrixXf> Read(const int num_frames, const long lookback = 0);
  // Advance write pointer
  // :param num_frames: Number of frames to advance
  void Advance(const int num_frames);
  // Get max buffer size (the value passed to Reset())
  int GetMaxBufferSize() const { return _max_buffer_size; }
  // Get number of channels (rows)
  int GetChannels() const { return _storage.rows(); }
  // Set the max lookback (maximum history needed when rewinding)
  void SetMaxLookback(const long max_lookback) { _max_lookback = max_lookback; }

private:
  // Wrap buffer when approaching end (called automatically if needed)
  void Rewind();
  // Check if rewind is needed before `num_frames` are written or read
  // :param num_frames: Number of frames that will be written
  // :return: true if rewind is needed
  bool NeedsRewind(const int num_frames) const;
  // Get current write position
  long GetWritePos() const { return _write_pos; }

  Eigen::MatrixXf _storage; // channels x storage_size
  long _write_pos = 0; // Current write position
  long _max_lookback = 0; // Maximum lookback needed when rewinding
  int _max_buffer_size = 0; // Maximum buffer size passed to Reset()
};
} // namespace nam
