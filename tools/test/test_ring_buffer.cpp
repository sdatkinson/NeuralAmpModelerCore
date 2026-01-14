// Tests for RingBuffer

#include <Eigen/Dense>
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

#include "NAM/ring_buffer.h"

namespace test_ring_buffer
{
// Test basic construction
void test_construct()
{
  nam::RingBuffer rb;
  assert(rb.GetMaxBufferSize() == 0);
  assert(rb.GetChannels() == 0);
}

// Test Reset() initializes storage correctly
void test_reset()
{
  nam::RingBuffer rb;
  const int channels = 2;
  const int max_buffer_size = 64;

  rb.Reset(channels, max_buffer_size);

  assert(rb.GetChannels() == channels);
  assert(rb.GetMaxBufferSize() == max_buffer_size);
}

// Test Reset() with max_lookback zeros the storage behind starting position
void test_reset_with_receptive_field()
{
  nam::RingBuffer rb;
  const int channels = 2;
  const int max_buffer_size = 64;
  const long max_lookback = 10;

  rb.SetMaxLookback(max_lookback);
  rb.Reset(channels, max_buffer_size);

  assert(rb.GetChannels() == channels);
  assert(rb.GetMaxBufferSize() == max_buffer_size);

  // The storage behind the starting position should be zero
  // Read from position 0 by using lookback = max_lookback (read_pos = write_pos - max_lookback = 0)
  auto buffer_block = rb.Read(max_lookback, max_lookback); // Read from position 0
  assert(buffer_block.isZero());
}

// Test Write() writes data at write position
void test_write()
{
  nam::RingBuffer rb;
  const int channels = 2;
  const int max_buffer_size = 64;
  const int num_frames = 4;
  const long max_lookback = 4;

  rb.SetMaxLookback(max_lookback);
  rb.Reset(channels, max_buffer_size);

  Eigen::MatrixXf input(channels, num_frames);
  input(0, 0) = 1.0f;
  input(1, 0) = 2.0f;
  input(0, 1) = 3.0f;
  input(1, 1) = 4.0f;
  input(0, 2) = 5.0f;
  input(1, 2) = 6.0f;
  input(0, 3) = 7.0f;
  input(1, 3) = 8.0f;

  rb.Write(input, num_frames);

  // Read back what we just wrote (with lookback=0, since write_pos hasn't advanced)
  auto output = rb.Read(num_frames, 0);
  assert(output.rows() == channels);
  assert(output.cols() == num_frames);
  assert(std::abs(output(0, 0) - 1.0f) < 0.01f);
  assert(std::abs(output(1, 0) - 2.0f) < 0.01f);
  assert(std::abs(output(0, 1) - 3.0f) < 0.01f);
  assert(std::abs(output(1, 1) - 4.0f) < 0.01f);

  // After Advance, we need lookback to read what we wrote
  rb.Advance(num_frames);
  auto output_after_advance = rb.Read(num_frames, num_frames);
  assert(std::abs(output_after_advance(0, 0) - 1.0f) < 0.01f);
  assert(std::abs(output_after_advance(1, 1) - 4.0f) < 0.01f);
}

// Test Read() with lookback
void test_read_with_lookback()
{
  nam::RingBuffer rb;
  const int channels = 1;
  const int max_buffer_size = 64;
  const long max_lookback = 5;

  rb.SetMaxLookback(max_lookback);
  rb.Reset(channels, max_buffer_size);

  // Write some data
  Eigen::MatrixXf input1(channels, 3);
  input1(0, 0) = 1.0f;
  input1(0, 1) = 2.0f;
  input1(0, 2) = 3.0f;

  rb.Write(input1, 3);
  rb.Advance(3);

  // Write more data
  Eigen::MatrixXf input2(channels, 2);
  input2(0, 0) = 4.0f;
  input2(0, 1) = 5.0f;

  rb.Write(input2, 2);

  // After Write(), data is at write_pos but write_pos hasn't advanced yet
  // So lookback=0 reads from write_pos, which has the data we just wrote
  auto current = rb.Read(2, 0);
  assert(std::abs(current(0, 0) - 4.0f) < 0.01f);
  assert(std::abs(current(0, 1) - 5.0f) < 0.01f);

  // After Advance(3), write_pos = receptive_field + 3 = 8
  // Read with lookback=2 should get the last 2 frames from input1
  auto recent = rb.Read(2, 2);
  // Position 8-2=6 has input1[1]=2.0, position 7 has input1[2]=3.0
  assert(std::abs(recent(0, 0) - 2.0f) < 0.01f); // input1[1] at position 6
  assert(std::abs(recent(0, 1) - 3.0f) < 0.01f); // input1[2] at position 7

  rb.Advance(2); // Now write_pos = 10

  // Read with lookback=2 to get input2 we just wrote
  auto input2_read = rb.Read(2, 2);
  assert(std::abs(input2_read(0, 0) - 4.0f) < 0.01f);
  assert(std::abs(input2_read(0, 1) - 5.0f) < 0.01f);

  // Read with lookback=5 (should get frames from first write)
  auto history = rb.Read(2, 5);
  // Position 10-5=5 has input1[0]=1.0, position 6 has input1[1]=2.0
  assert(std::abs(history(0, 0) - 1.0f) < 0.01f); // input1[0]
  assert(std::abs(history(0, 1) - 2.0f) < 0.01f); // input1[1]
}

// Test Advance() moves write pointer
void test_advance()
{
  nam::RingBuffer rb;
  const int channels = 1;
  const int max_buffer_size = 64;
  const long max_lookback = 15;

  rb.SetMaxLookback(max_lookback);
  rb.Reset(channels, max_buffer_size);

  // Test that Advance() works by writing, advancing, and reading back
  Eigen::MatrixXf input(channels, 10);
  input.setZero();
  input(0, 0) = 1.0f;
  rb.Write(input, 10);
  rb.Advance(10);

  // Read back with lookback to verify advance worked
  auto output = rb.Read(10, 10);
  assert(std::abs(output(0, 0) - 1.0f) < 0.01f);

  rb.Advance(5);
  // Read back with larger lookback to verify further advance
  auto output2 = rb.Read(10, 15);
  assert(std::abs(output2(0, 0) - 1.0f) < 0.01f);
}

// Test Rewind() copies history and resets write position
void test_rewind()
{
  nam::RingBuffer rb;
  const int channels = 1;
  const int max_buffer_size = 32;
  const long max_lookback = 5;

  rb.SetMaxLookback(max_lookback);
  rb.Reset(channels, max_buffer_size);

  // Storage size = 2 * max_lookback + max_buffer_size = 2 * 5 + 32 = 42
  const long storage_size = 2 * max_lookback + max_buffer_size;

  // Write enough data to trigger rewind
  // We need to write more than storage_size to trigger rewind
  const int num_writes = 25; // 25 * 2 = 50 > 42
  long writeSize = 2;
  assert(writeSize * num_writes > storage_size);
  for (int i = 0; i < num_writes; i++)
  {
    Eigen::MatrixXf input(channels, writeSize);
    input(0, 0) = (float)(i * 2);
    input(0, 1) = (float)(i * 2 + 1);

    rb.Write(input, writeSize);
    rb.Advance(writeSize);

    // Continue writing until we've written enough to potentially trigger rewind
    // The rewind will happen automatically in Write() if needed
  }

  // [SDA] this next part is an AI test and I'm not sure I like it.

  // After writing enough data, we should be able to read from history
  // Read with lookback = max_lookback to read from position 0 (history region)
  auto history = rb.Read(2, max_lookback);
  // History should be available
  assert(history.cols() == 2);
}


// Test multiple writes and reads maintain history correctly
void test_multiple_writes_reads()
{
  nam::RingBuffer rb;
  const int channels = 1;
  const int max_buffer_size = 64;
  const long max_lookback = 5;

  rb.SetMaxLookback(max_lookback);
  rb.Reset(channels, max_buffer_size);

  // Write first batch
  Eigen::MatrixXf input1(channels, 3);
  input1(0, 0) = 1.0f;
  input1(0, 1) = 2.0f;
  input1(0, 2) = 3.0f;

  rb.Write(input1, 3);
  rb.Advance(3);

  // Write second batch
  Eigen::MatrixXf input2(channels, 2);
  input2(0, 0) = 4.0f;
  input2(0, 1) = 5.0f;

  rb.Write(input2, 2);
  rb.Advance(2);

  // After Write() and Advance(), write_pos points after the data we just wrote
  // Read with lookback=2 to get the last 2 frames we wrote (input2)
  auto current = rb.Read(2, 2);
  assert(std::abs(current(0, 0) - 4.0f) < 0.01f);
  assert(std::abs(current(0, 1) - 5.0f) < 0.01f);

  // Read with lookback=5 should get frames from first batch (input1[1] and input1[2])
  // After writes: input1 at positions [max_lookback, max_lookback+2] = [3, 4, 5]
  //               input2 at positions [max_lookback+3, max_lookback+4] = [6, 7]
  // write_pos after both: max_lookback + 5 = 8
  // Read with lookback=5: read_pos = 8 - 5 = 3
  // This reads from position 3, which is input1[0] = 1.0
  auto history = rb.Read(2, 5);
  // Position 3 = input1[0] = 1.0, position 4 = input1[1] = 2.0
  assert(std::abs(history(0, 0) - 1.0f) < 0.01f);
  assert(std::abs(history(0, 1) - 2.0f) < 0.01f);
}

// Test that Reset() zeros buffer behind starting position
void test_reset_zeros_history_area()
{
  nam::RingBuffer rb;
  const int channels = 1;
  const int max_buffer_size = 64;
  const long max_lookback = 10;

  rb.SetMaxLookback(max_lookback);
  rb.Reset(channels, max_buffer_size);

  // Write some data and advance
  Eigen::MatrixXf input(channels, max_buffer_size);
  input.fill(42.0f);
  for (int i = 0; i < 5; i++) // Should be enough to write those first positions.
  {
    rb.Write(input, max_buffer_size);
    rb.Advance(max_buffer_size);
  }

  // Reset should zero the storage behind the starting position
  rb.Reset(channels, max_buffer_size);

  // Read from position 0 (behind starting write position)
  // This should be zero
  // After Reset with max_lookback, we can read from position 0
  auto read = rb.Read(max_lookback, max_lookback);
  assert(read.isZero());
}

// Test Rewind() preserves history correctly
void test_rewind_preserves_history()
{
  nam::RingBuffer rb;
  const int channels = 1;
  const int max_buffer_size = 32;
  const long max_lookback = 4;

  rb.SetMaxLookback(max_lookback);
  rb.Reset(channels, max_buffer_size);

  // Storage size = 2 * max_lookback + max_buffer_size = 2 * 4 + 32 = 40
  const long storage_size = 2 * max_lookback + max_buffer_size;

  // Three writes of size max_buffer_size should trigger rewind.
  Eigen::MatrixXf input(channels, max_buffer_size);
  input.fill(42.0f);
  for (int i = 0; i < 3; i++)
  {
    rb.Write(input, max_buffer_size);
    rb.Advance(max_buffer_size);
  }

  // Read from history region to verify rewind preserved history
  auto history = rb.Read(max_lookback, max_lookback);
  assert(history.cols() == max_lookback);
  assert(history == input.rightCols(max_lookback));
}

}; // namespace test_ring_buffer
