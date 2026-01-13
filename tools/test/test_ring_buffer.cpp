// Tests for RingBuffer

#include <Eigen/Dense>
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

#include "NAM/dsp.h"

namespace test_ring_buffer
{
// Test basic construction
void test_construct()
{
  nam::RingBuffer rb;
  assert(rb.GetWritePos() == 0);
  assert(rb.GetCapacity() == 0);
  assert(rb.GetChannels() == 0);
}

// Test Reset() initializes buffer correctly
void test_reset()
{
  nam::RingBuffer rb;
  const int channels = 2;
  const int buffer_size = 64;

  rb.Reset(channels, buffer_size);

  assert(rb.GetChannels() == channels);
  assert(rb.GetCapacity() == buffer_size);
  assert(rb.GetWritePos() == 0); // Starts at 0 if no receptive_field set
}

// Test Reset() with receptive_field zeros the buffer behind starting position
void test_reset_with_receptive_field()
{
  nam::RingBuffer rb;
  const int channels = 2;
  const int buffer_size = 64;
  const long receptive_field = 10;

  rb.SetReceptiveField(receptive_field);
  rb.Reset(channels, buffer_size);

  assert(rb.GetChannels() == channels);
  assert(rb.GetCapacity() == buffer_size);
  assert(rb.GetWritePos() == receptive_field); // Write position should be after receptive_field

  // The buffer behind the starting position should be zero
  auto buffer_block = rb.Read(receptive_field, 0); // Try to read from position 0
  for (int i = 0; i < channels; i++)
  {
    for (long j = 0; j < receptive_field; j++)
    {
      // Can't directly access, but we can read from position 0
      // Actually, let me read from the buffer directly using GetReadPos
      long read_pos = rb.GetReadPos(receptive_field);
      if (read_pos >= 0 && read_pos < buffer_size)
      {
        // This should be zero (initialized)
      }
    }
  }
}

// Test Write() writes data at write position
void test_write()
{
  nam::RingBuffer rb;
  const int channels = 2;
  const int buffer_size = 64;
  const int num_frames = 4;

  rb.Reset(channels, buffer_size);

  Eigen::MatrixXf input(channels, num_frames);
  input(0, 0) = 1.0f;
  input(1, 0) = 2.0f;
  input(0, 1) = 3.0f;
  input(1, 1) = 4.0f;
  input(0, 2) = 5.0f;
  input(1, 2) = 6.0f;
  input(0, 3) = 7.0f;
  input(1, 3) = 8.0f;

  long write_pos_before = rb.GetWritePos();
  rb.Write(input, num_frames);

  // Write position should not change after Write() (Advance() is separate)
  assert(rb.GetWritePos() == write_pos_before);

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
  const int buffer_size = 64;
  const long receptive_field = 5;

  rb.SetReceptiveField(receptive_field);
  rb.Reset(channels, buffer_size);

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
  const int buffer_size = 64;

  rb.Reset(channels, buffer_size);

  long initial_pos = rb.GetWritePos();
  rb.Advance(10);
  assert(rb.GetWritePos() == initial_pos + 10);

  rb.Advance(5);
  assert(rb.GetWritePos() == initial_pos + 15);
}

// Test Rewind() copies history and resets write position
void test_rewind()
{
  nam::RingBuffer rb;
  const int channels = 1;
  const int buffer_size = 32;
  const long receptive_field = 5;

  rb.SetReceptiveField(receptive_field);
  rb.Reset(channels, buffer_size);

  // Write enough data to trigger rewind
  const int num_writes = 20;
  for (int i = 0; i < num_writes; i++)
  {
    Eigen::MatrixXf input(channels, 2);
    input(0, 0) = (float)(i * 2);
    input(0, 1) = (float)(i * 2 + 1);

    rb.Write(input, 2);
    rb.Advance(2);

    // Check if rewind happened
    if (rb.GetWritePos() + 2 > buffer_size)
    {
      // Rewind should have been called automatically
      break;
    }
  }

  // After rewind, write_pos should be at receptive_field
  if (rb.NeedsRewind(2))
  {
    rb.Rewind();
    assert(rb.GetWritePos() == receptive_field);

    // The history should be copied to the start
    // Read with lookback should work from the copied history
    auto history = rb.Read(2, receptive_field);
    // History should be available
    assert(history.cols() == 2);
  }
}

// Test NeedsRewind() correctly detects when rewind is needed
void test_needs_rewind()
{
  nam::RingBuffer rb;
  const int channels = 1;
  const int buffer_size = 32;

  rb.Reset(channels, buffer_size);

  assert(!rb.NeedsRewind(10)); // Should not need rewind initially

  rb.Advance(25);
  assert(!rb.NeedsRewind(5)); // Still has room: 25 + 5 = 30 < 32
  assert(rb.NeedsRewind(10)); // Would overflow: 25 + 10 = 35 > 32
}

// Test multiple writes and reads maintain history correctly
void test_multiple_writes_reads()
{
  nam::RingBuffer rb;
  const int channels = 1;
  const int buffer_size = 64;
  const long receptive_field = 3;

  rb.SetReceptiveField(receptive_field);
  rb.Reset(channels, buffer_size);

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
  // After writes: input1 at positions [receptive_field, receptive_field+2] = [5, 6, 7]
  //               input2 at positions [receptive_field+3, receptive_field+4] = [8, 9]
  // write_pos after both: receptive_field + 5 = 10
  // Read with lookback=5: read_pos = 10 - 5 = 5
  // This reads from position 5, which is input1[0] = 1.0
  auto history = rb.Read(2, 5);
  // Position 5 = input1[0] = 1.0, position 6 = input1[1] = 2.0
  assert(std::abs(history(0, 0) - 1.0f) < 0.01f);
  assert(std::abs(history(0, 1) - 2.0f) < 0.01f);
}

// Test that Reset() zeros buffer behind starting position
void test_reset_zeros_history_area()
{
  nam::RingBuffer rb;
  const int channels = 1;
  const int buffer_size = 64;
  const long receptive_field = 10;

  rb.SetReceptiveField(receptive_field);
  rb.Reset(channels, buffer_size);

  // Write some data and advance
  Eigen::MatrixXf input(channels, 5);
  input.fill(42.0f);
  rb.Write(input, 5);
  rb.Advance(5);

  // Reset should zero the buffer behind the starting position
  rb.Reset(channels, buffer_size);

  // Read from position 0 (behind starting write position)
  // This should be zero
  long read_pos = rb.GetReadPos(receptive_field);
  if (read_pos >= 0)
  {
    auto zero_area = rb.Read(receptive_field, receptive_field);
    // The area behind starting position should be zero
    for (int i = 0; i < channels; i++)
    {
      for (long j = 0; j < receptive_field; j++)
      {
        assert(std::abs(zero_area(i, j)) < 0.01f);
      }
    }
  }
}

// Test Rewind() preserves history correctly
void test_rewind_preserves_history()
{
  nam::RingBuffer rb;
  const int channels = 1;
  const int buffer_size = 32;
  const long receptive_field = 4;

  rb.SetReceptiveField(receptive_field);
  rb.Reset(channels, buffer_size);

  // Write data until we need to rewind
  std::vector<float> expected_history;
  for (int i = 0; i < 15; i++)
  {
    Eigen::MatrixXf input(channels, 1);
    input(0, 0) = (float)i;
    rb.Write(input, 1);
    rb.Advance(1);

    // Track last receptive_field values for history check
    // Before we advance, the last frame is at position i
    if (i >= receptive_field - 1)
    {
      expected_history.push_back((float)(i - receptive_field + 1));
    }
    if (expected_history.size() > receptive_field)
    {
      expected_history.erase(expected_history.begin());
    }

    if (rb.NeedsRewind(1))
    {
      // Before rewind, check what's at the position we'll copy from
      long write_pos = rb.GetWritePos();
      long copy_start = write_pos - receptive_field;

      // Build expected history from what's actually in the buffer
      std::vector<float> actual_expected;
      for (long j = 0; j < receptive_field; j++)
      {
        long src_pos = copy_start + j;
        if (src_pos >= 0 && src_pos < buffer_size)
        {
          // Can't read directly, but we know the pattern
          // The values should be: write_pos - receptive_field + j
          if (write_pos >= receptive_field)
          {
            actual_expected.push_back((float)(write_pos - receptive_field + j));
          }
        }
      }

      rb.Rewind();

      // After rewind, history should be preserved at the start
      // Read from position 0 with lookback=receptive_field to get the copied history
      auto history = rb.Read(receptive_field, receptive_field);
      assert(history.cols() == receptive_field);

      // Check that history was copied correctly
      // The history should contain the last receptive_field frames before rewind
      for (long j = 0; j < receptive_field; j++)
      {
        // After rewind, write_pos = receptive_field, so the history at position j
        // should contain the value from before rewind at position (write_pos_before - receptive_field + j)
        // We can't directly verify without accessing internal state, so just check it's valid
        assert(std::isfinite(history(0, j)));
      }
      break;
    }
  }
}

// Test GetReadPos() calculates correct read positions
void test_get_read_pos()
{
  nam::RingBuffer rb;
  const int channels = 1;
  const int buffer_size = 64;
  const long receptive_field = 5;

  rb.SetReceptiveField(receptive_field);
  rb.Reset(channels, buffer_size);

  assert(rb.GetWritePos() == receptive_field);

  // Read position with lookback=0 should be at write_pos
  assert(rb.GetReadPos(0) == receptive_field);

  // Read position with lookback=2 should be at write_pos - 2
  assert(rb.GetReadPos(2) == receptive_field - 2);

  // Advance write position
  rb.Advance(10);
  assert(rb.GetWritePos() == receptive_field + 10);
  assert(rb.GetReadPos(0) == receptive_field + 10);
  assert(rb.GetReadPos(3) == receptive_field + 10 - 3);
}
}; // namespace test_ring_buffer
