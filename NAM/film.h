#pragma once

#include <Eigen/Dense>
#include <cassert>
#include <vector>

#include "dsp.h"

namespace nam
{
// Feature-wise Linear Modulation (FiLM)
//
// Given an `input` (out_channels x num_frames) and a `condition`
// (in_channels x num_frames), compute:
//   scale, shift = Conv1x1(condition) split across channels
//   output = input * scale + shift  (elementwise)
class FiLM
{
public:
  FiLM(const int in_channels, const int out_channels)
  : _cond_to_scale_shift(in_channels, 2 * out_channels, /*bias=*/true)
  {
  }

  // Get the entire internal output buffer. This is intended for internal wiring
  // between layers; callers should treat the buffer as pre-allocated storage
  // and only consider the first `num_frames` columns valid for a given
  // processing call. Slice with .leftCols(num_frames) as needed.
  Eigen::MatrixXf& GetOutput() { return _output; }
  const Eigen::MatrixXf& GetOutput() const { return _output; }

  void SetMaxBufferSize(const int maxBufferSize)
  {
    _cond_to_scale_shift.SetMaxBufferSize(maxBufferSize);
    _output.resize(get_out_channels(), maxBufferSize);
  }

  void set_weights_(std::vector<float>::iterator& weights) { _cond_to_scale_shift.set_weights_(weights); }

  long get_in_channels() const { return _cond_to_scale_shift.get_in_channels(); }
  long get_out_channels() const { return _cond_to_scale_shift.get_out_channels() / 2; }

  // :param input: (out_channels x num_frames)
  // :param condition: (in_channels x num_frames)
  // Writes (out_channels x num_frames) into internal output buffer; access via GetOutput().
  void Process(const Eigen::MatrixXf& input, const Eigen::MatrixXf& condition, const int num_frames)
  {
    assert(get_out_channels() == input.rows());
    assert(get_in_channels() == condition.rows());
    assert(num_frames <= input.cols());
    assert(num_frames <= condition.cols());
    assert(num_frames <= _output.cols());

    _cond_to_scale_shift.process_(condition, num_frames);
    const auto& scale_shift = _cond_to_scale_shift.GetOutput();

    // scale = top out_channels, shift = bottom out_channels
    const auto scale = scale_shift.topRows(get_out_channels()).leftCols(num_frames);
    const auto shift = scale_shift.bottomRows(get_out_channels()).leftCols(num_frames);

    _output.leftCols(num_frames).array() = input.leftCols(num_frames).array() * scale.array() + shift.array();
  }

private:
  Conv1x1 _cond_to_scale_shift; // in_channels -> 2*out_channels
  Eigen::MatrixXf _output; // out_channels x maxBufferSize
};
} // namespace nam
