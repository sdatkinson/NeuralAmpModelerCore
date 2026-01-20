#pragma once

#include <Eigen/Dense>
#include <cassert>
#include <vector>

#include "dsp.h"

namespace nam
{
// Feature-wise Linear Modulation (FiLM)
//
// Given an `input` (input_dim x num_frames) and a `condition`
// (condition_dim x num_frames), compute:
//   scale, shift = Conv1x1(condition) split across channels
//   output = input * scale + shift  (elementwise)
class FiLM
{
public:
  FiLM(const int condition_dim, const int input_dim, const bool shift)
  : _cond_to_scale_shift(condition_dim, (shift ? 2 : 1) * input_dim, /*bias=*/true)
  , _do_shift(shift)
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
    _output.resize(get_input_dim(), maxBufferSize);
  }

  void set_weights_(std::vector<float>::iterator& weights) { _cond_to_scale_shift.set_weights_(weights); }

  long get_condition_dim() const { return _cond_to_scale_shift.get_in_channels(); }
  long get_input_dim() const
  {
    return _do_shift ? (_cond_to_scale_shift.get_out_channels() / 2) : _cond_to_scale_shift.get_out_channels();
  }

  // :param input: (input_dim x num_frames)
  // :param condition: (condition_dim x num_frames)
  // Writes (input_dim x num_frames) into internal output buffer; access via GetOutput().
  void Process(const Eigen::MatrixXf& input, const Eigen::MatrixXf& condition, const int num_frames)
  {
    assert(get_input_dim() == input.rows());
    assert(get_condition_dim() == condition.rows());
    assert(num_frames <= input.cols());
    assert(num_frames <= condition.cols());
    assert(num_frames <= _output.cols());

    _cond_to_scale_shift.process_(condition, num_frames);
    const auto& scale_shift = _cond_to_scale_shift.GetOutput();

    const auto scale = scale_shift.topRows(get_input_dim()).leftCols(num_frames);
    if (_do_shift)
    {
      // scale = top input_dim, shift = bottom input_dim
      const auto shift = scale_shift.bottomRows(get_input_dim()).leftCols(num_frames);
      _output.leftCols(num_frames).array() = input.leftCols(num_frames).array() * scale.array() + shift.array();
    }
    else
    {
      _output.leftCols(num_frames).array() = input.leftCols(num_frames).array() * scale.array();
    }
  }

private:
  Conv1x1 _cond_to_scale_shift; // condition_dim -> (shift ? 2 : 1) * input_dim
  Eigen::MatrixXf _output; // input_dim x maxBufferSize
  bool _do_shift;
};
} // namespace nam
