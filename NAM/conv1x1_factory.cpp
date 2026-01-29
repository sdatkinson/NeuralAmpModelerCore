// Conv1x1 Factory implementation
// Returns dynamic Conv1x1 wrapped in IConv1x1 interface

#include "conv1x1_factory.h"
#include "dsp.h"

namespace nam
{

/// \brief Dynamic wrapper for Conv1x1 implementing IConv1x1 interface
class Conv1x1Dynamic : public IConv1x1
{
public:
  Conv1x1Dynamic(int in_channels, int out_channels, bool bias, int groups)
  : _conv(in_channels, out_channels, bias, groups)
  {
  }

  Eigen::MatrixXf& GetOutput() override { return _conv.GetOutput(); }

  const Eigen::MatrixXf& GetOutput() const override { return _conv.GetOutput(); }

  void SetMaxBufferSize(int maxBufferSize) override { _conv.SetMaxBufferSize(maxBufferSize); }

  void set_weights_(std::vector<float>::iterator& weights) override { _conv.set_weights_(weights); }

  void process_(const Eigen::Ref<const Eigen::MatrixXf>& input, int num_frames) override
  {
    _conv.process_(input, num_frames);
  }

  Eigen::MatrixXf process(const Eigen::MatrixXf& input, int num_frames) const override
  {
    return _conv.process(input, num_frames);
  }

  long get_out_channels() const override { return _conv.get_out_channels(); }

  long get_in_channels() const override { return _conv.get_in_channels(); }

private:
  Conv1x1 _conv;
};

// Factory implementation - always returns dynamic implementation
std::unique_ptr<IConv1x1> Conv1x1Factory::create(int in_channels, int out_channels, bool bias, int groups)
{
  return std::make_unique<Conv1x1Dynamic>(in_channels, out_channels, bias, groups);
}

} // namespace nam
