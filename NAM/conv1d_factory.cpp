// Conv1D Factory implementation
// Returns dynamic Conv1D wrapped in IConv1D interface

#include "conv1d_factory.h"
#include "conv1d.h"

namespace nam
{

/// \brief Dynamic wrapper for Conv1D implementing IConv1D interface
///
/// This class wraps the existing Conv1D implementation to provide the IConv1D
/// interface for configurations that don't have specialized template instantiations.
class Conv1DDynamicWrapper : public IConv1D
{
public:
  Conv1DDynamicWrapper(int in_channels, int out_channels, int kernel_size, int dilation, bool bias, int groups)
  {
    _conv.set_size_(in_channels, out_channels, kernel_size, bias, dilation, groups);
  }

  Eigen::MatrixXf& GetOutput() override { return _conv.GetOutput(); }

  const Eigen::MatrixXf& GetOutput() const override { return _conv.GetOutput(); }

  void SetMaxBufferSize(int maxBufferSize) override { _conv.SetMaxBufferSize(maxBufferSize); }

  void set_weights_(std::vector<float>::iterator& weights) override { _conv.set_weights_(weights); }

  void Process(const Eigen::MatrixXf& input, int num_frames) override { _conv.Process(input, num_frames); }

  long get_out_channels() const override { return _conv.get_out_channels(); }

  long get_in_channels() const override { return _conv.get_in_channels(); }

  long get_kernel_size() const override { return _conv.get_kernel_size(); }

  int get_dilation() const override { return _conv.get_dilation(); }

  bool has_bias() const override { return _conv.has_bias(); }

private:
  Conv1D _conv;
};

// Factory implementation - always returns dynamic implementation
std::unique_ptr<IConv1D> Conv1DFactory::create(int in_channels, int out_channels, int kernel_size, int dilation,
                                               bool bias, int groups)
{
  return std::make_unique<Conv1DDynamicWrapper>(in_channels, out_channels, kernel_size, dilation, bias, groups);
}

} // namespace nam
