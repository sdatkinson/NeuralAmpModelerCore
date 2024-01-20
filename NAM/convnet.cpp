#include <algorithm> // std::max_element
#include <algorithm>
#include <cmath> // pow, tanh, expf
#include <filesystem>
#include <fstream>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "dsp.h"
#include "convnet.h"

nam::convnet::BatchNorm::BatchNorm(const int dim, weightsIterator& weights)
{
  // Extract from param buffer
  Eigen::VectorXf running_mean(dim);
  Eigen::VectorXf running_var(dim);
  Eigen::VectorXf weight(dim);
  Eigen::VectorXf bias(dim);
  for (int i = 0; i < dim; i++)
    running_mean(i) = *(weights++);
  for (int i = 0; i < dim; i++)
    running_var(i) = *(weights++);
  for (int i = 0; i < dim; i++)
    weight(i) = *(weights++);
  for (int i = 0; i < dim; i++)
    bias(i) = *(weights++);
  float eps = *(weights++);

  // Convert to scale & loc
  mScale.resize(dim);
  mLoc.resize(dim);
  for (int i = 0; i < dim; i++)
    mScale(i) = weight(i) / sqrt(eps + running_var(i));
  mLoc = bias - mScale.cwiseProduct(running_mean);
}

void nam::convnet::BatchNorm::Process(Eigen::Ref<Eigen::MatrixXf> x, const long i_start, const long i_end) const
{
  // todo using colwise?
  // #speed but conv probably dominates
  for (auto i = i_start; i < i_end; i++)
  {
    x.col(i) = x.col(i).cwiseProduct(mScale);
    x.col(i) += mLoc;
  }
}

void nam::convnet::ConvNetBlock::SetWeights(const int inChannels, const int outChannels, const int dilation,
                                              const bool doBatchNorm, const std::string activation,
                                              weightsIterator& weights)
{
  mDoBatchNorm = doBatchNorm;
  // HACK 2 kernel
  conv.SetSizeAndWeights(inChannels, outChannels, 2, dilation, !doBatchNorm, weights);
  if (mDoBatchNorm)
    mBatchnorm = BatchNorm(outChannels, weights);
  mActivation = activations::Activation::GetActivation(activation);
}

void nam::convnet::ConvNetBlock::Process(const Eigen::Ref<const Eigen::MatrixXf> input, Eigen::Ref<Eigen::MatrixXf> output, const long i_start,
                                          const long i_end) const
{
  const long ncols = i_end - i_start;
  conv.Process(input, output, i_start, ncols, i_start);
  if (mDoBatchNorm)
    mBatchnorm.Process(output, i_start, i_end);

  mActivation->Apply(output.middleCols(i_start, ncols));
}

long nam::convnet::ConvNetBlock::GetOutChannels() const
{
  return conv.GetOutChannels();
}

nam::convnet::Head::Head(const int channels, weightsIterator& weights)
{
  mWeight.resize(channels);
  for (int i = 0; i < channels; i++)
    mWeight[i] = *(weights++);
  mBias = *(weights++);
}

void nam::convnet::Head::Process(const Eigen::Ref<const Eigen::MatrixXf> input, Eigen::VectorXf& output, const long i_start,
                                   const long i_end) const
{
  const long length = i_end - i_start;
  output.resize(length);
  for (long i = 0, j = i_start; i < length; i++, j++)
    output(i) = mBias + input.col(j).dot(mWeight);
}

nam::convnet::ConvNet::ConvNet(const int channels, const std::vector<int>& dilations, const bool batchnorm,
                               const std::string activation, std::vector<float>& weights,
                               const double expectedSampleRate)
: Buffer(*std::max_element(dilations.begin(), dilations.end()), expectedSampleRate)
{
  VerifyWeights(channels, dilations, batchnorm, weights.size());
  mBlocks.resize(dilations.size());
  weightsIterator it = weights.begin();
  for (size_t i = 0; i < dilations.size(); i++)
    mBlocks[i].SetWeights(i == 0 ? 1 : channels, channels, dilations[i], batchnorm, activation, it);
  mBlockVals.resize(mBlocks.size() + 1);
  for (auto& matrix : mBlockVals)
    matrix.setZero();
  std::fill(mInputBuffer.begin(), mInputBuffer.end(), 0.0f);
  mHead = Head(channels, it);
  if (it != weights.end())
    throw std::runtime_error("Didn't touch all the weights when initializing ConvNet");

  mPrewarmSamples = 1;
  for (size_t i = 0; i < dilations.size(); i++)
    mPrewarmSamples += dilations[i];
}


void nam::convnet::ConvNet::Process(float* input, float* output, const int numFrames)

{
  UpdateBuffers(input, numFrames);
  // Main computation!
  const long i_start = mInputBufferOffset;
  const long i_end = i_start + numFrames;
  // TODO one unnecessary copy :/ #speed
  for (auto i = i_start; i < i_end; i++)
    mBlockVals[0](0, i) = mInputBuffer[i];
  for (size_t i = 0; i < mBlocks.size(); i++)
    mBlocks[i].Process(mBlockVals[i], mBlockVals[i + 1], i_start, i_end);
  // TODO clean up this allocation
  mHead.Process(mBlockVals[mBlocks.size()], mHeadOutput, i_start, i_end);
  // Copy to required output array (TODO tighten this up)
  for (int s = 0; s < numFrames; s++)
    output[s] = mHeadOutput(s);
}

void nam::convnet::ConvNet::VerifyWeights(const int channels, const std::vector<int>& dilations, const bool batchnorm,
                                            const size_t actualWeights)
{
  // TODO
}

void nam::convnet::ConvNet::UpdateBuffers(float* input, const int numFrames)
{
  Buffer::UpdateBuffers(input, numFrames);

  const size_t buffer_size = mInputBuffer.size();

  if (mBlockVals[0].rows() != Eigen::Index(1) || mBlockVals[0].cols() != Eigen::Index(buffer_size))
  {
    mBlockVals[0].resize(1, buffer_size);
    mBlockVals[0].setZero();
  }

  for (size_t i = 1; i < mBlockVals.size(); i++)
  {
    if (mBlockVals[i].rows() == mBlocks[i - 1].GetOutChannels()
        && mBlockVals[i].cols() == Eigen::Index(buffer_size))
      continue; // Already has correct size
    mBlockVals[i].resize(mBlocks[i - 1].GetOutChannels(), buffer_size);
    mBlockVals[i].setZero();
  }
}

void nam::convnet::ConvNet::RewindBuffers()
{
  // Need to rewind the block vals first because Buffer::rewind_buffers()
  // resets the offset index
  // The last _block_vals is the output of the last block and doesn't need to be
  // rewound.
  for (size_t k = 0; k < mBlockVals.size() - 1; k++)
  {
    // We actually don't need to pull back a lot...just as far as the first
    // input sample would grab from dilation
    const long dilation = mBlocks[k].conv.GetDilation();
    for (long i = mReceptiveField - dilation, j = mInputBufferOffset - dilation;
         j < mInputBufferOffset; i++, j++)
      for (long r = 0; r < mBlockVals[k].rows(); r++)
        mBlockVals[k](r, i) = mBlockVals[k](r, j);
  }
  // Now we can do the rest of the rewind
  Buffer::RewindBuffers();
}
