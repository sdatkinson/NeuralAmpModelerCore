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

nam::convnet::BatchNorm::BatchNorm(const int dim, weights_it& weights)
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
  this->mScale.resize(dim);
  this->mLoc.resize(dim);
  for (int i = 0; i < dim; i++)
    this->mScale(i) = weight(i) / sqrt(eps + running_var(i));
  this->mLoc = bias - this->mScale.cwiseProduct(running_mean);
}

void nam::convnet::BatchNorm::Process(Eigen::Ref<Eigen::MatrixXf> x, const long i_start, const long i_end) const
{
  // todo using colwise?
  // #speed but conv probably dominates
  for (auto i = i_start; i < i_end; i++)
  {
    x.col(i) = x.col(i).cwiseProduct(this->mScale);
    x.col(i) += this->mLoc;
  }
}

void nam::convnet::ConvNetBlock::SetWeights(const int inChannels, const int outChannels, const int dilation,
                                              const bool doBatchNorm, const std::string activation,
                                              weights_it& weights)
{
  this->mDoBatchNorm = doBatchNorm;
  // HACK 2 kernel
  this->conv.SetSizeAndWeights(inChannels, outChannels, 2, dilation, !doBatchNorm, weights);
  if (this->mDoBatchNorm)
    this->mBatchnorm = BatchNorm(outChannels, weights);
  this->activation = activations::Activation::GetActivation(activation);
}

void nam::convnet::ConvNetBlock::Process(const Eigen::Ref<const Eigen::MatrixXf> input, Eigen::Ref<Eigen::MatrixXf> output, const long i_start,
                                          const long i_end) const
{
  const long ncols = i_end - i_start;
  this->conv.Process(input, output, i_start, ncols, i_start);
  if (this->mDoBatchNorm)
    this->mBatchnorm.Process(output, i_start, i_end);

  this->activation->Apply(output.middleCols(i_start, ncols));
}

long nam::convnet::ConvNetBlock::GetOutChannels() const
{
  return this->conv.GetOutChannels();
}

nam::convnet::Head::Head(const int channels, weights_it& weights)
{
  this->mWeight.resize(channels);
  for (int i = 0; i < channels; i++)
    this->mWeight[i] = *(weights++);
  this->mBias = *(weights++);
}

void nam::convnet::Head::Process(const Eigen::Ref<const Eigen::MatrixXf> input, Eigen::VectorXf& output, const long i_start,
                                   const long i_end) const
{
  const long length = i_end - i_start;
  output.resize(length);
  for (long i = 0, j = i_start; i < length; i++, j++)
    output(i) = this->mBias + input.col(j).dot(this->mWeight);
}

nam::convnet::ConvNet::ConvNet(const int channels, const std::vector<int>& dilations, const bool batchnorm,
                               const std::string activation, std::vector<float>& weights,
                               const double expectedSampleRate)
: Buffer(*std::max_element(dilations.begin(), dilations.end()), expectedSampleRate)
{
  this->VerifyWeights(channels, dilations, batchnorm, weights.size());
  this->mBlocks.resize(dilations.size());
  weights_it it = weights.begin();
  for (size_t i = 0; i < dilations.size(); i++)
    this->mBlocks[i].SetWeights(i == 0 ? 1 : channels, channels, dilations[i], batchnorm, activation, it);
  this->mBlockVals.resize(this->mBlocks.size() + 1);
  for (auto& matrix : this->mBlockVals)
    matrix.setZero();
  std::fill(this->mInputBuffer.begin(), this->mInputBuffer.end(), 0.0f);
  this->mHead = Head(channels, it);
  if (it != weights.end())
    throw std::runtime_error("Didn't touch all the weights when initializing ConvNet");

  mPrewarmSamples = 1;
  for (size_t i = 0; i < dilations.size(); i++)
    mPrewarmSamples += dilations[i];
}


void nam::convnet::ConvNet::Process(float* input, float* output, const int numFrames)

{
  this->UpdateBuffers(input, numFrames);
  // Main computation!
  const long i_start = this->mInputBufferOffset;
  const long i_end = i_start + numFrames;
  // TODO one unnecessary copy :/ #speed
  for (auto i = i_start; i < i_end; i++)
    this->mBlockVals[0](0, i) = this->mInputBuffer[i];
  for (size_t i = 0; i < this->mBlocks.size(); i++)
    this->mBlocks[i].Process(this->mBlockVals[i], this->mBlockVals[i + 1], i_start, i_end);
  // TODO clean up this allocation
  this->mHead.Process(this->mBlockVals[this->mBlocks.size()], this->mHeadOutput, i_start, i_end);
  // Copy to required output array (TODO tighten this up)
  for (int s = 0; s < numFrames; s++)
    output[s] = this->mHeadOutput(s);
}

void nam::convnet::ConvNet::VerifyWeights(const int channels, const std::vector<int>& dilations, const bool batchnorm,
                                            const size_t actualWeights)
{
  // TODO
}

void nam::convnet::ConvNet::UpdateBuffers(float* input, const int numFrames)
{
  this->Buffer::UpdateBuffers(input, numFrames);

  const size_t buffer_size = this->mInputBuffer.size();

  if (this->mBlockVals[0].rows() != Eigen::Index(1) || this->mBlockVals[0].cols() != Eigen::Index(buffer_size))
  {
    this->mBlockVals[0].resize(1, buffer_size);
    this->mBlockVals[0].setZero();
  }

  for (size_t i = 1; i < this->mBlockVals.size(); i++)
  {
    if (this->mBlockVals[i].rows() == this->mBlocks[i - 1].GetOutChannels()
        && this->mBlockVals[i].cols() == Eigen::Index(buffer_size))
      continue; // Already has correct size
    this->mBlockVals[i].resize(this->mBlocks[i - 1].GetOutChannels(), buffer_size);
    this->mBlockVals[i].setZero();
  }
}

void nam::convnet::ConvNet::RewindBuffers()
{
  // Need to rewind the block vals first because Buffer::rewind_buffers()
  // resets the offset index
  // The last _block_vals is the output of the last block and doesn't need to be
  // rewound.
  for (size_t k = 0; k < this->mBlockVals.size() - 1; k++)
  {
    // We actually don't need to pull back a lot...just as far as the first
    // input sample would grab from dilation
    const long dilation = this->mBlocks[k].conv.GetDilation();
    for (long i = this->mReceptiveField - dilation, j = this->mInputBufferOffset - dilation;
         j < this->mInputBufferOffset; i++, j++)
      for (long r = 0; r < this->mBlockVals[k].rows(); r++)
        this->mBlockVals[k](r, i) = this->mBlockVals[k](r, j);
  }
  // Now we can do the rest of the rewind
  this->Buffer::RewindBuffers();
}
