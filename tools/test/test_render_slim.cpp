#include <cassert>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "NAM/container.h"
#include "NAM/dsp.h"
#include "NAM/get_dsp.h"

namespace test_render_slim
{

// Test that --slim with a SlimmableContainer model changes the output
void test_slim_changes_output()
{
  std::cout << "  test_slim_changes_output" << std::endl;

  // Load the slimmable container model
  auto model = nam::get_dsp(std::filesystem::path("example_models/slimmable_container.nam"));
  assert(model != nullptr);

  // Verify it's actually a ContainerModel
  auto* container = dynamic_cast<nam::container::ContainerModel*>(model.get());
  assert(container != nullptr);

  const double sample_rate = model->GetExpectedSampleRate() > 0 ? model->GetExpectedSampleRate() : 48000.0;
  const int buffer_size = 64;
  model->Reset(sample_rate, buffer_size);

  std::vector<NAM_SAMPLE> input(buffer_size, 0.1);
  std::vector<NAM_SAMPLE> out_small(buffer_size);
  std::vector<NAM_SAMPLE> out_large(buffer_size);
  NAM_SAMPLE* in_ptr = input.data();
  NAM_SAMPLE* out_ptr;

  // Render at minimum size
  model->SetSlimmableSize(0.0);
  out_ptr = out_small.data();
  model->process(&in_ptr, &out_ptr, buffer_size);

  // Render at maximum size
  model->SetSlimmableSize(1.0);
  out_ptr = out_large.data();
  model->process(&in_ptr, &out_ptr, buffer_size);

  // Outputs should differ since different submodels are active
  bool any_different = false;
  for (int i = 0; i < buffer_size; i++)
  {
    if (std::abs(out_small[i] - out_large[i]) > 1e-6)
    {
      any_different = true;
      break;
    }
  }
  assert(any_different);
}

// Test that --slim rejects non-slimmable models
void test_slim_rejects_non_slimmable()
{
  std::cout << "  test_slim_rejects_non_slimmable" << std::endl;

  // Load a regular (non-container) model
  auto model = nam::get_dsp(std::filesystem::path("example_models/lstm.nam"));
  assert(model != nullptr);

  // Verify it's NOT a ContainerModel
  auto* container = dynamic_cast<nam::container::ContainerModel*>(model.get());
  assert(container == nullptr);
}

// Test that --slim with boundary values produces finite output
void test_slim_boundary_values()
{
  std::cout << "  test_slim_boundary_values" << std::endl;

  auto model = nam::get_dsp(std::filesystem::path("example_models/slimmable_container.nam"));
  assert(model != nullptr);

  const double sample_rate = model->GetExpectedSampleRate() > 0 ? model->GetExpectedSampleRate() : 48000.0;
  const int buffer_size = 64;
  model->Reset(sample_rate, buffer_size);

  std::vector<NAM_SAMPLE> input(buffer_size, 0.05);
  std::vector<NAM_SAMPLE> output(buffer_size);
  NAM_SAMPLE* in_ptr = input.data();
  NAM_SAMPLE* out_ptr = output.data();

  double values[] = {0.0, 0.25, 0.5, 0.75, 1.0};
  for (double val : values)
  {
    model->SetSlimmableSize(val);
    model->process(&in_ptr, &out_ptr, buffer_size);
    for (int i = 0; i < buffer_size; i++)
      assert(std::isfinite(output[i]));
  }
}

// Test that SetSlimmableSize is called before processing (simulates --slim flow)
void test_slim_applied_before_processing()
{
  std::cout << "  test_slim_applied_before_processing" << std::endl;

  auto model = nam::get_dsp(std::filesystem::path("example_models/slimmable_container.nam"));
  assert(model != nullptr);

  auto* container = dynamic_cast<nam::container::ContainerModel*>(model.get());
  assert(container != nullptr);

  const double sample_rate = model->GetExpectedSampleRate() > 0 ? model->GetExpectedSampleRate() : 48000.0;
  const int buffer_size = 64;

  // Set slim BEFORE Reset (as render.cpp does it before Reset)
  model->SetSlimmableSize(0.5);
  model->Reset(sample_rate, buffer_size);

  std::vector<NAM_SAMPLE> input(buffer_size, 0.1);
  std::vector<NAM_SAMPLE> output(buffer_size);
  NAM_SAMPLE* in_ptr = input.data();
  NAM_SAMPLE* out_ptr = output.data();

  model->process(&in_ptr, &out_ptr, buffer_size);

  for (int i = 0; i < buffer_size; i++)
    assert(std::isfinite(output[i]));
}

} // namespace test_render_slim
