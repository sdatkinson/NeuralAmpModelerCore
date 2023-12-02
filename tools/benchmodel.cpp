#include "malloc.h"
#include <iostream>
#include <chrono>

#include "NAM/dsp.h"

using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;

#define AUDIO_BUFFER_SIZE 64

double buffer[AUDIO_BUFFER_SIZE];

int main(int argc, char* argv[])
{
  if (argc > 1)
  {
    const char* modelPath = argv[1];

    std::cout << "Loading model " << modelPath << "\n";

    // Turn on fast tanh approximation
    nam::activations::Activation::enable_fast_tanh();

    std::unique_ptr<nam::DSP> model;

    model.reset();
    model = std::move(nam::get_dsp(modelPath));

    if (model == nullptr)
    {
      std::cerr << "Failed to load model\n";

      exit(1);
    }

    auto t1 = high_resolution_clock::now();

    size_t bufferSize = 64;
    size_t numBuffers = (48000 / 64) * 2;

    std::cout << "Running benchmark\n";

    for (size_t i = 0; i < numBuffers; i++)
    {
      model->process(buffer, buffer, AUDIO_BUFFER_SIZE);
      model->finalize_(AUDIO_BUFFER_SIZE);
    }

    std::cout << "Finished\n";

    auto t2 = high_resolution_clock::now();

    /* Getting number of milliseconds as an integer. */
    auto ms_int = duration_cast<milliseconds>(t2 - t1);

    /* Getting number of milliseconds as a double. */
    duration<double, std::milli> ms_double = t2 - t1;

    std::cout << ms_int.count() << "ms\n";
    std::cout << ms_double.count() << "ms\n";
  }
  else
  {
    std::cerr << "Usage: loadmodel <model_path>\n";
  }

  exit(0);
}
