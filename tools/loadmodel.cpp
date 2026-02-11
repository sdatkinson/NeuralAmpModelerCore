#include <stdlib.h>
#include <filesystem>
#include "NAM/dsp.h"
#include "NAM/get_dsp.h"
#include "NAM/get_dsp_namb.h"

int main(int argc, char* argv[])
{
  if (argc > 1)
  {
    char* modelPath = argv[1];

    fprintf(stderr, "Loading model [%s]\n", modelPath);

    std::filesystem::path path(modelPath);
    std::unique_ptr<nam::DSP> model;
    if (path.extension() == ".namb")
      model = nam::get_dsp_namb(path);
    else
      model = nam::get_dsp(path);

    if (model != nullptr)
    {
      fprintf(stderr, "Model loaded successfully\n");
    }
    else
    {
      fprintf(stderr, "Failed to load model\n");

      exit(1);
    }
  }
  else
  {
    fprintf(stderr, "Usage: loadmodel <model_path>\n");
  }

  exit(0);
}
