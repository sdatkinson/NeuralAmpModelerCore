/**
 * Neural Amp Modeler (NAM) WebAssembly Audio Processing Module
 * This module handles real-time audio processing using WebAssembly and Web Audio API
 * for neural network-based audio modeling.
 * See previous work by Kutalia: https://github.com/Kutalia/NeuralAmpModelerCore_WASM
 * @author: @woodybury
 * @date: 2025-06-02
 */

#include <emscripten/webaudio.h>
#include <algorithm>
#include <stdexcept>
#include <cmath>
#include <utility>
#include <cassert>
#include <string>

#include <NAM/activations.h>
#include <NAM/dsp.h>

// Threshold for level smoothing
#define SMOOTH_EPSILON .0001f

// Global state variables for audio processing
float inputLevel = 0; // Current input gain level
float outputLevel = 0; // Current output gain level
float input_level = 0; // Target input gain in dB
float output_level = 0; // Target output gain in dB
bool loading = false; // Flag to indicate model loading state

unsigned int sampleRate = 48000; // Default sample rate
float dcBlockerCoeff = 0.995f; // DC blocker coefficient (will be updated based on sample rate)

// Current neural network model instance
std::unique_ptr<nam::DSP> currentModel = nullptr;
// DC blocking filter state
float prevDCInput = 0;
float prevDCOutput = 0;

/**
 * Updates the DC blocker coefficient based on the current sample rate
 * Uses a cutoff frequency of 10Hz for the high-pass filter
 */
void updateDCBlockerCoeff() {
    const float cutoffFreq = 10.0f; // 10Hz cutoff frequency
    const float pi = 3.14159265358979323846f;
    float omega = 2.0f * pi * cutoffFreq / sampleRate;
    dcBlockerCoeff = 1.0f - omega;
}

/**
 * Main audio processing function that handles:
 * 1. Input gain adjustment
 * 2. Neural network model processing
 * 3. Output gain adjustment
 * 4. DC blocking
 */
void process(float* audio_in, float* audio_out, int n_samples)
{
  float level;

  // convert input level from db
  float desiredInputLevel = powf(10, input_level * 0.05f);

  if (fabs(desiredInputLevel - inputLevel) > SMOOTH_EPSILON)
  {
    level = inputLevel;
    for (int i = 0; i < n_samples; i++)
    {
      // do very basic smoothing
      level = (.99f * level) + (.01f * desiredInputLevel);

      audio_out[i] = audio_in[i] * level;
    }

    inputLevel = level;
  }
  else
  {
    level = inputLevel = desiredInputLevel;

    for (int i = 0; i < n_samples; i++)
    {
      audio_out[i] = audio_in[i] * level;
    }
  }

  float modelLoudnessAdjustmentDB = 0;

  if (currentModel != nullptr)
  {
    currentModel->process(audio_out, audio_out, n_samples);

    if (currentModel->HasLoudness())
    {
      // Normalize model to -18dB
      modelLoudnessAdjustmentDB = -18 - currentModel->GetLoudness();
    }
  }

  // Convert output level from db
  float desiredOutputLevel = powf(10, (output_level + modelLoudnessAdjustmentDB) * 0.05f);

  if (fabs(desiredOutputLevel - outputLevel) > SMOOTH_EPSILON)
  {
    level = outputLevel;

    for (int i = 0; i < n_samples; i++)
    {
      // do very basic smoothing
      level = (.99f * level) + (.01f * desiredOutputLevel);

      audio_out[i] = audio_out[i] * outputLevel;
    }

    outputLevel = level;
  }
  else
  {
    level = outputLevel = desiredOutputLevel;

    for (int i = 0; i < n_samples; i++)
    {
      audio_out[i] = audio_out[i] * level;
    }
  }

  for (int i = 0; i < n_samples; i++)
  {
    float dcInput = audio_out[i];

    // dc blocker with sample rate dependent coefficient
    audio_out[i] = audio_out[i] - prevDCInput + dcBlockerCoeff * prevDCOutput;

    prevDCInput = dcInput;
    prevDCOutput = audio_out[i];
  }
}

uint8_t audioThreadStack[16384 * 32];

/**
 * Web Audio API worklet processor callback
 * Handles audio processing in the audio worklet thread
 */
EM_BOOL NamProcessor(int numInputs, const AudioSampleFrame* inputs, int numOutputs, AudioSampleFrame* outputs,
                     int numParams, const AudioParamFrame* params, void* userData)
{
  if (loading == false)
  {
    process(inputs[0].data, outputs[0].data, 128);

    return EM_TRUE; // Keep the graph output going
  }
  else
  {
    return EM_FALSE;
  }
}

/**
 * Click handler to resume audio context
 * Required by browsers to start audio processing
 */
EM_BOOL OnElementClick(int eventType, const EmscriptenMouseEvent* mouseEvent, void* userData)
{
  EMSCRIPTEN_WEBAUDIO_T audioContext = (EMSCRIPTEN_WEBAUDIO_T)userData;
  if (emscripten_audio_context_state(audioContext) != AUDIO_CONTEXT_STATE_RUNNING)
  {
    emscripten_resume_audio_context_sync(audioContext);
  }
  return EM_FALSE;
}

/**
 * Callback when audio worklet processor is created
 * Sets up the audio processing node with appropriate configuration
 */
void AudioWorkletProcessorCreated(EMSCRIPTEN_WEBAUDIO_T audioContext, EM_BOOL success, void* userData)
{
  if (!success)
    return; // Check browser console in a debug build for detailed errors

  int outputChannelCounts[1] = {1};
  EmscriptenAudioWorkletNodeCreateOptions options = {
    .numberOfInputs = 1, .numberOfOutputs = 1, .outputChannelCounts = outputChannelCounts};

  // Create node
  EMSCRIPTEN_AUDIO_WORKLET_NODE_T wasmAudioWorklet =
    emscripten_create_wasm_audio_worklet_node(audioContext, "nam-worklet", &options, &NamProcessor, 0);

  EM_ASM(
    {
      // create global callback in your code
      // the first argument is audioContext, the second one - worklet node
      if (window.wasmAudioWorkletCreated)
      {
        window.wasmAudioWorkletCreated(emscriptenGetAudioObject($0), emscriptenGetAudioObject($1));
      }
    },
    wasmAudioWorklet, audioContext);

  // Resume context on mouse click on a specific element created in your html file
  emscripten_set_click_callback("#audio-worklet-resumer", (void*)audioContext, 0, OnElementClick);
}

/**
 * Callback when audio thread is initialized
 * Creates the audio worklet processor
 */
void AudioThreadInitialized(EMSCRIPTEN_WEBAUDIO_T audioContext, EM_BOOL success, void* userData)
{
  if (!success)
    return; // Check browser console in a debug build for detailed errors
  WebAudioWorkletProcessorCreateOptions opts = {
    .name = "nam-worklet",
  };
  emscripten_create_wasm_audio_worklet_processor_async(audioContext, &opts, &AudioWorkletProcessorCreated, 0);
}

/**
 * Queries the browser's audio context for the current sample rate
 * @return The sample rate of the audio context
 */
unsigned query_sample_rate_of_audiocontexts()
{
  return EM_ASM_INT({
    var AudioContext = window.AudioContext || window.webkitAudioContext;
    var ctx = new AudioContext();
    var sr = ctx.sampleRate;
    ctx.close();
    return sr;
  });
}

extern "C" {
/**
 * Sets up the DSP model from a JSON configuration
 * Initializes the audio context and worklet if this is the first model
 * @param jsonStr JSON string containing the model configuration
 */
void setDsp(const char* jsonStr)
{
  std::unique_ptr<nam::DSP> tmp = nullptr;

  loading = true;
  tmp = nam::get_dsp(jsonStr);

  if (currentModel == nullptr)
  {
    // Turn on fast tanh approximation
    nam::activations::Activation::enable_fast_tanh();
    currentModel = std::move(tmp);

    sampleRate = query_sample_rate_of_audiocontexts();
    updateDCBlockerCoeff(); // Update DC blocker coefficient for the current sample rate
    
    EmscriptenWebAudioCreateAttributes attrs = {.latencyHint = "interactive", .sampleRate = sampleRate};

    EMSCRIPTEN_WEBAUDIO_T context = emscripten_create_audio_context(&attrs);

    emscripten_start_wasm_audio_worklet_thread_async(
      context, audioThreadStack, sizeof(audioThreadStack), &AudioThreadInitialized, 0);
  }
  else
  {
    currentModel.reset();
    currentModel = std::move(tmp);
  }

  loading = false;
}

// int main() {
// 	return 1;
// }
}