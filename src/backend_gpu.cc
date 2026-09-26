#include "src/backend_gpu.h"

#include <utility>

#include "src/delegate_handle.h"
#include "src/fingerprint.h"
#include "src/tflite_interpreter.h"

#if MIE_ENABLE_GPU
#include "tensorflow/lite/delegates/gpu/delegate.h"
#endif

namespace mie {
namespace internal {

std::unique_ptr<Engine> BuildGpu(const Config& cfg, const ModelSource& model,
                                 std::string* error) {
#if !MIE_ENABLE_GPU
  (void)cfg;
  (void)model;
  if (error != nullptr) {
    *error = "GPU backend not compiled in (configure with -DMIE_ENABLE_GPU=ON)";
  }
  return nullptr;
#else
  TfLiteGpuDelegateOptionsV2 opts = TfLiteGpuDelegateOptionsV2Default();
  // An engine is built once and invoked many times: favour steady-state
  // throughput over one-shot latency.
  opts.inference_preference = TFLITE_GPU_INFERENCE_PREFERENCE_SUSTAINED_SPEED;

  // TfLiteGpuDelegateV2Create copies both strings, so these only have to
  // outlive this function.
  std::string cache_dir;
  std::string model_token;
  if (!cfg.cache_dir.empty()) {
    cache_dir = cfg.cache_dir;
    model_token = CacheToken(cfg, model);
    if (model_token.empty()) {
      if (error != nullptr) {
        *error = "GPU cache requested but the model token could not be derived";
      }
      return nullptr;
    }
    // Serialization writes the compiled program to disk on the first run and
    // replays it on every later run, skipping the costly shader compilation.
    opts.experimental_flags |= TFLITE_GPU_EXPERIMENTAL_FLAGS_ENABLE_SERIALIZATION;
    opts.serialization_dir = cache_dir.c_str();
    opts.model_token = model_token.c_str();
  }

  DelegateHandle handle;
  handle.Adopt(TfLiteGpuDelegateV2Create(&opts), &TfLiteGpuDelegateV2Delete);
  if (handle.empty()) {
    if (error != nullptr) *error = "TfLiteGpuDelegateV2Create failed";
    return nullptr;
  }

  return BuildTfliteEngine(cfg, model, std::move(handle), error);
#endif
}

}  // namespace internal
}  // namespace mie
