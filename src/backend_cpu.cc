#include "src/backend_cpu.h"

#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"

#include "src/delegate_handle.h"
#include "src/tflite_interpreter.h"

namespace mie {
namespace internal {
namespace {

// True when cfg.options carries key=value. Used for the CPU backend's own
// knob(s); vendor delegates receive the same options unfiltered through their
// own channel, so nothing here can collide with them.
bool HasOption(const Config& cfg, const char* key, const char* value) {
  for (const auto& kv : cfg.options) {
    if (kv.first == key) return kv.second == value;
  }
  return false;
}

}  // namespace

std::unique_ptr<Engine> BuildCpu(const Config& cfg, const ModelSource& model,
                                 std::string* error) {
  // "xnnpack=1" swaps TFLite's reference kernels for the linked XNNPACK
  // delegate: same interpreter, same zero-copy tensors. The C API does not
  // apply XNNPACK on its own (the Android JNI layer does that for the AAR),
  // so it has to be requested explicitly. The handle must outlive the
  // interpreter, which BuildTfliteEngine already guarantees through its
  // interpreter -> delegate -> model destruction order.
  DelegateHandle delegate;
  if (HasOption(cfg, opt::kXnnpack, "1")) {
    // The default options leave num_threads <= 0, which the header defines as
    // "no thread pool": TfLiteXNNPackDelegateCreate(nullptr) runs single
    // threaded even when the interpreter is told to use 4 threads (the AAR's
    // JNI layer patches this up for Java callers; the raw C API does not).
    // Copy the interpreter's thread count into the delegate options instead.
    TfLiteXNNPackDelegateOptions xnn_opts =
        TfLiteXNNPackDelegateOptionsDefault();
    // XNNPACK reads num_threads <= 0 as "no thread pool" (single-threaded),
    // unlike the interpreter where 0 means "let TFLite decide". Keep the
    // delegate default when the caller asked for auto, so the two paths agree.
    if (cfg.num_threads > 0) xnn_opts.num_threads = cfg.num_threads;
    TfLiteDelegate* xnn = TfLiteXNNPackDelegateCreate(&xnn_opts);
    if (xnn == nullptr) {
      if (error != nullptr) *error = "TfLiteXNNPackDelegateCreate failed";
      return nullptr;
    }
    // TfLiteXNNPackDelegateDelete takes exactly the plain destroy signature
    // DelegateHandle owns, so the linked delegate needs no plugin indirection.
    delegate.Adopt(xnn, &TfLiteXNNPackDelegateDelete);
  }
  return BuildTfliteEngine(cfg, model, std::move(delegate), error);
}

}  // namespace internal
}  // namespace mie
