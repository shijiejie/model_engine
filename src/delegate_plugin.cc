#include "src/delegate_plugin.h"

#include <cstddef>
#include <vector>

#include "src/dl.h"
#include "src/log.h"

namespace mie {
namespace internal {
namespace {

// Signature of the external-delegate plugin entry points, as declared in
// tensorflow/lite/delegates/gpu/delegate.h and external_delegate.h.
using PluginCreateFn = TfLiteDelegate* (*)(const char* const* keys,
                                           const char* const* values,
                                           std::size_t num_options,
                                           void (*report_error)(const char*));
using PluginDestroyFn = void (*)(TfLiteDelegate*);

void PluginReportError(const char* message) {
  MIE_LOGW("delegate plugin: %s", message != nullptr ? message : "(null)");
}

}  // namespace

bool LoadPluginDelegate(const std::string& lib, const Options& options,
                        DelegateHandle* out, std::string* error) {
  LibHandle handle = DlOpen(lib.c_str());
  if (handle == nullptr) {
    if (error != nullptr) *error = "failed to dlopen delegate library: " + lib;
    return false;
  }

  auto create = reinterpret_cast<PluginCreateFn>(
      DlSym(handle, "tflite_plugin_create_delegate"));
  auto destroy = reinterpret_cast<PluginDestroyFn>(
      DlSym(handle, "tflite_plugin_destroy_delegate"));
  if (create == nullptr || destroy == nullptr) {
    DlClose(handle);
    if (error != nullptr) {
      *error = lib + " does not export the TFLite external-delegate plugin ABI";
    }
    return false;
  }

  std::vector<const char*> keys;
  std::vector<const char*> values;
  keys.reserve(options.size());
  values.reserve(options.size());
  for (const auto& kv : options) {
    keys.push_back(kv.first.c_str());
    values.push_back(kv.second.c_str());
  }

  TfLiteDelegate* delegate =
      create(keys.data(), values.data(), keys.size(), &PluginReportError);
  if (delegate == nullptr) {
    DlClose(handle);
    if (error != nullptr) *error = "delegate plugin returned null: " + lib;
    return false;
  }

  out->AdoptPlugin(delegate, destroy, handle);
  return true;
}

}  // namespace internal
}  // namespace mie
