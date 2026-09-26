#ifndef MIE_INTERNAL_DELEGATE_PLUGIN_H_
#define MIE_INTERNAL_DELEGATE_PLUGIN_H_

#include <string>

#include "src/delegate_handle.h"
#include "src/options.h"

namespace mie {
namespace internal {

// dlopen `lib`, resolve the TFLite external-delegate plugin entry points
// (tflite_plugin_create_delegate / tflite_plugin_destroy_delegate) and create a
// delegate from `options`.
//
// On success `*out` owns both the delegate and the library handle. Returns
// false and fills *error otherwise.
bool LoadPluginDelegate(const std::string& lib, const Options& options,
                        DelegateHandle* out, std::string* error);

}  // namespace internal
}  // namespace mie

#endif  // MIE_INTERNAL_DELEGATE_PLUGIN_H_
