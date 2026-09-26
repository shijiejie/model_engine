#ifndef MIE_BACKEND_QNN_H_
#define MIE_BACKEND_QNN_H_

#include <memory>
#include <string>

#include "mie/engine.h"
#include "src/model_source.h"

namespace mie {
namespace internal {

// TFLite interpreter + the Qualcomm QNN delegate, loaded at runtime through the
// external-delegate plugin ABI so there is no build-time QNN dependency.
// Honours Config::cache_dir via the delegate's context-binary cache.
std::unique_ptr<Engine> BuildQnn(const Config& cfg, const ModelSource& model,
                                 std::string* error);

}  // namespace internal
}  // namespace mie

#endif  // MIE_BACKEND_QNN_H_
