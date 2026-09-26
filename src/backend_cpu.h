#ifndef MIE_BACKEND_CPU_H_
#define MIE_BACKEND_CPU_H_

#include <memory>
#include <string>

#include "mie/engine.h"
#include "src/model_source.h"

namespace mie {
namespace internal {

// Plain TFLite interpreter: built-in kernels, multi-threaded, no delegate.
std::unique_ptr<Engine> BuildCpu(const Config& cfg, const ModelSource& model,
                                 std::string* error);

}  // namespace internal
}  // namespace mie

#endif  // MIE_BACKEND_CPU_H_
