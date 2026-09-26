#ifndef MIE_BACKEND_GPU_H_
#define MIE_BACKEND_GPU_H_

#include <memory>
#include <string>

#include "mie/engine.h"
#include "src/model_source.h"

namespace mie {
namespace internal {

// TFLite interpreter + the linked GPU delegate (OpenCL/OpenGL).
// Honours Config::cache_dir via the delegate's kernel/model serialization.
std::unique_ptr<Engine> BuildGpu(const Config& cfg, const ModelSource& model,
                                 std::string* error);

}  // namespace internal
}  // namespace mie

#endif  // MIE_BACKEND_GPU_H_
