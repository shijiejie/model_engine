#ifndef MIE_BACKEND_RKNN_H_
#define MIE_BACKEND_RKNN_H_

#include <memory>
#include <string>

#include "mie/engine.h"
#include "src/model_source.h"

namespace mie {
namespace internal {

// Rockchip RKNN runtime. This backend is available only for ARM Linux builds
// with MIE_ENABLE_RKNN enabled. It consumes a native .rknn model and exposes
// the model's static input/output tensors through Engine's staging-buffer API.
std::unique_ptr<Engine> BuildRknn(const Config& cfg, const ModelSource& model,
                                  std::string* error);

}  // namespace internal
}  // namespace mie

#endif  // MIE_BACKEND_RKNN_H_
