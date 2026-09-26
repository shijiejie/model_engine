#ifndef MIE_BACKEND_MTK_H_
#define MIE_BACKEND_MTK_H_

#include <memory>
#include <string>

#include "mie/engine.h"
#include "src/model_source.h"

namespace mie {
namespace internal {

// MediaTek NeuroPilot NPU.
//
// Unlike the other three, this is NOT a TFLite interpreter with a delegate
// bolted on: NeuroPilotTFLiteShim.h is a separate runtime that dlopens
// libtflite_mtk.so and exposes its own tensor API. So this backend is a
// self-contained Engine implementation, and it needs no link-time library.
//
// Requires MIE_ENABLE_MTK=ON plus include/ on the SDK path, and ANDROID_API>=27
// because the shim depends on <android/NeuralNetworks.h>.
std::unique_ptr<Engine> BuildMtk(const Config& cfg, const ModelSource& model,
                                 std::string* error);

}  // namespace internal
}  // namespace mie

#endif  // MIE_BACKEND_MTK_H_
