#ifndef MIE_BACKEND_QNN_NATIVE_H_
#define MIE_BACKEND_QNN_NATIVE_H_

#include <memory>
#include <string>

#include "mie/engine.h"
#include "src/model_source.h"

namespace mie {
namespace internal {

// Qualcomm QNN (AI Engine Direct) HTP/NPU through the native C API — NOT
// through the TFLite delegate. Runs a precompiled context binary (a .bin
// produced by qnn-context-binary-generator, or the context binary the delegate
// serializes to its cache) via QnnInterface_getProviders() +
// QnnContext_createFromBinary(). No TFLite anywhere in the loop.
//
// This is deliberately NOT kQnn. kQnn hands a .tflite to libQnnTFLiteDelegate.so
// for online compilation. The native path expects the QNN compiler to have
// already run offline, which mirrors kMtkDla's contract (its input is a .dla,
// not a .tflite).
//
// Self-contained like backend_mtk_dla.cc: it dlopens libQnnHtp.so (the QNN
// core interface) and libQnnSystem.so (context-binary introspection, to read
// graph/tensor metadata without compiling anything), so nothing QNN is linked
// at build time.
//
// Requires MIE_ENABLE_QNN_NATIVE=1 plus MIE_QNN_SDK_INCLUDE_DIR pointing at the
// QNN SDK include dir (the one holding QnnInterface.h and System/...).
// Those headers are confidential Qualcomm material, so they are not fetched
// automatically — same policy as the MediaTek shim/SDK headers.
std::unique_ptr<Engine> BuildQnnNative(const Config& cfg, const ModelSource& model,
                                       std::string* error);

}  // namespace internal
}  // namespace mie

#endif  // MIE_BACKEND_QNN_NATIVE_H_