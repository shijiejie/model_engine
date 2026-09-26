#ifndef MIE_BACKEND_MTK_DLA_H_
#define MIE_BACKEND_MTK_DLA_H_

#include <memory>
#include <string>

#include "mie/engine.h"
#include "src/model_source.h"

namespace mie {
namespace internal {

// MediaTek NPU, precompiled-model route: runs a .dla produced ahead of time by
// the NeuroPilot SDK's `ncc-tflite` compiler, through the Neuron Runtime API.
//
// This is deliberately NOT the same thing as kMtk. kMtk hands a .tflite to the
// device's TFLite delegate (libtflite_mtk.so), which compiles what it can map
// and REFUSES the rest — several real float32 graphs die there with
// NEURON_UNMAPPABLE. The offline compiler has a much larger op/target surface
// and has compiled models that the delegate path rejects, so the two backends
// cover different halves of the device.
//
// Requires MIE_ENABLE_MTK_DLA=1 plus MIE_MTK_SDK_INCLUDE_DIR pointing at the
// SDK's per-chip include directory (the one holding neuron/api/RuntimeAPI.h).
// The runtime itself is dlopen'd at run time, so nothing is linked.
std::unique_ptr<Engine> BuildMtkDla(const Config& cfg, const ModelSource& model,
                                    std::string* error);

}  // namespace internal
}  // namespace mie

#endif  // MIE_BACKEND_MTK_DLA_H_
