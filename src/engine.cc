#include "mie/engine.h"

#include <memory>
#include <string>

#ifndef MIE_ENABLE_TFLITE_BACKENDS
#define MIE_ENABLE_TFLITE_BACKENDS 1
#endif
#ifndef MIE_ENABLE_RKNN
#define MIE_ENABLE_RKNN 0
#endif
#ifndef MIE_ENABLE_MTK_DLA
#define MIE_ENABLE_MTK_DLA 0
#endif
#ifndef MIE_ENABLE_QNN_NATIVE
#define MIE_ENABLE_QNN_NATIVE 0
#endif

#if MIE_ENABLE_TFLITE_BACKENDS
#include "src/backend_cpu.h"
#include "src/backend_gpu.h"
#include "src/backend_mtk.h"
#include "src/backend_qnn.h"
#endif
#if MIE_ENABLE_MTK_DLA
#include "src/backend_mtk_dla.h"
#endif
#if MIE_ENABLE_QNN_NATIVE
#include "src/backend_qnn_native.h"
#endif
#if MIE_ENABLE_RKNN
#include "src/backend_rknn.h"
#endif
#include "src/log.h"
#include "src/model_source.h"

namespace mie {
namespace {

const char* BackendName(Backend backend) {
  switch (backend) {
    case Backend::kCpu:
      return "kCpu";
    case Backend::kGpu:
      return "kGpu";
    case Backend::kQnn:
      return "kQnn";
    case Backend::kMtk:
      return "kMtk";
    case Backend::kMtkDla:
      return "kMtkDla";
    case Backend::kQnnNative:
      return "kQnnNative";
    case Backend::kRknn:
      return "kRknn";
  }
  return "unknown";
}

}  // namespace

std::unique_ptr<Engine> Engine::Create(const Config& config,
                                       std::string* error) {
  const internal::ModelSource model =
      internal::ModelSource::FromConfig(config);
  if (model.empty()) {
    if (error != nullptr) {
      *error = "no model given: set Config::model_path or Config::model_data";
    }
    return nullptr;
  }

  // delegate_lib is the plugin-ABI override and only the QNN backend reads it.
  // Saying so beats ignoring it silently, which is how a "why is my custom
  // delegate not loading?" afternoon starts.
  if (!config.delegate_lib.empty() && config.backend != Backend::kQnn) {
    MIE_LOGW(
        "Config::delegate_lib is set but the %s backend ignores it; only kQnn "
        "loads a delegate through the TFLite plugin ABI",
        BackendName(config.backend));
  }

  switch (config.backend) {
#if MIE_ENABLE_TFLITE_BACKENDS
    case Backend::kCpu:
      return internal::BuildCpu(config, model, error);
    case Backend::kGpu:
      return internal::BuildGpu(config, model, error);
    case Backend::kQnn:
      return internal::BuildQnn(config, model, error);
    case Backend::kMtk:
      return internal::BuildMtk(config, model, error);
#else
    case Backend::kCpu:
    case Backend::kGpu:
    case Backend::kQnn:
    case Backend::kMtk:
      if (error != nullptr) {
        *error = "TFLite backend family not compiled";
      }
      return nullptr;
#endif
#if MIE_ENABLE_MTK_DLA
    case Backend::kMtkDla:
      return internal::BuildMtkDla(config, model, error);
#else
    case Backend::kMtkDla:
      if (error != nullptr) {
        *error = "MediaTek DLA backend not compiled";
      }
      return nullptr;
#endif
#if MIE_ENABLE_QNN_NATIVE
    case Backend::kQnnNative:
      return internal::BuildQnnNative(config, model, error);
#else
    case Backend::kQnnNative:
      if (error != nullptr) {
        *error = "QNN native backend not compiled";
      }
      return nullptr;
#endif
#if MIE_ENABLE_RKNN
    case Backend::kRknn:
      return internal::BuildRknn(config, model, error);
#else
    case Backend::kRknn:
      if (error != nullptr) {
        *error = "RKNN backend not compiled (ARM Linux/aarch64 only)";
      }
      return nullptr;
#endif
  }
  if (error != nullptr) *error = "unknown backend";
  return nullptr;
}

}  // namespace mie
