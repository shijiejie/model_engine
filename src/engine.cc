#include "mie/engine.h"

#include <memory>
#include <string>

#include "src/backend_cpu.h"
#include "src/backend_gpu.h"
#include "src/backend_mtk.h"
#include "src/backend_mtk_dla.h"
#include "src/backend_qnn.h"
#include "src/backend_qnn_native.h"
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
    case Backend::kCpu:
      return internal::BuildCpu(config, model, error);
    case Backend::kGpu:
      return internal::BuildGpu(config, model, error);
    case Backend::kQnn:
      return internal::BuildQnn(config, model, error);
    case Backend::kMtk:
      return internal::BuildMtk(config, model, error);
    case Backend::kMtkDla:
      return internal::BuildMtkDla(config, model, error);
    case Backend::kQnnNative:
      return internal::BuildQnnNative(config, model, error);
  }
  if (error != nullptr) *error = "unknown backend";
  return nullptr;
}

}  // namespace mie
