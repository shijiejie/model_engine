#include "src/backend_qnn.h"

#include <utility>

#include "src/delegate_plugin.h"
#include "src/fingerprint.h"
#include "src/log.h"
#include "src/options.h"
#include "src/tflite_interpreter.h"

namespace mie {
namespace internal {
namespace {

// Qualcomm ships its delegate as libQnnTFLiteDelegate.so, exporting both its own
// C API and the TFLite external-delegate plugin ABI.
constexpr char kQnnDelegateLib[] = "libQnnTFLiteDelegate.so";
constexpr char kCacheDirKey[] = "cache_dir";
constexpr char kModelTokenKey[] = "model_token";

}  // namespace

std::unique_ptr<Engine> BuildQnn(const Config& cfg, const ModelSource& model,
                                 std::string* error) {
  const std::string lib =
      cfg.delegate_lib.empty() ? kQnnDelegateLib : cfg.delegate_lib;

  Options options = cfg.options;
  // kQnn means "the Qualcomm NPU", so default the accelerator to the Hexagon
  // HTP. The caller can still override backend_type through cfg.options.
  //
  // Do NOT default htp_performance_mode here: the delegate accepts only the
  // numeric enum value and hard-aborts the process on anything else, so a
  // wrong default would be a crash rather than a clear error.
  SetOption(&options, opt::kBackendType, "htp");

  if (!cfg.cache_dir.empty()) {
    SetOption(&options, kCacheDirKey, cfg.cache_dir);
    const std::string token = CacheToken(cfg, model);
    if (token.empty()) {
      // The QNN delegate needs BOTH keys; without a token it ignores cache_dir
      // entirely. Say so, rather than leaving the caller to wonder why every
      // launch stays slow. backend_gpu escalates this same condition to an error.
      MIE_LOGW(
          "cache_dir is set but no cache token could be derived from this "
          "model; the QNN cache will NOT be used. Set Config::cache_token.");
    } else {
      SetOption(&options, kModelTokenKey, token);
    }
  }

  DelegateHandle handle;
  if (!LoadPluginDelegate(lib, options, &handle, error)) return nullptr;
  return BuildTfliteEngine(cfg, model, std::move(handle), error);
}

}  // namespace internal
}  // namespace mie
