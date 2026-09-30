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

// Generic alias keys (performance_mode, perf_ctrl_strategy, pd_session) mapped
// onto the prefixed key each accelerator actually reads. nullptr = this
// backend has no equivalent: the alias is warned about and dropped, since the
// delegate would reject the bare key anyway.
struct BackendAliases {
  const char* performance_mode;
  const char* perf_ctrl_strategy;
  const char* pd_session;
};

BackendAliases AliasesForBackend(const std::string& backend) {
  if (backend == "gpu") {
    return {opt::kGpuPerformanceMode, nullptr, nullptr};
  }
  if (backend == "dsp") {
    return {opt::kDspPerformanceMode, opt::kDspPerfCtrlStrategy,
            opt::kDspPdSession};
  }
  return {opt::kHtpPerformance, opt::kHtpPerfCtrlStrategy, opt::kHtpPdSession};
}

// Resolves the generic aliases against the effective backend_type and strips
// them from the list, so the delegate only ever sees keys it knows. An
// explicit prefixed key wins (SetOption never overwrites an existing entry).
void ApplyBackendAliases(Options* options) {
  const std::string* backend = FindOption(*options, opt::kBackendType);
  const std::string name = backend != nullptr ? *backend : "htp";
  const BackendAliases keys = AliasesForBackend(name);

  const std::pair<const char*, const char*> aliases[] = {
      {opt::kPerformanceMode, keys.performance_mode},
      {opt::kPerfCtrlStrategy, keys.perf_ctrl_strategy},
      {opt::kPdSession, keys.pd_session},
  };
  for (const auto& alias : aliases) {
    const std::string* value = FindOption(*options, alias.first);
    if (value == nullptr) continue;
    if (alias.second != nullptr) {
      SetOption(options, alias.second, *value);
    } else {
      MIE_LOGW("option '%s' has no %s-backend equivalent; ignored",
               alias.first, name.c_str());
    }
  }
  RemoveOption(options, opt::kPerformanceMode);
  RemoveOption(options, opt::kPerfCtrlStrategy);
  RemoveOption(options, opt::kPdSession);
}

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
    // The GPU backend keeps its compiled-kernel repo on disk too; point it at
    // the same cache_dir so backend_type=gpu gets kernel persistence without a
    // separate knob. A caller-supplied gpu_kernel_repo_dir wins (SetOption
    // never overwrites), and the other backends ignore the key.
    const std::string* backend = FindOption(options, opt::kBackendType);
    if (backend != nullptr && *backend == "gpu") {
      SetOption(&options, opt::kGpuKernelRepoDir, cfg.cache_dir);
    }
  }

  DelegateHandle handle;
  ApplyBackendAliases(&options);
  if (!LoadPluginDelegate(lib, options, &handle, error)) return nullptr;
  return BuildTfliteEngine(cfg, model, std::move(handle), error);
}

}  // namespace internal
}  // namespace mie
