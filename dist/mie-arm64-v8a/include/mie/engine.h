// Minimal inference engine.
//
// One abstraction, seven backends, one implementation file each:
//   kCpu       -> TFLite built-in kernels (multi-threaded XNNPACK)
//   kGpu       -> TFLite GPU delegate (OpenCL/OpenGL), serialized model cache
//   kQnn       -> Qualcomm QNN (AI Engine Direct) HTP/NPU, via the delegate plugin ABI
//   kQnnNative -> Qualcomm QNN HTP/NPU, a precompiled context binary via the native C API
//   kMtk       -> MediaTek NeuroPilot, via the NeuroPilotTFLiteShim runtime
//   kMtkDla    -> MediaTek DLA, a precompiled .dla via the Neuron Runtime
//   kRknn      -> Rockchip NPU, a native .rknn via RKNN Runtime (ARM Linux)
//
// CPU/GPU/QNN are all "one TFLite interpreter plus an optional delegate", so
// they share src/tflite_interpreter. kQnnNative/kMtk/kMtkDla are different
// runtimes that are not TFLite delegates at all, so each is a self-contained
// Engine implementation.
//
// Write into Input(i).data, call Run(), read Output(i).data. That is genuinely
// zero-copy on CPU/GPU/QNN. MTK's runtime only takes copied buffers, so there
// Input(i).data is an engine-owned staging buffer and Run() copies in and out —
// the calling convention is identical, the cost is not. See backend_mtk.cc.

#ifndef MIE_ENGINE_H_
#define MIE_ENGINE_H_

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace mie {

// ---------------------------------------------------------------------------
// Tensors
// ---------------------------------------------------------------------------

enum class Type : int {
  kUnknown = 0,
  kFloat32,
  kFloat16,
  kInt32,
  kInt64,
  kUInt8,
  kInt8,
  kBool,
};

struct Tensor {
  std::string name;
  Type type = Type::kUnknown;
  std::vector<int> shape;
  std::size_t bytes = 0;
  // Host view of the tensor's memory: zero-copy on kCpu/kGpu/kQnn, an
  // engine-owned staging buffer on kMtk/kMtkDla/kRknn. Writable for inputs,
  // read-only for outputs. Stays valid until ResizeInput() is called.
  void* data = nullptr;
};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

enum class Backend : int {
  kCpu = 0,
  kGpu,       // TFLite GPU delegate
  kQnn,       // Qualcomm NPU, .tflite through the TFLite delegate plugin ABI
  kMtk,       // MediaTek NPU, .tflite through the device's NeuroPilot delegate
  kMtkDla,    // MediaTek NPU, precompiled .dla through the Neuron Runtime
  kQnnNative, // Qualcomm NPU, precompiled context binary through the native C API
  kRknn,      // Rockchip NPU, native .rknn model through RKNN Runtime (ARM Linux only)
};

struct Config {
  // ---- Model source: set a path OR a buffer ----------------------------
  // Path to a model file: a .tflite for kCpu/kGpu/kQnn/kMtk, a .dla produced
  // by the NeuroPilot SDK's `ncc-tflite` for kMtkDla, a QNN context binary
  // (.bin) produced by qnn-context-binary-generator for kQnnNative, or a native
  // .rknn model for kRknn (ARM Linux only).
  // Ignored when a buffer is supplied.
  std::string model_path;

  // In-memory model (.tflite, a QNN context binary for kQnnNative, or a
  // native .rknn model for kRknn). RKNN copies the bytes during Create().
  //
  // NON-OWNING for every backend except kRknn: most runtimes may keep a pointer
  // into these bytes instead of copying them. Keep the buffer alive and
  // unmodified for the entire Engine lifetime when using those backends. The
  // kRknn backend copies the model before calling rknn_init, so its input
  // buffer only needs to remain valid during Create().
  //
  // Takes precedence over model_path when both are set.
  const void* model_data = nullptr;
  std::size_t model_size = 0;

  Backend backend = Backend::kCpu;

  // CPU only: worker thread count (0 == let TFLite decide).
  int num_threads = 4;

  // ---- Model cache (kGpu, kQnn and kMtk) -------------------------------
  // kGpu: directory holding serialized GPU kernels/data (TFLite GPU
  //       serialization). Compiled once, reused across launches.
  // kQnn: directory holding the QNN compiled context binary.
  // kMtk: directory handed to the NeuroPilot compiler cache.
  // Empty string disables caching for the backend.
  // kMtkDla needs none: the .dla file IS the compiled artifact.
  std::string cache_dir;

  // Unique id of the model inside cache_dir. Required by kGpu/kQnn/kMtk when
  // caching is on; auto-derived from the model bytes when left empty.
  std::string cache_token;

  // Escape hatch: an explicit delegate shared library to dlopen.
  // When non-empty it overrides the backend default and is loaded through the
  // TFLite external-delegate plugin ABI
  // (tflite_plugin_create_delegate), with `options` and the cache keys passed
  // as strings. Empty selects the backend default:
  //   kQnn -> "libQnnTFLiteDelegate.so"   (plugin ABI)
  // Ignored by kCpu/kGpu/kMtk/kMtkDla: the GPU delegate is linked, and neither
  // MediaTek backend is a TFLite delegate (kMtk goes through
  // NeuroPilotTFLiteShim, kMtkDla through the Neuron Runtime C API).
  std::string delegate_lib;

  // Backend-specific knobs, forwarded to the delegate as key/value strings
  // (plugin ABI) or mapped onto the vendor's own options (kMtk).
  // See the mie::opt::* constants below for well-known keys.
  std::vector<std::pair<std::string, std::string>> options;
};

// Well-known keys for Config::options. Values are backend-defined strings.
namespace opt {
// ---- CPU (kCpu) -----------------------------------------------------------
// "1" attaches TFLite's linked XNNPACK delegate instead of the reference
// kernels. Same interpreter and zero-copy tensors, but order-of-magnitude
// faster on quantized conv models (measured ~86x on SSD MobileNet v1 x86-64).
// Numerics differ slightly from the reference path (fp32 accumulation order).
// The delegate follows Config::num_threads. Needs a runtime that exports the
// XNNPACK C API (libtensorflowlite_c does, both the 2.18 Windows dll and the
// official Android AAR).
constexpr char kXnnpack[] = "xnnpack";

// ---- Shared across backends ----------------------------------------------
// "true" | "false" -> allow fp16 instead of fp32. Read by kGpu
// (is_precision_loss_allowed) and kMtk (setAllowFp16PrecisionForFp32). QNN
// does NOT read this key — the HTP already runs fp16 by default (see
// kHtpPrecision).
constexpr char kAllowFp16[] = "allow_fp16";

// ---- Qualcomm QNN (libQnnTFLiteDelegate.so) -----------------------------
// Selects the accelerator. "htp" is the kQnn default.
constexpr char kBackendType[] = "backend_type";

// Generic aliases for the per-backend performance knobs. backend_qnn.cc maps
// each onto the htp_/gpu_/dsp_-prefixed key the delegate actually reads,
// chosen by backend_type, so MIE_OPTIONS survives switching accelerators. An
// explicit prefixed key always wins over the generic one.
//   performance_mode   -> value ladders differ per backend (HTP 0..9,
//                         GPU 0..3, DSP 0..8); 1 always means "go fast"
//   perf_ctrl_strategy -> htp/dsp only (GPU has no equivalent; warn+ignore)
//   pd_session         -> htp: unsigned|signed; dsp also takes adaptive
constexpr char kPerformanceMode[] = "performance_mode";
constexpr char kPerfCtrlStrategy[] = "perf_ctrl_strategy";
constexpr char kPdSession[] = "pd_session";
// NUMERIC enum value only: 0=default, 1=sustained_high_performance, 2=burst,
// 3=high_performance, 4=power_saver ... 9=extreme_power_saver.
// WARNING: the delegate accepts nothing else. A name string such as "burst" or
// "balanced" makes it call std::terminate and abort the whole process instead
// of reporting a bad option, so never pass anything unvalidated here.
constexpr char kHtpPerformance[] = "htp_performance_mode";
// Perf control strategy: numeric, 0=manual (default) | 1=auto (HTP/DSP decide).
constexpr char kHtpPerfCtrlStrategy[] = "htp_perf_ctrl_strategy";
// Precision for fp32 graphs: numeric, 0=quantized | 1=fp16 (default). fp16 is
// already the default and the fastest, so this mainly lets you force quantized.
constexpr char kHtpPrecision[] = "htp_precision";
// Graph optimization strategy: numeric, 0=inference (default) | 1=prepare
// (faster build, less optimal graph) | 2=inference_O3 (slowest build, most
// optimal graph).
constexpr char kHtpOptimizationStrategy[] = "htp_optimization_strategy";
constexpr char kHtpPdSession[] = "htp_pd_session";  // "unsigned" | "signed"
// true|false, default true. Lets the HTP use the short-conv HMX path, which is
// faster, but convs with short depth and/or unsymmetric weights can produce
// inaccurate results. Set "false" when output numerics drift on conv-heavy
// graphs.
constexpr char kHtpUseConvHmx[] = "htp_use_conv_hmx";
// true|false, default false. Folds Relu into the preceding conv for speed.
// Numerically correct only when the conv's quantization range equals, or is a
// subset of, the Relu range — otherwise clamp semantics change.
constexpr char kHtpUseFoldRelu[] = "htp_use_fold_relu";
// Directory holding libQnnHtpV<NN>Skel.so (sets $ADSP_LIBRARY_PATH).
constexpr char kSkelLibraryDir[] = "skel_library_dir";
// Multi-HTP SoCs. NOTE: the delegate's key really is "htp_device_id"; a bare
// "device_id" is not recognized and is silently ignored.
constexpr char kHtpDeviceId[] = "htp_device_id";

// Numeric enum only: 0=user_provided, 1=fp32, 2=fp16 (default), 3=hybrid
// (fp16 math, fp32 accumulate). Controls numerics AND speed of the Adreno
// path; fp32 can be rejected by the GPU op-package on older OpenCL drivers
// (GPU_ERROR_INVALID_TYPE 10012 at graph finalize). A non-numeric value
// atoi()s to 0 (user_provided) instead of erroring, so validate before use.
constexpr char kGpuPrecision[] = "gpu_precision";
// Numeric enum: 0=default, 1=high, 2=normal, 3=low.
constexpr char kGpuPerformanceMode[] = "gpu_performance_mode";
// Directory for the GPU backend's on-disk compiled-kernel cache (kernel
// persistence): repeated launches skip OpenCL recompilation. Defaults to
// Config::cache_dir when backend_type=gpu and a cache_dir is set (wired up in
// backend_qnn.cc); set this key explicitly only to override that.
constexpr char kGpuKernelRepoDir[] = "gpu_kernel_repo_dir";

// Numeric enum: 0=default, 1=sustained, 2=burst, 3=high_performance,
// 4..8=power_saver ladder (like HTP but no 9=extreme_power_saver).
constexpr char kDspPerformanceMode[] = "dsp_performance_mode";
// Numeric, 0=manual (default) | 1=auto (DSP votes/releases its own perf mode).
constexpr char kDspPerfCtrlStrategy[] = "dsp_perf_ctrl_strategy";
// "unsigned" (default) | "signed" | "adaptive" PD session.
constexpr char kDspPdSession[] = "dsp_pd_session";
// Directory holding libQnnHtp*.so. Only useful when the QNN libraries are NOT
// already reachable via LD_LIBRARY_PATH / jniLibs — setting it in that case
// makes delegate application fail with "Restored original execution plan after
// delegate application failure". Leave it unset unless you need it.
constexpr char kLibraryPath[] = "library_path";
constexpr char kLogLevel[] = "log_level";  // numeric, QNN verbosity

// ---- Qualcomm QNN native C API (kQnnNative) -----------------------------
// Shared-library name or absolute path for the QNN HTP backend, default
// "libQnnHtp.so". dlopen'd at run time, so nothing is linked.
constexpr char kQnnNativeLibrary[] = "qnn_native_library";
// Shared-library name or absolute path for the QNN System library used to
// introspect the context binary, default "libQnnSystem.so".
constexpr char kQnnSystemLibrary[] = "qnn_system_library";
// Name of the graph to run when the context binary holds several; empty (the
// default) selects the first graph.
constexpr char kQnnGraphName[] = "qnn_graph_name";

// ---- MediaTek NeuroPilot (NeuroPilotTFLiteShim.h) -----------------------
// These map onto the shim's ANeuralNetworksTFLiteOptions_set* calls.
// "low_power" | "fast_single_answer" | "sustained_speed" -> setPreference
constexpr char kExecutionPreference[] = "execution_preference";
// Numeric -> setExecutionPriority. Higher is more important.
constexpr char kExecutionPriority[] = "execution_priority";
// Vendor accelerator id -> setAcceleratorName. On MediaTek parts the NPU is the
// "mdla" device (e.g. "mtk-mdla"). Recommended: without it, layer verification
// for some graphs reaches the vendor's ArmNN/OpenCL path, which aborts the
// process with no error message.
constexpr char kAcceleratorName[] = "accelerator_name";
// "true" | "false" -> setLowLatency
constexpr char kLowLatency[] = "low_latency";
// "true" | "false" -> setDeepFusion
constexpr char kDeepFusion[] = "deep_fusion";
// "true" | "false" -> setBatchProcessing
constexpr char kBatchProcessing[] = "batch_processing";
// "true" | "false" -> setUseIon
constexpr char kUseIon[] = "use_ion";

// ---- MediaTek DLA / Neuron Runtime (kMtkDla) -----------------------------
// Shared-library name or absolute path for the Neuron Runtime, default
// "libneuron_runtime.so" (the device's vendor copy, found through the linker
// namespace). Point it at the SDK's own build (e.g.
// "<sdk>/<chip>/lib/libneuron_runtime.so.7.3.15") when the runtime must match
// the compiler: the runtime can only load .dla files produced by an ncc-tflite
// of the same MAJOR version.
constexpr char kDlaLibrary[] = "dla_library";

// ---- Rockchip RKNN (kRknn) -----------------------------------------------
// Optional numeric rknn_core_mask value. Valid values are 0 (auto), 1, 2, 4,
// 7, or 65535. The backend is ARM Linux-only and does not use cache_dir,
// delegate_lib, or num_threads.
constexpr char kRknnCoreMask[] = "rknn_core_mask";
// "true" | "false". When false (the default), Run() submits UINT8/NHWC data
// with RKNN's conversion path, matching the scene-classifier wrappers in the
// source tree. When true, the public input view uses the model-native type,
// format, rank, and byte size.
constexpr char kRknnPassThrough[] = "rknn_pass_through";
// "true" | "false". When true (the default), Run() asks RKNN to convert every
// output to float32 before copying it into the persistent Engine output view.
// When false, the output view exposes the model-native scalar type.
constexpr char kRknnWantFloat[] = "rknn_want_float";
}  // namespace opt

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------

class Engine {
 public:
  // Builds a ready-to-run engine. Returns nullptr and fills *error on failure.
  static std::unique_ptr<Engine> Create(const Config& config,
                                        std::string* error = nullptr);

  virtual ~Engine() = default;

  virtual int NumInputs() const = 0;
  virtual int NumOutputs() const = 0;

  // Tensor views, stable for the engine lifetime. Zero-copy on kCpu/kGpu/kQnn;
  // engine-owned staging buffers on kMtk/kMtkDla/kRknn, which copy in and/or
  // out on Run().
  //
  // `index` must be within [0, NumInputs()) / [0, NumOutputs()). An out-of-range
  // index is logged and returns an empty tensor (bytes == 0, data == nullptr)
  // rather than reading past the end of the underlying array.
  virtual const Tensor& Input(int index) const = 0;
  virtual const Tensor& Output(int index) const = 0;

  // Reshape an input for dynamic models. Re-allocates the interpreter tensors
  // and re-applies the delegate; Input()/Output() views must be re-read after
  // this call. Returns false if the shape is invalid. The self-contained
  // precompiled/runtime backends (kMtk/kMtkDla/kQnnNative/kRknn) currently
  // expose static shapes and return false.
  virtual bool ResizeInput(int index, const std::vector<int>& shape) = 0;

  // Runs inference on the current Input() buffers. Returns false on failure
  // (bad input, delegate error, device error); the outputs are then undefined.
  virtual bool Run() = 0;
};

}  // namespace mie

#endif  // MIE_ENGINE_H_
