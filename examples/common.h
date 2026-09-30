// Shared helpers for the example programs. Kept here so mie_run and mie_bench
// cannot drift apart on argument semantics (backend names, option strings).
//
// C++11 on purpose — the Android build targets NDK r16b (clang 5.0).

#ifndef MIE_EXAMPLES_COMMON_H_
#define MIE_EXAMPLES_COMMON_H_

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "mie/engine.h"

namespace example {

// Strict backend name lookup. An unknown name must fail loudly, not silently
// fall back to cpu — a mistyped or misplaced argument (e.g. a path landing in
// the backend slot) would otherwise run the wrong backend and look healthy.
inline bool ParseBackend(const char* name, mie::Backend* backend,
                         std::string* error) {
  const std::string s = name != nullptr ? name : "";
  if (s.empty() || s == "cpu") {
    *backend = mie::Backend::kCpu;
    return true;
  }
  if (s == "gpu") {
    *backend = mie::Backend::kGpu;
    return true;
  }
  if (s == "qnn") {
    *backend = mie::Backend::kQnn;
    return true;
  }
  if (s == "mtk") {
    *backend = mie::Backend::kMtk;
    return true;
  }
  if (s == "dla" || s == "mtk-dla") {
    *backend = mie::Backend::kMtkDla;
    return true;
  }
  if (s == "qnnnative" || s == "qnn-native") {
    *backend = mie::Backend::kQnnNative;
    return true;
  }
  if (s == "rknn") {
    *backend = mie::Backend::kRknn;
    return true;
  }
  *error = "unknown backend '" + s +
           "' (valid: cpu|gpu|qnn|mtk|dla|qnnnative|rknn)";
  return false;
}

// Splits "key=value,key=value" into Config::options. Malformed pieces (no '=',
// empty key) are skipped silently — this feeds vendor delegates whose own
// logging is the better place to catch a bad key.
inline void ParseOptionsString(const std::string& all, mie::Config* config) {
  std::size_t pos = 0;
  while (pos < all.size()) {
    std::size_t comma = all.find(',', pos);
    if (comma == std::string::npos) comma = all.size();
    const std::string kv = all.substr(pos, comma - pos);
    const std::size_t eq = kv.find('=');
    if (eq != std::string::npos && eq > 0) {
      config->options.push_back(
          std::make_pair(kv.substr(0, eq), kv.substr(eq + 1)));
    }
    pos = comma + 1;
  }
}

// Extra backend options from the environment, "key=value,key=value". A
// debugging aid: vendor delegates are opaque, and their own logging is usually
// the fastest way to find out why one refused to initialise.
inline void AppendEnvOptions(mie::Config* config) {
  const char* raw = std::getenv("MIE_OPTIONS");
  if (raw == nullptr) return;
  ParseOptionsString(std::string(raw), config);
}

// Deterministic, non-zero fill. Filling with zeros would let a model that never
// ran produce the "right" answer, so the pattern has to be distinguishable.
inline void FillInput(const mie::Tensor& tensor) {
  if (tensor.data == nullptr) return;
  if (tensor.type == mie::Type::kFloat32) {
    float* values = static_cast<float*>(tensor.data);
    const std::size_t count = tensor.bytes / sizeof(float);
    for (std::size_t i = 0; i < count; ++i) values[i] = 1.0f;
  } else {
    std::memset(tensor.data, 1, tensor.bytes);
  }
}

// A cheap "did anything actually happen" check: an all-zero output after a
// non-zero input means the data path or the model is broken.
inline bool IsDegenerate(const mie::Tensor& tensor) {
  const unsigned char* bytes = static_cast<const unsigned char*>(tensor.data);
  if (bytes == nullptr) return true;
  for (std::size_t i = 0; i < tensor.bytes; ++i) {
    if (bytes[i] != 0) return false;
  }
  return true;
}

// FNV-1a over the WHOLE output buffer, so two runs can be compared byte-exactly
// without dumping megabytes.
inline std::uint32_t TensorChecksum(const mie::Tensor& tensor) {
  const unsigned char* bytes = static_cast<const unsigned char*>(tensor.data);
  std::uint32_t hash = 2166136261u;
  if (bytes == nullptr) return hash;
  for (std::size_t i = 0; i < tensor.bytes; ++i) {
    hash ^= static_cast<std::uint32_t>(bytes[i]);
    hash *= 16777619u;
  }
  return hash;
}

// Every well-known Config::options key, grouped by backend, with values,
// defaults and what each knob is FOR. Kept in sync with the mie::opt::*
// constants in mie/engine.h — the constants are the source of truth; this is
// the human-readable mirror. Both example programs print this from --help.
inline void PrintOptionHelp() {
  std::printf(
      "backend options (MIE_OPTIONS=\"key=value,key=value\" or mie_bench -o; "
      "all keys are strings):\n"
      "\n"
      "  cpu (kCpu):\n"
      "    xnnpack=1                       attach the XNNPACK delegate instead\n"
      "                                    of TFLite's reference kernels --\n"
      "                                    faster on quantized conv models\n"
      "                                    (~86x measured on SSD MobileNet v1),\n"
      "                                    slightly different fp rounding.\n"
      "                                    Default off; follows [threads].\n"
      "\n"
      "  shared:\n"
      "    allow_fp16=true|false           let fp32 ops run as fp16 -- faster\n"
      "                                    and smaller, small accuracy cost.\n"
      "                                    Read by the TFLite GPU delegate and\n"
      "                                    MTK; ignored by qnn (HTP is fp16\n"
      "                                    anyway).\n"
      "\n"
      "  qnn / qnnnative environment (Android):\n"
      "    The Android linker does not search the current directory, so a\n"
      "    bare ./mie_run fails with \"library ... not found\". Export both\n"
      "    of these before running (paths shown for a flat deployment dir):\n"
      "\n"
      "      export LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH\n"
      "        must reach libtensorflowlite_c.so (linked by the demos\n"
      "        themselves) and, for qnn, libQnnTFLiteDelegate.so plus the\n"
      "        QNN backend libraries it dlopens: libQnnHtp.so /\n"
      "        libQnnGpu.so / libQnnDsp.so, their libQnnHtpV<NN>Stub.so\n"
      "        companions, libQnnHtpPrepare.so, and libQnnSystem.so.\n"
      "        Backend libs resolve by soname in LD_LIBRARY_PATH order, so\n"
      "        a stale SDK copy earlier in the path silently wins and\n"
      "        version-mismatches the delegate (SIGSEGV).\n"
      "\n"
      "      export ADSP_LIBRARY_PATH=./skels\n"
      "        must reach the Hexagon skels (libQnnHtpV<NN>Skel.so,\n"
      "        libQnnDspV<NN>Skel.so) that the CDSP session loads when it\n"
      "        opens. The qnn delegate can set this itself via the\n"
      "        skel_library_dir option; qnnnative cannot -- always export\n"
      "        ADSP_LIBRARY_PATH for it.\n"
      "\n"
      "  qnn (libQnnTFLiteDelegate.so):\n"
      "    backend_type=htp|gpu|dsp        which accelerator (default htp). dsp\n"
      "                                    needs an int8-quantized model plus\n"
      "                                    DSP session access (root/testsig on\n"
      "                                    retail devices). There is NO \"cpu\"\n"
      "                                    value -- it parses as an undefined\n"
      "                                    backend and crashes.\n"
      "    performance_mode=<n>            one knob for every accelerator:\n"
      "                                    mapped to htp_/gpu_/dsp_performance_\n"
      "                                    mode by backend_type. Ladders htp\n"
      "                                    0..9, gpu 0..3, dsp 0..8 -- 1 always\n"
      "                                    means \"go fast\".\n"
      "    perf_ctrl_strategy=0|1          generic alias, htp/dsp only: 0 =\n"
      "                                    apply performance_mode manually\n"
      "                                    (default), 1 = accelerator manages\n"
      "                                    its own clocks.\n"
      "    pd_session=unsigned|signed      generic alias for the Hexagon\n"
      "                                    protection-domain selection (dsp\n"
      "                                    also takes adaptive; see *_pd_session).\n"
      "    htp_performance_mode=0..9       HTP power/perf point: 0 default,\n"
      "                                    1 sustained, 2 burst, 3 high_perf,\n"
      "                                    4 power_saver, 9 extreme_power_saver.\n"
      "                                    Numeric only -- a name string like\n"
      "                                    \"burst\" aborts the process.\n"
      "    htp_perf_ctrl_strategy=0|1      0 manual (default), 1 auto (HTP\n"
      "                                    decides its own clocks).\n"
      "    htp_precision=0|1               fp32 graphs only: 0 quantized (int8\n"
      "                                    path), 1 fp16 (default, fastest).\n"
      "                                    int8 models ignore it.\n"
      "    htp_optimization_strategy=0|1|2 graph-build trade-off: 0 inference\n"
      "                                    (default), 1 prepare (faster build,\n"
      "                                    less optimal graph), 2 inference_O3\n"
      "                                    (slowest build, most optimal).\n"
      "    htp_pd_session=unsigned|signed  Hexagon protection domain: unsigned\n"
      "                                    is the default and needs unsigned\n"
      "                                    skels; signed needs a vendor-signed\n"
      "                                    skel/PD.\n"
      "    htp_device_id=<id>              pick one HTP on multi-HTP SoCs. NOT\n"
      "                                    \"device_id\" -- that key is\n"
      "                                    silently ignored by the delegate.\n"
      "    htp_use_conv_hmx=true|false     default true: run short convs on the\n"
      "                                    HMX unit. Faster, but short-depth or\n"
      "                                    unsymmetric weights may lose accuracy.\n"
      "    htp_use_fold_relu=true|false    default false: fuse ReLU into the\n"
      "                                    preceding conv. Faster, correct only\n"
      "                                    when the conv quant range fits the\n"
      "                                    ReLU range.\n"
      "    skel_library_dir=<dir>          holds libQnnHtpV<NN>Skel.so or\n"
      "                                    libQnnDspV<NN>Skel.so; sets\n"
      "                                    ADSP_LIBRARY_PATH (htp/dsp).\n"
      "    gpu_precision=0..3              GPU numerics: 0 user_provided,\n"
      "                                    1 fp32, 2 fp16 (default), 3 hybrid\n"
      "                                    (fp16 math, fp32 accumulate).\n"
      "                                    Numeric only -- non-numeric atoi()s\n"
      "                                    to 0.\n"
      "    gpu_performance_mode=0..3       0 default, 1 high, 2 normal, 3 low.\n"
      "    gpu_kernel_repo_dir=<dir>       on-disk compiled-OpenCL-kernel cache\n"
      "                                    so relaunches skip recompilation;\n"
      "                                    defaults to [cache_dir] when\n"
      "                                    backend_type=gpu.\n"
      "    dsp_performance_mode=0..8       same ladder as htp, minus 9\n"
      "                                    (extreme_power_saver).\n"
      "    dsp_perf_ctrl_strategy=0|1      0 manual (default), 1 auto (DSP\n"
      "                                    decides its own clocks).\n"
      "    dsp_pd_session=unsigned|signed|adaptive\n"
      "                                    Hexagon protection-domain selection.\n"
      "    library_path=<dir>              holds libQnnHtp*.so / libQnnGpu.so /\n"
      "                                    libQnnDsp.so. Leave unset when\n"
      "                                    LD_LIBRARY_PATH already covers it --\n"
      "                                    setting it redundantly breaks\n"
      "                                    delegate application.\n"
      "    log_level=<n>                   QNN log verbosity; 5 prints every\n"
      "                                    option the delegate parsed -- the\n"
      "                                    fastest way to see what actually\n"
      "                                    reached it.\n"
      "\n"
      "  qnnnative (QNN context .bin):\n"
      "    qnn_native_library=<lib>        QNN backend .so to dlopen; default\n"
      "                                    libQnnHtp.so, e.g. libQnnDsp.so for\n"
      "                                    the DSP backend.\n"
      "    qnn_system_library=<lib>        default libQnnSystem.so; used to\n"
      "                                    introspect the context binary's\n"
      "                                    graphs and tensors.\n"
      "    qnn_graph_name=<name>           which graph to run when the .bin\n"
      "                                    holds several (default: the first).\n"
      "\n"
      "  mtk (NeuroPilot / NNAPI):\n"
      "    execution_preference=low_power|fast_single_answer|sustained_speed\n"
      "                                    NNAPI power hint for the accelerator.\n"
      "    execution_priority=<n>          numeric; higher is more important.\n"
      "    accelerator_name=<name>         force a device accelerator, e.g.\n"
      "                                    mtk-mdla (recommended, see README).\n"
      "    low_latency=true|false          favor single-inference latency over\n"
      "                                    throughput.\n"
      "    deep_fusion=true|false          vendor multi-op fusion optimization.\n"
      "    batch_processing=true|false     favor throughput over latency.\n"
      "    use_ion=true|false              ION zero-copy buffers for tensor I/O.\n"
      "\n"
      "  dla (MediaTek Neuron Runtime):\n"
      "    dla_library=<lib>               default libneuron_runtime.so (vendor\n"
      "                                    copy; must match the ncc-tflite major\n"
      "                                    version that built the .dla).\n"
      "\n"
      "  rknn (ARM Linux only):\n"
      "    rknn_core_mask=0|1|2|4|7|65535  which NPU cores: 0 auto (default),\n"
      "                                    1/2/4 = core 0/1/2, 7 = all three,\n"
      "                                    65535 = ALL.\n"
      "    rknn_pass_through=true|false    default false: skip RKNN's built-in\n"
      "                                    pre/post-processing.\n"
      "    rknn_want_float=true|false      default true: return float outputs\n"
      "                                    instead of the model's quantized\n"
      "                                    format.\n");
}

}  // namespace example

#endif  // MIE_EXAMPLES_COMMON_H_
