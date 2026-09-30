// Minimal end-to-end driver: build an engine, feed a deterministic pattern,
// verify the output is not degenerate, measure steady-state latency.
//
//   mie_run model [cpu|gpu|qnn|mtk|dla|qnnnative|rknn] [cache_dir] [runs]

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "mie/engine.h"

#include "common.h"

namespace {

void PrintTensor(const char* label, int index, const mie::Tensor& tensor) {
  std::printf("  %s[%d] %-16s type=%d bytes=%zu shape=[", label, index,
              tensor.name.c_str(), static_cast<int>(tensor.type), tensor.bytes);
  for (std::size_t i = 0; i < tensor.shape.size(); ++i) {
    std::printf("%s%d", i == 0 ? "" : ",", tensor.shape[i]);
  }
  std::printf("]\n");
}

// Prints the head of a tensor, decoded per type, so a human can sanity-check it.
void PrintOutputHead(const mie::Tensor& tensor, std::size_t count) {
  std::printf("out[0] head:");
  if (tensor.type == mie::Type::kFloat32) {
    const float* values = static_cast<const float*>(tensor.data);
    const std::size_t available = tensor.bytes / sizeof(float);
    for (std::size_t i = 0; i < count && i < available; ++i) {
      std::printf(" %.4f", values[i]);
    }
  } else {
    const unsigned char* bytes = static_cast<const unsigned char*>(tensor.data);
    for (std::size_t i = 0; i < count && i < tensor.bytes; ++i) {
      std::printf(" %02x", bytes[i]);
    }
  }
  std::printf("\n");
}

// Reads a whole file. Used to exercise the in-memory model-buffer path.
bool ReadWholeFile(const char* path, std::vector<char>* out) {
  std::FILE* file = std::fopen(path, "rb");
  if (file == nullptr) return false;
  char chunk[16384];
  std::size_t n = 0;
  while ((n = std::fread(chunk, 1, sizeof(chunk), file)) > 0) {
    out->insert(out->end(), chunk, chunk + n);
  }
  const bool ok = std::ferror(file) == 0;
  std::fclose(file);
  return ok;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc > 1 && (std::strcmp(argv[1], "-h") == 0 ||
                   std::strcmp(argv[1], "--help") == 0)) {
    std::printf(
        "usage: %s model [cpu|gpu|qnn|mtk|dla|qnnnative|rknn] [cache_dir] "
        "[runs] [threads]\n"
        "  (dla takes a precompiled .dla; qnnnative takes a QNN context .bin;\n"
        "   rknn takes a native .rknn model and is ARM Linux-only)\n"
        "  env MIE_MODEL_BUFFER=1   pass the model as an in-memory "
        "buffer instead of a path\n"
        "  env MIE_PROBE_OOB=1      exercise the Input()/Output() range guard\n",
        argv[0]);
  example::PrintOptionHelp();
  return 0;
  }
  if (argc < 2) {
    std::fprintf(stderr,
                 "usage: %s model [cpu|gpu|qnn|mtk|dla|qnnnative|rknn] [cache_dir] [runs] [threads]\n"
                 "  (dla takes a precompiled .dla; qnnnative takes a QNN context .bin;\n"
                 "   rknn takes a native .rknn model and is ARM Linux-only)\n"
                 "  env MIE_MODEL_BUFFER=1   pass the model as an in-memory "
                 "buffer instead of a path\n"
                 "  env MIE_OPTIONS=k=v,...  extra backend options "
                 "(list them with: %s --help)\n",
                 argv[0], argv[0]);
    return 1;
  }

  // Declared before the engine on purpose. Config::model_data is NON-OWNING, so
  // whatever backs it has to be destroyed after the engine; declaring it first
  // makes that fall out of C++'s reverse destruction order.
  std::vector<char> model_bytes;
  std::string error;

  mie::Config config;
  config.model_path = argv[1];
  if (std::getenv("MIE_MODEL_BUFFER") != nullptr) {
    if (!ReadWholeFile(argv[1], &model_bytes) || model_bytes.empty()) {
      std::fprintf(stderr, "failed to read model into memory: %s\n", argv[1]);
      return 1;
    }
    config.model_data = model_bytes.data();
    config.model_size = model_bytes.size();
  }
  mie::Backend backend;
  if (!example::ParseBackend(argc > 2 ? argv[2] : nullptr, &backend, &error)) {
    std::fprintf(stderr, "%s\n", error.c_str());
    return 1;
  }
  config.backend = backend;
  if (argc > 3) config.cache_dir = argv[3];
  const int runs = argc > 4 ? std::atoi(argv[4]) : 20;
  // Only a positive count overrides the Config default; garbage or 0 keeps
  // num_threads untouched (TFLite maps <= 0 to "no explicit setting" anyway).
  if (argc > 5) {
    const int threads = std::atoi(argv[5]);
    if (threads > 0) config.num_threads = threads;
    // num_threads is consumed only by the kCpu backend (interpreter + XNNPACK).
    // The accelerator backends manage their own parallelism, so a [threads]
    // value here would be silently dropped — say so instead of guessing.
    if (config.backend != mie::Backend::kCpu) {
      std::fprintf(stderr,
                   "warning: [threads] is CPU-only and ignored by the %s backend\n",
                   argv[2]);
    }
  }

  // No backend-specific options are forced here: a driver should not silently
  // pick a power profile. Pass what the backend needs through the environment,
  // e.g. MIE_OPTIONS="library_path=/data/local/tmp/mie,skel_library_dir=..."
  example::AppendEnvOptions(&config);

  // Echo what is actually in effect — vendor delegates are opaque, and a run
  // that "behaves differently" is often just an option that never reached it.
  std::fprintf(stderr, "options in effect:");
  if (config.options.empty()) {
    std::fprintf(stderr, " (none)");
  } else {
    for (const auto& kv : config.options) {
      std::fprintf(stderr, " %s=%s", kv.first.c_str(), kv.second.c_str());
    }
  }
  std::fprintf(stderr, "\n");

  const auto create_start = std::chrono::steady_clock::now();
  std::unique_ptr<mie::Engine> engine = mie::Engine::Create(config, &error);
  const auto create_end = std::chrono::steady_clock::now();
  if (engine == nullptr) {
    std::fprintf(stderr, "Engine::Create failed: %s\n", error.c_str());
    return 1;
  }
  const double init_ms = std::chrono::duration<double, std::milli>(
                             create_end - create_start).count();

  std::printf("backend=%s inputs=%d outputs=%d init=%.1f ms\n",
              argc > 2 ? argv[2] : "cpu", engine->NumInputs(),
              engine->NumOutputs(), init_ms);
  for (int i = 0; i < engine->NumInputs(); ++i) {
    PrintTensor("in", i, engine->Input(i));
  }
  for (int i = 0; i < engine->NumOutputs(); ++i) {
    PrintTensor("out", i, engine->Output(i));
  }

  // MIE_PROBE_OOB=1 exercises the out-of-range guard on Input()/Output(). Under
  // AddressSanitizer this must produce a log line and NOT a heap-buffer-overflow.
  if (std::getenv("MIE_PROBE_OOB") != nullptr) {
    const mie::Tensor& below = engine->Input(-1);
    const mie::Tensor& above = engine->Input(engine->NumInputs());
    const mie::Tensor& out_above = engine->Output(engine->NumOutputs());
    std::printf("oob probe: in[-1].bytes=%zu in[%d].bytes=%zu out[%d].bytes=%zu\n",
                below.bytes, engine->NumInputs(), above.bytes,
                engine->NumOutputs(), out_above.bytes);
  }

  for (int j = 0; j < engine->NumInputs(); ++j) example::FillInput(engine->Input(j));

  // Warm up, then time the steady state. The first Run() can pay one-time
  // device costs (lazy APU power-up, first-request setup on the DLA path), so
  // each warm-up is timed and printed here — and NONE of them enters the
  // steady-state average below.
  for (int i = 0; i < 3; ++i) {
    const auto warm_start = std::chrono::steady_clock::now();
    if (!engine->Run()) {
      std::fprintf(stderr, "warm-up Run failed\n");
      return 1;
    }
    const auto warm_end = std::chrono::steady_clock::now();
    std::printf("warmup %d: %.3f ms\n", i + 1,
                std::chrono::duration<double, std::milli>(warm_end - warm_start)
                    .count());
  }

  const auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < runs; ++i) {
    if (!engine->Run()) {
      std::fprintf(stderr, "Run failed\n");
      return 1;
    }
  }
  const auto end = std::chrono::steady_clock::now();
  const double micros =
      std::chrono::duration<double, std::micro>(end - start).count() / runs;
  std::printf("avg %.1f us (%.3f ms) over %d runs\n", micros, micros / 1000.0,
              runs);

  PrintOutputHead(engine->Output(0), 8);
  for (int i = 0; i < engine->NumOutputs(); ++i) {
    std::printf("out[%d] checksum: %08x\n", i,
                example::TensorChecksum(engine->Output(i)));
  }
  // A warning rather than a failure: plenty of real models legitimately emit an
  // all-zero tensor (no detections, a zero count, ...). For a model known to
  // produce non-zero output this still catches a dead data path.
  for (int i = 0; i < engine->NumOutputs(); ++i) {
    if (example::IsDegenerate(engine->Output(i))) {
      std::fprintf(stderr,
                   "WARN: output[%d] is all zeros for a non-zero input\n", i);
    }
  }
  std::printf("OK\n");
  return 0;
}
