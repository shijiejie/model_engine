// Diagnostic for the MediaTek DLA backend's buffer-binding contract.
//
// backend_mtk_dla.cc drives the V2 request-based runtime API
// (NeuronRuntimeV2_*): every Run() carries the IOBuffer descriptors in the
// request, and the runtime re-reads the staging buffers each time — so this
// probe PASSES on the verified device (libneuron_runtime 7.3.15):
// rewriting the input between runs flips the output.
//
// It exists because the V1 family (NeuronRuntime_* set-once-then-infer) was
// different: there the input was LATCHED at the first inference and no
// re-setInput, clone, enqueue-trigger or create-option could revive it (the
// full 24-step matrix is in tools/dla_raw_probe.cc). If the backend ever
// moves back to a set-once API — or a runtime regresses — this probe is what
// catches it: FAIL means the output stopped tracking changed inputs.
//
//   dla_stale_probe <model.dla>
//
// Fills the input with 1.0, runs, records the output; refills the SAME buffers
// with 2.0, runs again, and compares. The two outputs MUST differ for a model
// that consumes its input (add.dla yields 3.0 then 6.0).

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>

#include "mie/engine.h"

namespace {

void FillInputs(mie::Engine* engine, float value, int fill_byte) {
  for (int i = 0; i < engine->NumInputs(); ++i) {
    const mie::Tensor& tensor = engine->Input(i);
    if (tensor.data == nullptr) continue;
    if (tensor.type == mie::Type::kFloat32) {
      float* values = static_cast<float*>(tensor.data);
      const std::size_t count = tensor.bytes / sizeof(float);
      for (std::size_t j = 0; j < count; ++j) values[j] = value;
    } else {
      std::memset(tensor.data, fill_byte, tensor.bytes);
    }
  }
}

void PrintHead(const mie::Tensor& tensor, std::size_t count) {
  std::printf("  head:");
  const float* values = static_cast<const float*>(tensor.data);
  const std::size_t available = tensor.bytes / sizeof(float);
  for (std::size_t i = 0; i < count && i < available; ++i) {
    std::printf(" %.4f", values[i]);
  }
  std::printf("\n");
}

// FNV-1a over the whole output buffer, same as examples/run_model.cc, so the
// two outputs can be compared byte-exactly without dumping them.
std::uint32_t Checksum(const mie::Tensor& tensor) {
  const unsigned char* bytes = static_cast<const unsigned char*>(tensor.data);
  std::uint32_t hash = 2166136261u;
  for (std::size_t i = 0; i < tensor.bytes; ++i) {
    hash ^= static_cast<std::uint32_t>(bytes[i]);
    hash *= 16777619u;
  }
  return hash;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <model.dla>\n", argv[0]);
    return 1;
  }

  mie::Config config;
  config.model_path = argv[1];
  config.backend = mie::Backend::kMtkDla;

  std::string error;
  std::unique_ptr<mie::Engine> engine = mie::Engine::Create(config, &error);
  if (engine == nullptr) {
    std::fprintf(stderr, "Engine::Create failed: %s\n", error.c_str());
    return 1;
  }

  FillInputs(engine.get(), 1.0f, 1);
  if (!engine->Run()) {
    std::fprintf(stderr, "first Run failed\n");
    return 1;
  }
  const std::uint32_t first = Checksum(engine->Output(0));
  std::printf("run 1 (input 1.0) checksum=%08x\n", first);
  PrintHead(engine->Output(0), 4);

  FillInputs(engine.get(), 2.0f, 2);
  if (!engine->Run()) {
    std::fprintf(stderr, "second Run failed\n");
    return 1;
  }
  const std::uint32_t second = Checksum(engine->Output(0));
  std::printf("run 2 (input 2.0) checksum=%08x\n", second);
  PrintHead(engine->Output(0), 4);

  if (first == second) {
    std::fprintf(stderr,
                 "FAIL: outputs identical across a changed input — the output "
                 "no longer tracks Input(i).data. With the V2 request API this "
                 "must not happen; if it does, the backend or the runtime has "
                 "regressed (see the README's MediaTek DLA section and "
                 "tools/dla_raw_probe.cc for the V1 latch history).\n");
    return 1;
  }
  std::printf("PASS: output tracked the new input bytes\n");
  return 0;
}
