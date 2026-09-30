// Benchmark driver: build an engine once, then report where the time and the
// memory go — engine init, the cold first run, warm-up, steady-state latency
// (avg/min/max/p50/p90/p99/stddev) and RSS deltas across each phase.
//
// Functionally the sibling of mie_run (same backends, same MIE_OPTIONS keys),
// but arguments are parsed with the single-header cmdline library:
//
//   mie_bench <model> [-b backend] [-n runs] [-t threads]
//                     [-c cache_dir] [-o key=value,key=value] [-?|--help]
//
// Memory stats read /proc/self/status (Android/Linux) or the process working
// set (Windows); on platforms with neither they stay zero and are not printed.
//
// C++11 on purpose — the Android build targets NDK r16b (clang 5.0).

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "mie/engine.h"

#include "common.h"

#include "cmdline.h"

#if defined(_WIN32)
#define NOMINMAX
#define PSAPI_VERSION 2  // K32* kernel32 exports; no psapi.lib needed
#include <windows.h>
#include <psapi.h>
#endif

namespace {

// Resident memory and its high-water mark, in kB.
struct MemUsage {
  long rss_kb;
  long peak_kb;
};

MemUsage GetMemUsage() {
  MemUsage usage = {0, 0};
#if defined(__ANDROID__) || defined(__linux__)
  std::FILE* file = std::fopen("/proc/self/status", "r");
  if (file == nullptr) return usage;
  char line[256];
  while (std::fgets(line, sizeof(line), file) != nullptr) {
    if (std::strncmp(line, "VmRSS:", 6) == 0) {
      std::sscanf(line + 6, "%ld", &usage.rss_kb);
    } else if (std::strncmp(line, "VmHWM:", 6) == 0) {
      std::sscanf(line + 6, "%ld", &usage.peak_kb);
    }
  }
  std::fclose(file);
#elif defined(_WIN32)
  PROCESS_MEMORY_COUNTERS counters;
  std::memset(&counters, 0, sizeof(counters));
  counters.cb = sizeof(counters);
  if (GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof(counters))) {
    usage.rss_kb = static_cast<long>(counters.WorkingSetSize / 1024);
    usage.peak_kb = static_cast<long>(counters.PeakWorkingSetSize / 1024);
  }
#endif
  return usage;
}

bool MemSupported() {
#if defined(__ANDROID__) || defined(__linux__) || defined(_WIN32)
  return true;
#else
  return false;
#endif
}

double ToMs(std::chrono::steady_clock::duration d) {
  return std::chrono::duration<double, std::milli>(d).count();
}

double Mb(long kb) { return static_cast<double>(kb) / 1024.0; }

// Nearest-rank percentile over a sorted vector.
double Percentile(const std::vector<double>& sorted_ms, double fraction) {
  if (sorted_ms.empty()) return 0.0;
  const std::size_t rank = static_cast<std::size_t>(
      std::ceil(fraction * static_cast<double>(sorted_ms.size())));
  const std::size_t index = rank == 0 ? 0 : rank - 1;
  return sorted_ms[std::min(index, sorted_ms.size() - 1)];
}

double StdDev(const std::vector<double>& values, double mean) {
  if (values.size() < 2) return 0.0;
  double sum = 0.0;
  for (std::size_t i = 0; i < values.size(); ++i) {
    const double d = values[i] - mean;
    sum += d * d;
  }
  return std::sqrt(sum / static_cast<double>(values.size() - 1));
}

void PrintMem(const char* label, const MemUsage& usage) {
  if (!MemSupported()) return;
  std::printf("  %-22s %8.1f MB (peak %.1f MB)\n", label, Mb(usage.rss_kb),
              Mb(usage.peak_kb));
}

void PrintMemSkippedNote() {
  if (!MemSupported()) {
    std::printf("  (memory stats unavailable on this platform)\n");
  }
}

}  // namespace

int main(int argc, char** argv) {
  cmdline::parser parser;
  parser.set_program_name("mie_bench");
  parser.footer("<model>");
  parser.add<std::string>("backend", 'b',
                          "cpu|gpu|qnn|mtk|dla|qnnnative|rknn", false, "cpu");
  parser.add<int>("runs", 'n', "timed steady-state iterations", false, 100,
                  cmdline::range(1, 1000000));
  parser.add<int>("threads", 't',
                  "CPU worker threads (0 = TFLite decides; ignored by "
                  "non-CPU backends)",
                  false, 4, cmdline::range(0, 256));
  parser.add<std::string>("cache_dir", 'c', "backend cache directory", false,
                          "");
  parser.add<std::string>("options", 'o',
                          "backend options, key=value,key=value (--help lists "
                          "every key and what it does)",
                          false, "");
  // Hand-rolled instead of parse_check() so --help can print the full backend
  // option table after the generated usage. parse() does not auto-register a
  // help flag — parse_check() does — so add it here.
  parser.add("help", '?', "print this message");
  if (!parser.parse(argc, argv)) {
    std::fprintf(stderr, "%s\n%s", parser.error().c_str(),
                 parser.usage().c_str());
    return 1;
  }
  if (parser.exist("help")) {
    std::printf("%s\n", parser.usage().c_str());
    example::PrintOptionHelp();
    return 0;
  }
  if (parser.rest().empty()) {
    std::fprintf(stderr, "error: missing <model> argument\n\n%s",
                 parser.usage().c_str());
    return 1;
  }

  const std::string model_path = parser.rest()[0];
  const std::string backend_name = parser.get<std::string>("backend");
  const int runs = parser.get<int>("runs");
  // Fixed warm-up count: enough to leave the cold path (lazy NPU power-up,
  // first-request setup) behind, not so many that a slow backend drags on.
  const int warmup = 3;
  const int threads = parser.get<int>("threads");

  mie::Backend backend;
  std::string error;
  if (!example::ParseBackend(backend_name.c_str(), &backend, &error)) {
    std::fprintf(stderr, "%s\n", error.c_str());
    return 1;
  }
  if (backend != mie::Backend::kCpu && parser.exist("threads")) {
    std::fprintf(stderr,
                 "WARN: [threads] is CPU-only; ignored by this backend\n");
  }

  mie::Config config;
  config.model_path = model_path;
  config.backend = backend;
  config.num_threads = threads;
  config.cache_dir = parser.get<std::string>("cache_dir");
  example::AppendEnvOptions(&config);
  example::ParseOptionsString(parser.get<std::string>("options"), &config);

  std::printf("== mie_bench ==\n");
  std::printf("  model                  %s\n", model_path.c_str());
  std::printf("  backend                %s (threads=%d)\n",
              backend_name.c_str(), threads);
  if (!config.cache_dir.empty()) {
    std::printf("  cache_dir              %s\n", config.cache_dir.c_str());
  }
  if (!config.options.empty()) {
    std::printf("  options");
    for (std::size_t i = 0; i < config.options.size(); ++i) {
      std::printf("%s%s=%s", i == 0 ? "                    " : " ",
                  config.options[i].first.c_str(),
                  config.options[i].second.c_str());
    }
    std::printf("\n");
  }

  typedef std::chrono::steady_clock Clock;
  const MemUsage before = GetMemUsage();
  const Clock::time_point create_begin = Clock::now();
  std::unique_ptr<mie::Engine> engine = mie::Engine::Create(config, &error);
  if (!engine) {
    std::fprintf(stderr, "engine create failed: %s\n", error.c_str());
    return 1;
  }
  const double init_ms = ToMs(Clock::now() - create_begin);
  const MemUsage after_init = GetMemUsage();

  std::printf("  init (Engine::Create)  %8.2f ms\n", init_ms);
  PrintMem("mem before init", before);
  PrintMem("mem after init", after_init);

  for (int i = 0; i < engine->NumInputs(); ++i) {
    example::FillInput(engine->Input(i));
  }

  // Cold start: the first Run() can pay one-time device costs (lazy NPU
  // power-up, GPU shader/kernel compile, DLA first-request setup).
  const Clock::time_point cold_begin = Clock::now();
  engine->Run();
  const double cold_ms = ToMs(Clock::now() - cold_begin);
  std::printf("  first run (cold)       %8.2f ms\n", cold_ms);

  // Warm-up: not timed individually, just observed in aggregate.
  const Clock::time_point warm_begin = Clock::now();
  for (int i = 0; i < warmup; ++i) engine->Run();
  const double warm_total_ms = ToMs(Clock::now() - warm_begin);
  const MemUsage after_warmup = GetMemUsage();
  std::printf("  warmup                 %d runs in %.2f ms (avg %.2f ms)\n",
              warmup, warm_total_ms, warm_total_ms / warmup);
  PrintMem("mem after warmup", after_warmup);

  // Steady state: every Run() timed separately.
  std::vector<double> run_ms;
  run_ms.reserve(static_cast<std::size_t>(runs));
  for (int i = 0; i < runs; ++i) {
    const Clock::time_point begin = Clock::now();
    engine->Run();
    run_ms.push_back(ToMs(Clock::now() - begin));
  }
  std::vector<double> sorted = run_ms;
  std::sort(sorted.begin(), sorted.end());

  double sum = 0.0;
  for (std::size_t i = 0; i < run_ms.size(); ++i) sum += run_ms[i];
  const double avg = sum / static_cast<double>(run_ms.size());

  std::printf("  steady state           %d runs\n", runs);
  std::printf("    avg                  %8.2f ms\n", avg);
  std::printf("    min / max            %8.2f / %.2f ms\n", sorted.front(),
              sorted.back());
  std::printf("    p50 / p90 / p99      %8.2f / %.2f / %.2f ms\n",
              Percentile(sorted, 0.50), Percentile(sorted, 0.90),
              Percentile(sorted, 0.99));
  std::printf("    stddev               %8.2f ms\n", StdDev(run_ms, avg));
  PrintMem("mem after runs", GetMemUsage());
  PrintMemSkippedNote();

  // Sanity: a non-zero input must not produce an all-zero output, and the
  // checksum makes runs comparable across devices and backends.
  for (int i = 0; i < engine->NumOutputs(); ++i) {
    const mie::Tensor& out = engine->Output(i);
    std::printf("  out[%d] bytes=%zu checksum=%08x%s\n", i, out.bytes,
               example::TensorChecksum(out),
               example::IsDegenerate(out) ? "  WARN: all zeros!" : "");
  }
  return 0;
}
