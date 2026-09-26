// Diagnostic that drives the Neuron Runtime C API DIRECTLY — no mie in the
// middle — to pin down exactly when the runtime captures the contents of a
// HOST input buffer.
//
// CONCLUSION (libneuron_runtime 7.3.15, Android 16), in two parts:
//
// V1 (NeuronRuntime_*): the input is latched at the FIRST inference of a
// runtime instance. Every approach probed below failed to revive it:
//
//   1-4  V1 setInput re-bind (same pointer / other pointer / none) — latched
//   9-11 EnqueueTrigger                                             — not
//        supported by the model (NEURONRUNTIME_INCOMPLETE)
//   12-14 dma_heap fd buffer (system / system-uncached / mtk_mm-uncached)
//        — unreachable: every node refuses CPU mmap (EACCES), so the caller
//        cannot fill the buffer in the first place
//   15-17 AHardwareBuffer BLOB — unreachable: allocation fails (EINVAL)
//   18-20 NeuronRuntime_clone — a clone re-arms its OWN first inference but
//        latches afterwards too; ~3.8 ms per clone on a 4 KB model, ~7.1 ms
//        on a 1.7 MB one (scales with the .dla, clone re-reads the file)
//   21-22 NeuronAdapter restoreFromCompiledNetwork(+V2) — NEURON_BAD_DATA(4)
//        on a raw ncc-tflite .dla: the Adapter's restore only accepts its own
//        storeCompiledNetwork blobs (format pairing confirmed by the public
//        mdla-cnn-engine project on a MediaTek part; the two loaders are NOT
//        interchangeable in either direction)
//   23-24 create_with_options("suppress-input"...) — accepted but no effect
//
// V2 (NeuronRuntimeV2_*): the request-based API IS honored, with one quirk
// this probe's own steps 7-8 tripped over — presenting a BRAND-NEW IOBuffer
// descriptor (and data pointer) on each request went stale, while the stable
// pattern the engine uses (same descriptor array, same staging pointer,
// rewritten CONTENTS) re-reads the buffer on every run. That stable-pointer
// pattern is what tools/dla_stale_probe.cc verifies end-to-end through the
// engine: input 1.0 → 2.0 flips the output on add.dla and the 384² detector.
// A fresh instance (create + load) reads its input correctly either way,
// which is what neuronrt does per process.
//
//   dla_raw_probe <model.dla> [runtime.so]
//
// <runtime.so> defaults to libneuron_runtime.so (the device's vendor copy);
// pass the SDK's build to match a specific compiler version.

#include <cerrno>
#include <cstdint>
#include <ctime>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#include "neuron/api/RuntimeAPI.h"
#include "neuron/api/RuntimeV2.h"
#include "neuron/api/NeuronAdapter.h"

#include "src/dl.h"

using mie::internal::DlClose;
using mie::internal::DlOpen;
using mie::internal::DlSym;
using mie::internal::LibHandle;

namespace {

void PrintHead(const char* label, const std::vector<char>& out) {
  const float* values = reinterpret_cast<const float*>(out.data());
  const std::size_t count = out.size() / sizeof(float) < 4
                                ? out.size() / sizeof(float)
                                : 4;
  std::printf("%s head:", label);
  for (std::size_t i = 0; i < count; ++i) std::printf(" %.4f", values[i]);
  std::uint32_t hash = 2166136261u;
  const unsigned char* bytes = reinterpret_cast<const unsigned char*>(out.data());
  for (std::size_t i = 0; i < out.size(); ++i) {
    hash ^= static_cast<std::uint32_t>(bytes[i]);
    hash *= 16777619u;
  }
  std::printf("  checksum=%08x\n", hash);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <model.dla> [runtime.so]\n", argv[0]);
    return 1;
  }
  const char* library = argc > 2 ? argv[2] : "libneuron_runtime.so";

  LibHandle lib = DlOpen(library);
  if (lib == nullptr) {
    std::fprintf(stderr, "failed to dlopen %s\n", library);
    return 1;
  }

  int (*create)(const EnvOptions*, void**);
  int (*load_file)(void*, const char*);
  int (*input_size)(void*, uint64_t, std::size_t*);
  int (*output_size)(void*, uint64_t, std::size_t*);
  int (*set_input)(void*, uint64_t, const void*, std::size_t, BufferAttribute);
  int (*set_output)(void*, uint64_t, void*, std::size_t, BufferAttribute);
  int (*inference)(void*);
  void (*release)(void*);
  create = reinterpret_cast<int (*)(const EnvOptions*, void**)>(
      DlSym(lib, "NeuronRuntime_create"));
  load_file = reinterpret_cast<int (*)(void*, const char*)>(
      DlSym(lib, "NeuronRuntime_loadNetworkFromFile"));
  input_size = reinterpret_cast<int (*)(void*, uint64_t, std::size_t*)>(
      DlSym(lib, "NeuronRuntime_getInputSize"));
  output_size = reinterpret_cast<int (*)(void*, uint64_t, std::size_t*)>(
      DlSym(lib, "NeuronRuntime_getOutputSize"));
  set_input = reinterpret_cast<int (*)(void*, uint64_t, const void*, std::size_t,
                                       BufferAttribute)>(
      DlSym(lib, "NeuronRuntime_setInput"));
  set_output = reinterpret_cast<int (*)(void*, uint64_t, void*, std::size_t,
                                        BufferAttribute)>(
      DlSym(lib, "NeuronRuntime_setOutput"));
  inference = reinterpret_cast<int (*)(void*)>(
      DlSym(lib, "NeuronRuntime_inference"));
  release = reinterpret_cast<void (*)(void*)>(
      DlSym(lib, "NeuronRuntime_release"));
  if (create == nullptr || load_file == nullptr || input_size == nullptr ||
      output_size == nullptr || set_input == nullptr ||
      set_output == nullptr || inference == nullptr || release == nullptr) {
    std::fprintf(stderr, "missing NeuronRuntime_* symbol\n");
    return 1;
  }

  void* rt = nullptr;
  EnvOptions env;
  std::memset(&env, 0, sizeof(env));
  env.deviceKind = kEnvOptHardware;
  int status = create(&env, &rt);
  if (status != NEURONRUNTIME_NO_ERROR || rt == nullptr) {
    std::fprintf(stderr, "create failed: %d\n", status);
    return 1;
  }
  status = load_file(rt, argv[1]);
  if (status != NEURONRUNTIME_NO_ERROR) {
    std::fprintf(stderr, "load failed: %d\n", status);
    return 1;
  }

  std::size_t in_bytes = 0;
  std::size_t out_bytes = 0;
  input_size(rt, 0, &in_bytes);
  output_size(rt, 0, &out_bytes);
  std::printf("in=%zu B out=%zu B\n", in_bytes, out_bytes);

  std::vector<char> a(in_bytes, 0);
  std::vector<char> b(in_bytes, 0);
  std::vector<char> out(out_bytes, 0);
  float* fa = reinterpret_cast<float*>(a.data());
  float* fb = reinterpret_cast<float*>(b.data());
  for (std::size_t i = 0; i < in_bytes / sizeof(float); ++i) {
    fa[i] = 1.0f;
    fb[i] = 2.0f;
  }

  BufferAttribute attr;
  attr.ionFd = NON_ION_FD;

  // 1. baseline
  status = set_input(rt, 0, fa, in_bytes, attr);
  status |= set_output(rt, 0, out.data(), out_bytes, attr);
  status |= inference(rt);
  std::printf("1. set(A=1.0)            infer  (st=%d)", status);
  PrintHead("", out);

  // 2. same pointer, contents rewritten to 2.0, re-set
  std::memcpy(a.data(), b.data(), in_bytes);
  status = set_input(rt, 0, fa, in_bytes, attr);
  status |= inference(rt);
  std::printf("2. set(A=2.0) same ptr   infer  (st=%d)", status);
  PrintHead("", out);

  // 3. different pointer
  status = set_input(rt, 0, fb, in_bytes, attr);
  status |= inference(rt);
  std::printf("3. set(B=2.0) other ptr  infer  (st=%d)", status);
  PrintHead("", out);

  // 4. no re-set at all
  status = inference(rt);
  std::printf("4. no set                infer  (st=%d)", status);
  PrintHead("", out);

  release(rt);
  DlClose(lib);

  // 7/8. V2 request-based API: buffers travel WITH each request, so a fresh
  // descriptor per run is the contract. Alternate B (2.0) and A (1.0).
  int (*v2_create)(const char*, size_t, void**, size_t);
  int (*v2_run)(void*, SyncInferenceRequest);
  void (*v2_release)(void*);
  v2_create = reinterpret_cast<int (*)(const char*, size_t, void**, size_t)>(
      DlSym(lib, "NeuronRuntimeV2_create"));
  v2_run = reinterpret_cast<int (*)(void*, SyncInferenceRequest)>(
      DlSym(lib, "NeuronRuntimeV2_run"));
  v2_release =
      reinterpret_cast<void (*)(void*)>(DlSym(lib, "NeuronRuntimeV2_release"));
  if (v2_create == nullptr || v2_run == nullptr || v2_release == nullptr) {
    std::fprintf(stderr, "missing NeuronRuntimeV2_* symbol\n");
    return 1;
  }

  void* rt2 = nullptr;
  status = v2_create(argv[1], 1, &rt2, 2048);
  if (status != NEURONRUNTIME_NO_ERROR || rt2 == nullptr) {
    std::fprintf(stderr, "V2 create failed: %d\n", status);
    return 1;
  }

  IOBuffer in_b{fb, in_bytes, NON_ION_FD};
  IOBuffer out_b{out.data(), out_bytes, NON_ION_FD};
  SyncInferenceRequest req{&in_b, &out_b};
  status = v2_run(rt2, req);
  std::printf("7. V2 req B=2.0         run    (st=%d)", status);
  PrintHead("", out);

  IOBuffer in_a{fa, in_bytes, NON_ION_FD};
  req.inputs = &in_a;
  status = v2_run(rt2, req);
  std::printf("8. V2 req A=1.0         run    (st=%d)", status);
  PrintHead("", out);

  v2_release(rt2);
  DlClose(lib);

  // 9/10/11. EnqueueTrigger: the header says "once the inference settings are
  // changed, user should enqueue the job one more time" — enqueue is where the
  // pre-execution tasks (input conversion/upload) run. If enqueue per run
  // re-reads the host buffer, this is the V1 fix.
  int (*is_et_supported)(void*, uint8_t*);
  int (*enqueue)(void*);
  int (*trigger)(void*);
  is_et_supported = reinterpret_cast<int (*)(void*, uint8_t*)>(
      DlSym(lib, "NeuronRuntime_isEnqueueTriggerSupported"));
  enqueue = reinterpret_cast<int (*)(void*)>(
      DlSym(lib, "NeuronRuntime_inferenceEnqueue"));
  trigger = reinterpret_cast<int (*)(void*)>(
      DlSym(lib, "NeuronRuntime_inferenceTrigger"));
  if (is_et_supported == nullptr || enqueue == nullptr ||
      trigger == nullptr) {
    std::fprintf(stderr, "missing EnqueueTrigger symbol\n");
    return 1;
  }

  LibHandle lib3 = DlOpen(library);
  if (lib3 == nullptr) {
    std::fprintf(stderr, "failed to dlopen(3) %s\n", library);
    return 1;
  }
  int (*create3)(const EnvOptions*, void**);
  int (*load3)(void*, const char*);
  create3 = reinterpret_cast<int (*)(const EnvOptions*, void**)>(
      DlSym(lib3, "NeuronRuntime_create"));
  load3 = reinterpret_cast<int (*)(void*, const char*)>(
      DlSym(lib3, "NeuronRuntime_loadNetworkFromFile"));
  void* rt3 = nullptr;
  std::memset(&env, 0, sizeof(env));
  env.deviceKind = kEnvOptHardware;
  status = create3(&env, &rt3);
  if (status != NEURONRUNTIME_NO_ERROR || rt3 == nullptr) {
    std::fprintf(stderr, "create(3) failed: %d\n", status);
    return 1;
  }
  status = load3(rt3, argv[1]);
  if (status != NEURONRUNTIME_NO_ERROR) {
    std::fprintf(stderr, "load(3) failed: %d\n", status);
    return 1;
  }

  uint8_t et = 0;
  status = is_et_supported(rt3, &et);
  std::printf("9. enqueueTrigger supported=%d (st=%d)\n", et, status);

  std::memcpy(a.data(), b.data(), in_bytes);  // a = 2.0 now
  status = set_input(rt3, 0, fa, in_bytes, attr);
  status |= set_output(rt3, 0, out.data(), out_bytes, attr);
  status |= enqueue(rt3);
  status |= trigger(rt3);
  std::printf("10. set(A=2.0) enq+trg       (st=%d)", status);
  PrintHead("", out);

  std::memset(a.data(), 0, in_bytes);  // a = 0.0 (float zeros)
  status = set_input(rt3, 0, fa, in_bytes, attr);
  status |= enqueue(rt3);
  status |= trigger(rt3);
  std::printf("11. set(A=0.0) enq+trg       (st=%d)", status);
  PrintHead("", out);

  release(rt3);
  DlClose(lib3);

  // 12/13/14. dma-buf fd buffer: with an imported fd the runtime maps the
  // shared memory instead of copying, so the CURRENT contents must be read at
  // every inference. The dma-heap UAPI is defined by hand: r16b's sysroot
  // predates linux/dma-heap.h. Kept skippable: every heap on this MTK kernel
  // refuses CPU mmap (EACCES) and must not block the later sections.
  auto TestDmabuf = [&]() {
  struct DmaHeapAllocData {
    std::uint64_t len;
    std::uint32_t fd;
    std::uint32_t fd_flags;
    std::uint64_t heap_flags;
  };
  const unsigned long kDmaHeapIocMagic = 'H';
  const unsigned long kDmaHeapIocAlloc =
      _IOWR(kDmaHeapIocMagic, 0x0, DmaHeapAllocData);

  const char* heaps[] = {"/dev/dma_heap/mtk_mm-uncached",
                         "/dev/dma_heap/system-uncached",
                         "/dev/dma_heap/system"};
  int heap_fd = -1;
  int alloc_fd = -1;
  for (const char* path : heaps) {
    int hf = open(path, O_RDWR);
    if (hf < 0) {
      std::printf("dma heap %s: errno=%d\n", path, errno);
      continue;
    }
    DmaHeapAllocData ad = {};
    ad.len = in_bytes;
    if (ioctl(hf, kDmaHeapIocAlloc, &ad) != 0) {
      std::printf("dma heap %s: alloc errno=%d\n", path, errno);
      close(hf);
      continue;
    }
    void* m = mmap(nullptr, in_bytes, PROT_READ | PROT_WRITE, MAP_SHARED,
                   ad.fd, 0);
    if (m == MAP_FAILED) {
      std::printf("dma heap %s: mmap errno=%d\n", path, errno);
      close(ad.fd);
      close(hf);
      continue;
    }
    munmap(m, in_bytes);
    std::printf("dma heap: %s (mmap OK)\n", path);
    heap_fd = hf;
    alloc_fd = ad.fd;
    break;
  }
  if (heap_fd < 0) {
    std::fprintf(stderr, "no dma heap with a CPU-mappable allocation\n");
    return;
  }
  DmaHeapAllocData alloc = {};
  alloc.fd = alloc_fd;
  // Cached-heap CPU writes must be bracketed with DMA_BUF_IOCTL_SYNC to be
  // coherent for the device. Uncached heaps ignore/need-not-use it, so failures
  // are non-fatal.
  struct DmaBufSync {
    std::uint64_t flags;
  };
  const unsigned long kDmaBufBase = 'b';
  const unsigned long kDmaBufIocSync =
      _IOW(kDmaBufBase, 0, DmaBufSync);
  const std::uint64_t kSyncRw = 3;      // READ | WRITE
  const std::uint64_t kSyncStart = 0;   // 0 << 2
  const std::uint64_t kSyncEnd = 4;     // 1 << 2
  auto SyncDmaBuf = [&](std::uint64_t phase) {
    DmaBufSync s{phase | kSyncRw};
    ioctl(alloc.fd, kDmaBufIocSync, &s);
  };

  void* map =
      mmap(nullptr, in_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, alloc.fd, 0);
  if (map == MAP_FAILED) {
    std::printf("mmap dmabuf failed: errno=%d (skipping 12-14)\n", errno);
    map = nullptr;
  }
  if (map != nullptr) {
  float* fm = reinterpret_cast<float*>(map);
  auto FillDmaBuf = [&](float value) {
    SyncDmaBuf(kSyncStart);
    for (std::size_t i = 0; i < in_bytes / sizeof(float); ++i) fm[i] = value;
    SyncDmaBuf(kSyncEnd);
  };
  FillDmaBuf(2.0f);

  BufferAttribute ion_attr;
  ion_attr.ionFd = alloc.fd;

  LibHandle lib4 = DlOpen(library);
  if (lib4 == nullptr) {
    std::fprintf(stderr, "failed to dlopen(4) %s\n", library);
    return;
  }
  int (*create4)(const EnvOptions*, void**);
  int (*load4)(void*, const char*);
  create4 = reinterpret_cast<int (*)(const EnvOptions*, void**)>(
      DlSym(lib4, "NeuronRuntime_create"));
  load4 = reinterpret_cast<int (*)(void*, const char*)>(
      DlSym(lib4, "NeuronRuntime_loadNetworkFromFile"));
  void* rt4 = nullptr;
  std::memset(&env, 0, sizeof(env));
  env.deviceKind = kEnvOptHardware;
  status = create4(&env, &rt4);
  if (status != NEURONRUNTIME_NO_ERROR || rt4 == nullptr) {
    std::fprintf(stderr, "create(4) failed: %d\n", status);
    return;
  }
  status = load4(rt4, argv[1]);
  if (status != NEURONRUNTIME_NO_ERROR) {
    std::fprintf(stderr, "load(4) failed: %d\n", status);
    return;
  }

  status = set_input(rt4, 0, map, in_bytes, ion_attr);
  status |= set_output(rt4, 0, out.data(), out_bytes, attr);
  status |= inference(rt4);
  std::printf("12. fd buffer 2.0       infer  (st=%d)", status);
  PrintHead("", out);

  FillDmaBuf(1.0f);
  status = inference(rt4);
  std::printf("13. fd rewritten 1.0    infer  (st=%d)", status);
  PrintHead("", out);

  FillDmaBuf(3.0f);
  status = inference(rt4);
  std::printf("14. fd rewritten 3.0    infer  (st=%d)", status);
  PrintHead("", out);

  release(rt4);
  DlClose(lib4);
  munmap(map, in_bytes);
  }
  close(alloc.fd);
  close(heap_fd);
  };
  TestDmabuf();

  // 15/16/17. AHardwareBuffer(BLOB) fd: the dma_heap nodes refuse CPU mmap on
  // this MTK kernel, but gralloc-backed BLOB buffers are CPU-mappable by
  // construction and their native handle carries an ion/dmabuf fd the runtime
  // can import. AHardwareBuffer_* are resolved dynamically: r16b predates the
  // libnativewindow link surface. Kept skippable: this MTK gralloc rejects
  // BLOB allocation (EINVAL) and must not block the clone tests.
  auto TestAhwb = [&]() {
  typedef struct {
    std::uint64_t width;
    std::uint64_t height;
    std::uint32_t layers;
    std::uint32_t format;
    std::uint64_t usage;
    std::uint32_t stride;
    std::uint32_t rsvd0;
  } AhbDesc;
  struct NativeHandle {
    int version;
    int num_fds;
    int num_ints;
    int data[1];
  };
  typedef void Ahb;
  const std::uint32_t kBlobFormat = 0x5247494E;  // AHARDWAREBUFFER_FORMAT_BLOB
  const std::uint64_t kCpuReadOften = 3ULL << 4;
  const std::uint64_t kCpuWriteOften = 3ULL << 6;
  const std::uint64_t kGpuDataBuffer = 1ULL << 24;
  const int kCpuWriteOftenInt = static_cast<int>(kCpuWriteOften);

  LibHandle nw = DlOpen("libnativewindow.so");
  if (nw == nullptr) {
    std::fprintf(stderr, "failed to dlopen libnativewindow.so\n");
    return;
  }
  int (*ahb_allocate)(const AhbDesc*, Ahb**);
  int (*ahb_lock)(Ahb*, std::uint64_t, int, const void*, void**);
  int (*ahb_unlock)(Ahb*, std::int32_t*);
  const NativeHandle* (*ahb_get_handle)(const Ahb*);
  void (*ahb_release)(Ahb*);
  ahb_allocate = reinterpret_cast<int (*)(const AhbDesc*, Ahb**)>(
      DlSym(nw, "AHardwareBuffer_allocate"));
  ahb_lock = reinterpret_cast<int (*)(Ahb*, std::uint64_t, int, const void*,
                                      void**)>(DlSym(nw, "AHardwareBuffer_lock"));
  ahb_unlock = reinterpret_cast<int (*)(Ahb*, std::int32_t*)>(
      DlSym(nw, "AHardwareBuffer_unlock"));
  ahb_get_handle = reinterpret_cast<const NativeHandle* (*)(const Ahb*)>(
      DlSym(nw, "AHardwareBuffer_getNativeHandle"));
  ahb_release = reinterpret_cast<void (*)(Ahb*)>(
      DlSym(nw, "AHardwareBuffer_release"));
  if (ahb_allocate == nullptr || ahb_lock == nullptr || ahb_unlock == nullptr ||
      ahb_get_handle == nullptr || ahb_release == nullptr) {
    std::fprintf(stderr, "missing AHardwareBuffer_* symbol\n");
    return;
  }

  AhbDesc ahb_desc = {};
  ahb_desc.width = (in_bytes + 4095) & ~4095ULL;  // BLOB: width is bytes; gralloc wants alignment
  ahb_desc.height = 1;
  ahb_desc.layers = 1;
  ahb_desc.format = kBlobFormat;
  ahb_desc.usage = kCpuReadOften | kCpuWriteOften | kGpuDataBuffer;
  Ahb* ahb = nullptr;
  status = ahb_allocate(&ahb_desc, &ahb);
  if (status != 0 || ahb == nullptr) {
    std::fprintf(stderr, "AHardwareBuffer_allocate failed: %d\n", status);
    return;
  }
  void* ahb_map = nullptr;
  status = ahb_lock(ahb, kCpuWriteOften, -1, nullptr, &ahb_map);
  if (status != 0 || ahb_map == nullptr) {
    std::fprintf(stderr, "AHardwareBuffer_lock failed: %d\n", status);
    return;
  }
  const NativeHandle* nh = ahb_get_handle(ahb);
  if (nh == nullptr || nh->num_fds < 1) {
    std::fprintf(stderr, "AHardwareBuffer_getNativeHandle failed\n");
    return;
  }
  const int ahb_fd = nh->data[0];
  std::printf("ahwb blob fd=%d\n", ahb_fd);

  BufferAttribute ahwb_attr;
  ahwb_attr.ionFd = ahb_fd;
  float* fahb = reinterpret_cast<float*>(ahb_map);

  LibHandle lib5 = DlOpen(library);
  if (lib5 == nullptr) {
    std::fprintf(stderr, "failed to dlopen(5) %s\n", library);
    return;
  }
  int (*create5)(const EnvOptions*, void**);
  int (*load5)(void*, const char*);
  create5 = reinterpret_cast<int (*)(const EnvOptions*, void**)>(
      DlSym(lib5, "NeuronRuntime_create"));
  load5 = reinterpret_cast<int (*)(void*, const char*)>(
      DlSym(lib5, "NeuronRuntime_loadNetworkFromFile"));
  void* rt5 = nullptr;
  std::memset(&env, 0, sizeof(env));
  env.deviceKind = kEnvOptHardware;
  status = create5(&env, &rt5);
  if (status != NEURONRUNTIME_NO_ERROR || rt5 == nullptr) {
    std::fprintf(stderr, "create(5) failed: %d\n", status);
    return;
  }
  status = load5(rt5, argv[1]);
  if (status != NEURONRUNTIME_NO_ERROR) {
    std::fprintf(stderr, "load(5) failed: %d\n", status);
    return;
  }

  for (std::size_t i = 0; i < in_bytes / sizeof(float); ++i) fahb[i] = 2.0f;
  status = set_input(rt5, 0, ahb_map, in_bytes, ahwb_attr);
  status |= set_output(rt5, 0, out.data(), out_bytes, attr);
  status |= inference(rt5);
  std::printf("15. ahwb 2.0            infer  (st=%d)", status);
  PrintHead("", out);

  for (std::size_t i = 0; i < in_bytes / sizeof(float); ++i) fahb[i] = 1.0f;
  status = inference(rt5);
  std::printf("16. ahwb rewritten 1.0  infer  (st=%d)", status);
  PrintHead("", out);

  for (std::size_t i = 0; i < in_bytes / sizeof(float); ++i) fahb[i] = 3.0f;
  status = inference(rt5);
  std::printf("17. ahwb rewritten 3.0  infer  (st=%d)", status);
  PrintHead("", out);

  release(rt5);
  DlClose(lib5);
  ahb_unlock(ahb, nullptr);
  ahb_release(ahb);
  DlClose(nw);
  };
  TestAhwb();

  // 18/19/20. Clone: the doc says constant (weight) data is shared, so a clone
  // per run is much cheaper than create+load. Question: does a clone re-arm the
  // input stage (fresh bindings) or inherit the parent's latched input?
  // Parent here has NOT inferred yet — a fresh loaded runtime is the doc's
  // requirement ("existing Runtime must be DLA-loaded").
  int (*clone_rt)(void*, void**);
  clone_rt = reinterpret_cast<int (*)(void*, void**)>(
      DlSym(lib, "NeuronRuntime_clone"));
  if (clone_rt == nullptr) {
    std::fprintf(stderr, "missing NeuronRuntime_clone symbol\n");
    return 1;
  }

  LibHandle lib6 = DlOpen(library);
  if (lib6 == nullptr) {
    std::fprintf(stderr, "failed to dlopen(6) %s\n", library);
    return 1;
  }
  int (*create6)(const EnvOptions*, void**);
  int (*load6)(void*, const char*);
  create6 = reinterpret_cast<int (*)(const EnvOptions*, void**)>(
      DlSym(lib6, "NeuronRuntime_create"));
  load6 = reinterpret_cast<int (*)(void*, const char*)>(
      DlSym(lib6, "NeuronRuntime_loadNetworkFromFile"));
  void* rt6 = nullptr;
  std::memset(&env, 0, sizeof(env));
  env.deviceKind = kEnvOptHardware;
  status = create6(&env, &rt6);
  if (status != NEURONRUNTIME_NO_ERROR || rt6 == nullptr) {
    std::fprintf(stderr, "create(6) failed: %d\n", status);
    return 1;
  }
  status = load6(rt6, argv[1]);
  if (status != NEURONRUNTIME_NO_ERROR) {
    std::fprintf(stderr, "load(6) failed: %d\n", status);
    return 1;
  }

  void* rtc = nullptr;
  status = clone_rt(rt6, &rtc);
  if (status != NEURONRUNTIME_NO_ERROR || rtc == nullptr) {
    std::fprintf(stderr, "clone failed: %d\n", status);
    return 1;
  }

  for (std::size_t i = 0; i < in_bytes / sizeof(float); ++i) fb[i] = 2.0f;
  status = set_input(rtc, 0, fb, in_bytes, attr);
  status |= set_output(rtc, 0, out.data(), out_bytes, attr);
  status |= inference(rtc);
  std::printf("18. clone set(B=2.0)     infer  (st=%d)", status);
  PrintHead("", out);

  status = set_input(rtc, 0, fa, in_bytes, attr);  // fa holds 1.0
  status |= inference(rtc);
  std::printf("19. clone re-set(A=1.0)  infer  (st=%d)", status);
  PrintHead("", out);

  // 20. clone cost: the parent was loaded once; time 100 clones.
  const int kClones = 100;
  struct timespec ts0, ts1;
  clock_gettime(CLOCK_MONOTONIC, &ts0);
  int clone_ok = 0;
  for (int i = 0; i < kClones; ++i) {
    void* c = nullptr;
    if (clone_rt(rt6, &c) == NEURONRUNTIME_NO_ERROR && c != nullptr) {
      ++clone_ok;
      release(c);
    }
  }
  clock_gettime(CLOCK_MONOTONIC, &ts1);
  const double clone_us =
      ((ts1.tv_sec - ts0.tv_sec) * 1e9 + (ts1.tv_nsec - ts0.tv_nsec)) / 1e3 /
      kClones;
  std::printf("20. %d clones ok=%d, avg %.1f us per clone\n", kClones,
              clone_ok, clone_us);

  release(rtc);
  release(rt6);
  DlClose(lib6);

  // 21/22. NeuronAdapter execution API (libneuron_adapter.so): NNAPI-style
  // setInput + compute. The kMtk delegate path sits on this API, and real
  // applications feed different frames per compute, so per-compute buffers
  // should be honored by contract. restoreFromCompiledNetwork takes the .dla
  // as a buffer and hands back a ready compilation. Skippable: the restore
  // rejects the raw .dla on this build (BAD_DATA) and must not block 23/24.
  auto TestAdapter = [&]() {
  LibHandle ad = DlOpen("libneuron_adapter.so");
  if (ad == nullptr) ad = DlOpen("libneuron_adapter.7.so");
  if (ad == nullptr) ad = DlOpen("libneuron_adapter.so.7.3.15");
  if (ad == nullptr) {
    std::fprintf(stderr, "failed to dlopen libneuron_adapter\n");
    return;
  }
  int (*restore)(NeuronModel**, NeuronCompilation**, const void*, const size_t);
  int (*comp_finish)(NeuronCompilation*);
  int (*exec_create)(NeuronCompilation*, NeuronExecution**);
  int (*exec_set_in)(NeuronExecution*, int32_t, const NeuronOperandType*,
                     const void*, size_t);
  int (*exec_set_out)(NeuronExecution*, int32_t, const NeuronOperandType*,
                      void*, size_t);
  int (*exec_compute)(NeuronExecution*);
  restore = reinterpret_cast<int (*)(NeuronModel**, NeuronCompilation**,
                                     const void*, const size_t)>(
      DlSym(ad, "NeuronModel_restoreFromCompiledNetwork"));
  comp_finish = reinterpret_cast<int (*)(NeuronCompilation*)>(
      DlSym(ad, "NeuronCompilation_finish"));
  exec_create = reinterpret_cast<int (*)(NeuronCompilation*, NeuronExecution**)>(
      DlSym(ad, "NeuronExecution_create"));
  exec_set_in = reinterpret_cast<int (*)(NeuronExecution*, int32_t,
                                         const NeuronOperandType*, const void*,
                                         size_t)>(
      DlSym(ad, "NeuronExecution_setInput"));
  exec_set_out = reinterpret_cast<int (*)(NeuronExecution*, int32_t,
                                          const NeuronOperandType*, void*,
                                          size_t)>(
      DlSym(ad, "NeuronExecution_setOutput"));
  exec_compute = reinterpret_cast<int (*)(NeuronExecution*)>(
      DlSym(ad, "NeuronExecution_compute"));
  if (restore == nullptr || comp_finish == nullptr || exec_create == nullptr ||
      exec_set_in == nullptr || exec_set_out == nullptr ||
      exec_compute == nullptr) {
    std::fprintf(stderr, "missing NeuronAdapter symbol\n");
    return;
  }

  std::FILE* f = std::fopen(argv[1], "rb");
  if (f == nullptr) {
    std::fprintf(stderr, "failed to open %s\n", argv[1]);
    return;
  }
  std::fseek(f, 0, SEEK_END);
  const long dla_size = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  std::vector<char> dla_data(dla_size);
  const size_t got = std::fread(dla_data.data(), 1, dla_data.size(), f);
  std::fclose(f);
  if (got != dla_data.size()) {
    std::fprintf(stderr, "short read of the .dla\n");
    return;
  }

  NeuronModel* amodel = nullptr;
  NeuronCompilation* acomp = nullptr;
  status = restore(&amodel, &acomp, dla_data.data(), dla_data.size());
  if (status != NEURON_NO_ERROR) {
    // V2 variant with an explicit compilation type before giving up.
    int (*restore2)(NeuronModel**, NeuronCompilation**, const void*,
                    const size_t, const int&);
    restore2 = reinterpret_cast<int (*)(NeuronModel**, NeuronCompilation**,
                                        const void*, const size_t, const int&)>(
        DlSym(ad, "NeuronModel_restoreFromCompiledNetworkV2"));
    if (restore2 != nullptr) {
      const int kNormal = 0;  // COMPILATION_TYPE_NORMAL
      status = restore2(&amodel, &acomp, dla_data.data(), dla_data.size(),
                        kNormal);
    }
    if (status != NEURON_NO_ERROR) {
      std::fprintf(stderr, "restoreFromCompiledNetwork failed: %d\n", status);
      return;
    }
  }
  status = comp_finish(acomp);
  if (status != NEURON_NO_ERROR) {
    std::fprintf(stderr, "NeuronCompilation_finish failed: %d\n", status);
    return;
  }
  NeuronExecution* aexec = nullptr;
  status = exec_create(acomp, &aexec);
  if (status != NEURON_NO_ERROR || aexec == nullptr) {
    std::fprintf(stderr, "NeuronExecution_create failed: %d\n", status);
    return;
  }

  status = exec_set_in(aexec, 0, nullptr, fb, in_bytes);  // 2.0
  status |= exec_set_out(aexec, 0, nullptr, out.data(), out_bytes);
  status |= exec_compute(aexec);
  std::printf("21. adapter set(B=2.0)  compute (st=%d)", status);
  PrintHead("", out);

  status = exec_set_in(aexec, 0, nullptr, fa, in_bytes);  // 1.0
  status |= exec_compute(aexec);
  std::printf("22. adapter set(A=1.0)  compute (st=%d)", status);
  PrintHead("", out);
  };
  TestAdapter();

  // 23/24. create_with_options("suppress-input"): the runtime .so carries
  // 'suppress-input'/'suppress-output' option strings. If suppressing the
  // input conversion removes the conversion cache, host buffers may finally
  // be read live across inferences.
  int (*create_opt)(const char*, const EnvOptions*, void**);
  create_opt = reinterpret_cast<int (*)(const char*, const EnvOptions*, void**)>(
      DlSym(lib, "NeuronRuntime_create_with_options"));
  if (create_opt == nullptr) {
    std::fprintf(stderr, "missing NeuronRuntime_create_with_options\n");
    return 1;
  }
  const char* opt_variants[] = {"suppress-input", "suppress-input=1",
                                "suppress-input=true"};
  for (int v = 0; v < 3; ++v) {
    LibHandle lib7 = DlOpen(library);
    if (lib7 == nullptr) continue;
    int (*create7)(const EnvOptions*, void**);
    int (*load7)(void*, const char*);
    create7 = reinterpret_cast<int (*)(const EnvOptions*, void**)>(
        DlSym(lib7, "NeuronRuntime_create"));
    load7 = reinterpret_cast<int (*)(void*, const char*)>(
        DlSym(lib7, "NeuronRuntime_loadNetworkFromFile"));
    void* rt7 = nullptr;
    std::memset(&env, 0, sizeof(env));
    env.deviceKind = kEnvOptHardware;
    status = create_opt(opt_variants[v], &env, &rt7);
    if (status != NEURONRUNTIME_NO_ERROR || rt7 == nullptr) {
      std::printf("23. options=\"%s\" create failed: %d\n", opt_variants[v],
                  status);
      DlClose(lib7);
      continue;
    }
    if (load7(rt7, argv[1]) != NEURONRUNTIME_NO_ERROR) {
      std::printf("23. options=\"%s\" load failed\n", opt_variants[v]);
      release(rt7);
      DlClose(lib7);
      continue;
    }
    for (std::size_t i = 0; i < in_bytes / sizeof(float); ++i) fb[i] = 2.0f;
    status = set_input(rt7, 0, fb, in_bytes, attr);
    status |= set_output(rt7, 0, out.data(), out_bytes, attr);
    status |= inference(rt7);
    std::printf("23. options=\"%s\" set(B=2.0) infer (st=%d)", opt_variants[v],
                status);
    PrintHead("", out);
    for (std::size_t i = 0; i < in_bytes / sizeof(float); ++i) fa[i] = 1.0f;
    status = set_input(rt7, 0, fa, in_bytes, attr);
    status |= inference(rt7);
    std::printf("24. options=\"%s\" set(A=1.0) infer (st=%d)", opt_variants[v],
                status);
    PrintHead("", out);
    release(rt7);
    DlClose(lib7);
    if (true) break;  // first variant that creates cleanly is enough
  }

  return 0;
}
