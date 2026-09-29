# mie — minimal multi-backend model inference engine

One abstraction, seven backends, **one implementation file per backend**.

| `Backend` | What runs | How it is reached | Build-time dep |
|-----------|-----------|-------------------|----------------|
| `kCpu`    | TFLite built-in kernels (multi-threaded); XNNPACK via the `xnnpack=1` option | TFLite interpreter, optional linked XNNPACK delegate | TFLite C API |
| `kGpu`    | TFLite GPU delegate (OpenCL / OpenGL) | linked delegate, serialized model cache | GPU delegate lib |
| `kQnn`    | Qualcomm QNN (AI Engine Direct) HTP / NPU | `dlopen` + TFLite external-delegate plugin ABI | none |
| `kMtk`    | MediaTek NeuroPilot NPU, from a `.tflite` | `dlopen` + `NeuroPilotTFLiteShim.h` | shim header only |
| `kMtkDla` | MediaTek NPU, from a precompiled `.dla` | `dlopen` + Neuron Runtime **V2** C API | `neuron/api` headers |
| `kQnnNative` | Qualcomm QNN HTP / NPU, from a precompiled context binary (`.bin`) | `dlopen` + QNN native C API (`QnnInterface`) | QNN SDK headers |
| `kRknn` | Rockchip NPU, from a native `.rknn` model | RKNN Runtime staging-buffer API | ARM64 Linux RKNN SDK |

CPU/GPU/QNN are all "one TFLite interpreter plus an optional delegate", so they
share `src/tflite_interpreter.cc` and differ only in which delegate they build.
MTK is *not* a TFLite delegate at all — it is a separate runtime — so it is a
self-contained `Engine` implementation, and `kMtkDla` is a second one: it loads
a model compiled ahead of time by the NeuroPilot SDK's `ncc-tflite` and runs it
through the Neuron Runtime. `kQnnNative` is a third self-contained runtime: it
runs a QNN context binary compiled ahead of time by
`qnn-context-binary-generator` straight through the QNN native C API. The two
MediaTek backends are not redundant — the
offline compiler maps models the on-device delegate rejects (see the DLA note
below).

## Layout

```
include/mie/engine.h            the entire public API
src/engine.cc                   factory: dispatch on Config::backend
src/model_source.h              ModelSource: a file path or an in-memory buffer
src/options.{h,cc}              key/value option helpers
src/fingerprint.{h,cc}          FNV-1a cache token, from a file or from memory
src/delegate_handle.h           RAII owner for a TfLiteDelegate
src/delegate_plugin.{h,cc}      dlopen + external-delegate plugin loader
src/tflite_interpreter.{h,cc}   shared TFLite core (CPU + GPU + QNN)
src/backend_cpu.{h,cc}
src/backend_gpu.{h,cc}
src/backend_qnn.{h,cc}
src/backend_mtk.{h,cc}          self-contained: NeuroPilot, no TFLite
src/backend_mtk_dla.{h,cc}      self-contained: Neuron Runtime, no TFLite
src/backend_qnn_native.{h,cc}   self-contained: QNN C API, no TFLite
src/backend_rknn.{h,cc}         self-contained: RKNN C API, no TFLite (ARM64 Linux)
examples/run_model.cc
tools/qnn_probe.cc              drives the QNN delegate's own C API (diagnostic)
tools/dla_compile.sh            on-device .tflite -> .dla compiler front-end
tools/dla_stale_probe.cc        detects whether the runtime honors changed inputs
tools/dla_raw_probe.cc          Neuron Runtime API behavior matrix (diagnostic)
jni/Android.mk                  ndk-build modules
jni/Application.mk              ndk-build ABI / platform / STL
cmake/android-r16b.cmake        CMake toolchain file for NDK r16b
cmake/aarch64-rockchip1240-linux-gnu.cmake  RKNN ARM64 Linux cross toolchain file
third_party/tflite-dist/        prebuilt TFLite 2.18.0: include/ + libs/ (committed)
third_party/rknn/               Rockchip RKNN header + AArch64 librknnrt.so
scripts/build_android.bat       ndk-build + package (see Build)
```

## API

```cpp
mie::Config config;
config.backend = mie::Backend::kGpu;
config.cache_dir = "/data/data/com.app/cache";   // enables the model cache

// Either a path...
config.model_path = "model.tflite";
// ...or bytes you already have in memory (see the lifetime rule below):
config.model_data = bytes.data();
config.model_size = bytes.size();

std::string error;
auto engine = mie::Engine::Create(config, &error);
if (!engine) { /* error */ }

std::memcpy(engine->Input(0).data, pixels, engine->Input(0).bytes);
engine->Run();
const void* logits = engine->Output(0).data;
```

`Input(i)` / `Output(i)` are views into engine-owned memory. `Run()` is a bare
invoke. `Create()` returns `nullptr` and fills `error` rather than throwing, so
it works in `-fno-exceptions` builds.

### Model buffer lifetime

`Config::model_data` is **non-owning**. The bytes go straight to the runtime,
which may keep a pointer into them instead of copying, so the buffer must stay
alive and unmodified for the whole lifetime of the `Engine`. A `std::vector`
that goes out of scope before the `Engine` does is a use-after-free.
`examples/run_model.cc` shows the discipline: declare the buffer *before* the
engine so C++'s reverse destruction order does the right thing.

(MTK copies internally, so it would tolerate a shorter-lived buffer — but the
rule is documented identically for every backend so that switching backends
does not silently change the contract.)

## Build

C++11, three target families, each producing a self-contained package under
`dist/mie-<platform>/` with the same layout (`include/`, `lib/`, `bin/`,
`examples/`, `testdata/`, `README.md`, `BUILD.txt`):

* **Android** (arm64-v8a) — all backends except RKNN; `ndk-build` or CMake.
* **Windows** (x86_64, MSVC) — CPU backend; CMake + Ninja.
* **ARM64 Linux** (aarch64, Rockchip) — RKNN backend; CMake cross-compile.

### ndk-build + packaging script (Android)

Tested configuration: **NDK r16b (clang 5.0), arm64-v8a**.

```bat
set ANDROID_NDK_ROOT=D:\path\to\android-ndk-r16b
scripts\build_android.bat /zip
```

The NDK path and the vendor SDK include dirs can instead be recorded once in
`scripts/local_paths.bat` (a committed template with example paths), which the
script reads automatically — so you don't have to set them every run. Its three
variables are all optional and each is only consulted when the matching backend
is built:

* `ANDROID_NDK_ROOT` — the NDK; required for every Android build.
* `MIE_MTK_SDK_INCLUDE_DIR` — only for `/dla`; the MediaTek NeuroPilot SDK's
  per-chip include dir (the one holding `neuron/api/RuntimeV2.h`).
* `MIE_QNN_SDK_INCLUDE_DIR` — only for `/qnnnative`; the Qualcomm QNN (QAIRT)
  SDK include dir (the one holding `QnnInterface.h`).

Pass `/sdk <dir>` / `/qnnsdk <dir>` to `build_android.bat` and it writes those
paths back into `scripts/local_paths.bat` for the next run.

It uses the committed prebuilt TFLite under `third_party/tflite-dist/`, runs
`ndk-build`, and writes a self-contained package:

```
dist/mie-arm64-v8a/
  include/mie/engine.h
  lib/libmie.a
  bin/mie_run                     + the TFLite runtime .so it needs
  bin/libtensorflowlite_c.so
  bin/libtensorflowlite_gpu_delegate.so
  examples/run_model.cc
  testdata/ssd/detect.tflite + labelmap.txt
  README.md  BUILD.txt
```

Options: `/mtk` `/dla` `/api27` `/abi <abi>` `/debug` `/zip` `/push` `/no-clean`,
plus `/qnnnative` (QNN native backend; needs `/qnnsdk <dir>`).
`/push` deploys `bin/` to `/data/local/tmp/mie` and makes the binary executable.
`/mtk` and `/dla` need MediaTek material that is not fetched automatically: the
shim header (`third_party/mtk/include/NeuroPilotTFLiteShim.h`) and the SDK's
`neuron/api` headers (`MIE_MTK_SDK_INCLUDE_DIR`, or a copy under
`third_party/mtk/sdk_include/`) respectively; `/qnnnative` needs the QNN SDK's
C API headers (`/qnnsdk <dir holding QnnInterface.h>`).

Bare `ndk-build` works too — the build files live in `jni/`, so from the project
root a plain `<ndk>\ndk-build.cmd` needs no extra flags.

Three ndk-build behaviours that each cost real time here:

* **`Application.mk` must be in `jni/`.** On r16b, `build/core/build-local.mk`
  resolves it with
  `NDK_APPLICATION_MK := $(strip $(wildcard $(NDK_PROJECT_PATH)/jni/Application.mk))`,
  which *clobbers* a value passed on the command line — so `NDK_APPLICATION_MK=…`
  does nothing at all and the build silently uses the default app config.
* **`APP_PLATFORM` and the `MIE_*` flags are not in the object path**, so changing
  them does not invalidate `obj/`. A stale tree silently yields a binary built for
  the *previous* configuration — which is why the script cleans by default.
* **Pass `-stdlib=libc++` explicitly when linking.** ndk-build adds the libc++
  archives itself but never passes `-stdlib`, so a bare `clang++` falls back to
  its own default and appends a spurious `-lstdc++` — visible as `libstdc++.so`
  in the executable's `DT_NEEDED`, the deprecated system STL riding along beside
  a statically linked libc++. `jni/Android.mk` sets it (plus a `-L` for the libc++
  directory, because `libc++.a` is a linker *script* whose `-l` references need it).

### CMake (Android)

```powershell
$env:ANDROID_NDK_ROOT = "D:\path\to\android-ndk-r16b"
cmake -B build -G Ninja -DCMAKE_TOOLCHAIN_FILE=cmake/android-r16b.cmake `
  -DANDROID_ABI=arm64-v8a -DANDROID_API=21
cmake --build build
```

Produces `build/libmie.a` and `build/mie_run` (ELF64 AArch64). Both routes emit
byte-comparable results — the same checksums on device.

For a modern NDK or a Gradle build, skip the toolchain file and point CMake at
your own TFLite with `-DMIE_TFLITE_INCLUDE_DIRS=` / `-DMIE_TFLITE_LIBRARIES=` /
`-DMIE_TFLITE_GPU_DELEGATE_LIB=`.

### Windows x86_64 (MSVC)

CPU backend only — the GPU delegate ships only for Android, and QNN/MTK need the
vendor's device runtime. The prebuilt TFLite C API import library
(`tensorflowlite_c.dll.if.lib` + `tensorflowlite_c.dll`) lives in
`third_party/tflite-dist/libs/windows_x86_64/` and is picked up automatically.
From a Visual Studio developer prompt (or any shell with `cl`/MSVC on `PATH`):

```powershell
cmake -B build-win -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build-win
```

Produces `build-win/mie.lib` (static) and `build-win/mie_run.exe`. The exe loads
`tensorflowlite_c.dll` from its own directory, so ship them together:

```
dist/mie-windows_x86_64/
  include/mie/engine.h
  lib/mie.lib
  bin/mie_run.exe + tensorflowlite_c.dll
  examples/run_model.cc
  testdata/ssd/detect.tflite + labelmap.txt
  README.md  BUILD.txt
```

### ARM64 Linux / RKNN

RKNN is a separate ARM64 Linux build family and is deliberately unavailable to
Android and Windows builds. The repository includes the RKNN header and the
AArch64 Linux `librknnrt.so` under `third_party/rknn/`. The Rockchip cross
toolchain is a Linux ELF with symlinks that Windows `tar.exe` cannot extract, so
it must live on a Linux filesystem. From WSL, extract it under e.g. `/opt` and
build with the supplied `cmake/aarch64-rockchip1240-linux-gnu.cmake`:

```bash
cmake -S . -B build-rknn \
  -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64-rockchip1240-linux-gnu.cmake \
  -DMIE_ROCKCHIP_TOOLCHAIN_ROOT=/opt/aarch64-rockchip1240-linux-gnu \
  -DMIE_ENABLE_RKNN=ON -DMIE_ENABLE_TFLITE_BACKENDS=OFF -DMIE_ENABLE_GPU=OFF
cmake --build build-rknn -j2
```

`cmake --install build-rknn --prefix <staging>` rewrites `mie_run`'s rpath to
`$ORIGIN/../lib` and installs `librknnrt.so` next to `libmie.a`, giving the same
layout as the other targets:

```
dist/mie-linux_aarch64/
  include/mie/engine.h
  lib/libmie.a + librknnrt.so
  bin/mie_run        (aarch64 ELF; finds librknnrt.so via $ORIGIN/../lib)
  examples/run_model.cc
  testdata/ssd/detect.tflite + labelmap.txt   (reference only; RKNN loads .rknn)
  README.md  BUILD.txt
```

The generic RKNN engine accepts a native `.rknn` model and exposes static model
I/O through engine-owned staging buffers. `ResizeInput()` is not supported.
By default it follows the scene-wrapper convention (`UINT8`/`NHWC` input
conversion and float32 outputs). For model-native buffers, pass
`MIE_OPTIONS="rknn_pass_through=true,rknn_want_float=false"`; use
`rknn_core_mask=core0`, `core0_1`, `core0_1_2`, or `all` to select NPU cores.

## Measured on a real device

A Qualcomm device, Android 16, arm64-v8a. `add.bin` is a 544-byte toy and
`multi_add.bin` a 4-input / 2-output toy; `detect.tflite` is a real 4.2 MB
SSD-MobileNet.

| model | backend | nodes delegated | init | per inference |
|-------|---------|-----------------|------|---------------|
| `add.bin` | `cpu` | – | 0.9 ms | 0.4 us |
| `add.bin` | `gpu` | 2/2 | 263 ms | 294 us |
| `add.bin` | `qnn` | 2/2 | 285 ms | 337 us |
| `multi_add.bin` | `cpu` | – | 7.8 ms | 0.9 us |
| `multi_add.bin` | `gpu` | 3/3 | 408 ms | 911 us |
| `multi_add.bin` | `qnn` | 3/3 | 712 ms | 918 us |
| `multi_add.bin` | `qnnnative` | – | 125 ms | 927 us |
| `detect.tflite` | `cpu` | – | 2.3 ms | 6.05 ms |
| `detect.tflite` | `gpu` cold | partial | 1187.0 ms | 9.68 ms |
| `detect.tflite` | `gpu` **warm cache** | partial | **160.5 ms** | 11.51 ms |
| `detect.tflite` | `qnn` | **64/64** | 630 ms | **0.82 ms** |
| `detect.tflite` | `qnnnative` | – | 146 ms | 2.169 ms |

* **QNN is the clear winner on real work: 0.82 ms — 7× faster than CPU and 11×
  faster than GPU**, because it delegates the entire graph where the GPU
  delegate only takes part of it and pays for the fallback copies.
* **GPU does not automatically win.** On SSD-MobileNet it was *slower* than CPU.
  Measure before choosing.
* **The cache pays for itself**: GPU init drops 1187 ms → 160.5 ms (7.4×) on the
  real model; QNN 440.6 ms → 124.8 ms (3.5×). The log tells you which happened —
  GPU flips to `Initialized OpenCL-based API from serialized data.`
* `multi_add.bin` (4 inputs, 2 outputs) is the cross-backend consistency
  benchmark: both outputs are byte-identical across `cpu`/`gpu`/`qnn`/
  `qnnnative` (FNV-1a checksum `5cba39c5` each), which exercises the multi-input
  → multi-output tensor mapping on every path.
* `detect.tflite` on `qnnnative` was measured with the on-device delegate-cache
  context binary, not the offline pipeline. Its init/latency are valid, but that
  binary drops source tensor names and may reorder same-shape outputs (see the
  QNN native section) — use the offline pipeline when output identity/order
  matters.

A MediaTek device (Android 16). `dla` numbers
are the same models compiled with `ncc-tflite --arch=mdla3.0` on the device,
run through the V2 request-based runtime API (`NeuronRuntimeV2_*`):

| model | backend | init | per inference |
|-------|---------|------|---------------|
| `ssd/detect.tflite` | `cpu` | 2.3 ms | 7.93 ms |
| `ssd/detect.tflite` | `mtk` (63/64 nodes) | 874.6 ms cold / 315.2 ms warm cache | 2.77 ms |
| `add.bin` | `dla` | 11.4 ms | **0.55 ms** |

`dla` also skips the compile step entirely — init is a plain load of the `.dla`
instead of an on-device compilation — and its outputs agree with the CPU
reference to about three decimal places (fp16 rounding). The V2 request path
pays a few ms per run over the earlier V1 bind-once measurements — that
overhead is what re-submits and re-reads the input every run, i.e. the price of
correct per-frame inputs; compare against CPU before choosing, as always.

## Model cache

`Config::cache_dir` turns on the backend's compiled-model cache:

* **`kGpu`** — sets `serialization_dir` + `ENABLE_SERIALIZATION` on the GPU
  delegate. **The directory must already exist** — the delegate does not
  create it, and with a missing dir it fails silently (`ERROR: Failed to save
  serialized data`), every init recompiles (~2 s instead of ~0.26 s warm), and
  the early steady-state runs measure noticeably slower until the OpenCL
  driver's own shader cache warms up. Same rule as `kMtk` below.
* **`kQnn`** — sets the `cache_dir` / `model_token` plugin options (QNN context
  binary). Both are required.
* **`kMtk`** — sets `ANeuralNetworksTFLiteOptions_setCacheDir` (visible in the
  vendor log as `NeuronCompilation_setCaching`). The directory must already
  exist — the runtime does not create it.
* **`kMtkDla`** — needs none: the `.dla` file *is* the compiled artifact. There
  is no compile step at load time, which is exactly why its init is ~20 ms.

`cache_token` defaults to an FNV-1a hash of the model bytes (file or buffer),
which is safe: identical bytes ⇒ identical graph ⇒ the cached artifact is valid.
Make `cache_dir` app-private — a writable dir other apps can reach is both a
correctness and a security problem.

## Backend notes

**Qualcomm QNN** — no build-time dependency; the delegate is `dlopen`ed. Get the
QAIRT SDK from `https://softwarecenter.qualcomm.com/#/catalog/item/Qualcomm_AI_Runtime_Community`
(verified against 2.50.0.260828; the ZIP downloads without an account). Deploy
`lib/aarch64-android/*.so` to `jniLibs` and `lib/hexagon-v<NN>/unsigned/*` (the
skels — **not** from jniLibs) to a device dir, then point `ADSP_LIBRARY_PATH` or
`opt::kSkelLibraryDir` at them.

Two traps that each cost an afternoon:

* **`htp_performance_mode` takes the NUMERIC enum only** — `2` for burst, never
  `"burst"`. Any name string makes the delegate call `std::terminate` and
  **abort the whole process**, with no error message. Measured on the real
  device, `2` (burst) drops `detect.tflite` from ~1.67 ms to ~0.25 ms per
  inference (6.6×) with byte-identical outputs — the default leaves the HTP in
  an unconfigured low-power state, so set `opt::kHtpPerformance` explicitly for
  latency-critical work.
* **Do not set `library_path`** when the libs are already on `LD_LIBRARY_PATH` /
  in `jniLibs`; it makes delegate application fail.

**Qualcomm QNN native C API (`kQnnNative`)** — no TFLite; it `dlopen`s
`libQnnHtp.so` and `libQnnSystem.so` and drives the QNN C API directly. Input is
a precompiled context binary (`.bin`), produced on the device by the `kQnn`
delegate cache (`cache_dir`/`model_token`) or offline by
`qnn-context-binary-generator`. Build with `-DMIE_ENABLE_QNN_NATIVE=ON` and
`-DMIE_QNN_SDK_INCLUDE_DIR=<dir holding QnnInterface.h>`, or `/qnnnative` /
`/qnnsdk <dir>` on the packaging script.

**Offline generation (canonical pipeline).** Prefer this over the delegate cache:
the delegate cache drops source tensor names and may reorder same-shape outputs;
the offline pipeline preserves both. From the SDK's `bin/x86_64-linux-clang/`
(the converter/model-lib-generator are Python3; run them on Linux x86_64):

```bash
# 1. TFLite -> QNN model (.cpp + weight .bin)
qnn-tflite-converter --input_network model.tflite --output_path model.cpp

# 2. Model library (.so) — needs clang++ on PATH
qnn-model-lib-generator -c model.cpp -o model_lib

# 3. SoC-specific context binary
qnn-context-binary-generator \
  --model model_lib/x86_64-linux-clang/libmodel.so \
  --backend <qnn>/lib/x86_64-linux-clang/libQnnHtp.so \
  --htp_socs sm8750 \
  --binary_file model_ctx.bin
```

* `--htp_socs <asic-id>` targets a specific SoC (e.g. `sm8750` = SD 8 Elite /
  V79). Without it the host emits a fallback arch and the device fails with
  `QnnContext_createFromBinary ... must match this device's SoC`. The SoC -> arch
  table lives in `docs/QAIRT-Docs/QNN/general/overview.html` (e.g. `sm8750` -> 69 / V79).
* The "may fall back to host default SoC/arch" WARNING that appears without
  `--backend_extensions_lib_path` is misleading for the `.so -> .bin` path: the
  generated file is still named `<file>.SM8750.bin` and loads on the device.
  (Passing `--backend_extensions_lib_path libQnnHtpNetRunExtensions.so` instead
  fails with `ComposeGraphs Failed`; it also needs a graph-level config, so omit it.)
* Ops with no HTP implementation fail at generation: `TFLite_Detection_PostProcess`
  becomes `DetectionOutput` and aborts prepare (err 1002) — those ops must stay on
  the CPU side. A pure-`ADD` model (like `testdata/multi_add.bin` with two
  same-shape outputs `x`,`y`) is the right probe for order preservation.

* **Graph I/O tensors are baked into the binary.** Introspect them with
  `QnnSystemContext_getBinaryInfo` and hand the raw descriptors straight to
  `graphExecute` — do not call `tensorCreateGraphTensor` /
  `tensorCreateContextTensor`, the tensors already exist.
* **Force `QNN_TENSOR_VERSION_1` on the execution descriptors.** The introspected
  tensor may be V2, whose extra fields shift the `clientBuf` union offset.
* **QAIRT 2.50 no longer exports `QnnSystemContext_*`.** `libQnnSystem.so` only
  exports `QnnSystemInterface_getProviders`; resolve `systemContextCreate` /
  `getBinaryInfo` / `free` from the returned function table.
* Options: `opt::kQnnNativeLibrary` / `kQnnSystemLibrary` (the `.so` name or
  path) and `kQnnGraphName` (which graph, when the binary holds several).
* `ResizeInput` is always a no-op: shapes are baked in.
* **Delegate-cache binaries lose tensor names and order.** Their introspected
  names are QNN-internal ids (`"0"`, `"1"`, …) — not the source `.tflite` names —
  and same-shape outputs may be reordered. The offline pipeline preserves source
  names and order, so use it when output identity/order matters.

**MediaTek NeuroPilot** — `-DMIE_ENABLE_MTK=ON -DMIE_MTK_INCLUDE_DIR=<dir with
NeuroPilotTFLiteShim.h> -DANDROID_API=27`. There is **no `MIE_MTK_LIB`**: the
shim is header-only and resolves `libtflite_mtk.so` with `dlopen`, so nothing is
linked. The header is confidential MediaTek material and is not fetched
automatically.

* **`ANDROID_API` must be ≥ 27.** The shim uses NNAPI constants
  (`ANEURALNETWORKS_BAD_STATE`) and `ANeuralNetworksModel` without including
  `<android/NeuralNetworks.h>` — `backend_mtk.cc` includes it first, and that
  header's contents are gated to API 27+. CMake fails with a clear message
  otherwise.
* **MTK is not zero-copy.** `setInputTensorData` / `getOutputTensorData` take
  copied buffers, so `Input(i).data` is an engine-owned staging buffer and
  `Run()` copies in and out. The calling convention is identical to the other
  backends; the cost is not.
* **`ResizeInput` always returns false** for MTK: the shim's resize is a
  creation-time option, not a runtime call.
* **Set `opt::kAcceleratorName` to the NPU device id.** On the tested device the
  vendor enumerates `mtk-gpu` and `mtk-mdla`; leaving the name unset lets layer
  verification for some graphs (observed with an ElementWiseAdd) reach the
  vendor's ArmNN/OpenCL path, which throws inside `libarmnn.so` and **aborts the
  process** (`terminating`) with no error message. `accelerator_name=mtk-mdla`
  avoids that and pins compilation to the NPU.
* **Not every graph maps to the NPU.** Float32 models exported from TF2
  (DeepLabV3/MBV2 and UNet segmentation, a YOLO-style detector) were accepted by
  the delegate but failed compilation with `NEURON_UNMAPPABLE`; fully-quantized
  int8/uint8 models compiled and ran. The failure is clean — `Create` returns an
  error instead of falling back silently.

**MediaTek DLA** — `-DMIE_ENABLE_MTK_DLA=ON -DMIE_MTK_SDK_INCLUDE_DIR=<dir with
neuron/api/RuntimeAPI.h>`, or `/dla` on the packaging script. No
`ANDROID_API` floor: these headers are plain C. Nothing is linked either — the
Neuron Runtime (`libneuron_runtime.so`, present on MediaTek devices) is
`dlopen`ed, so the backend runs on a stock phone with no deployment step.

* **It runs models the `kMtk` path cannot.** The offline compiler has the larger
  op/target surface: several real models that die in the delegate path with
  `NEURON_UNMAPPABLE` compile and run through `kMtkDla`. The two backends are
  complements, not duplicates.
* **Compile the `.dla` with the SDK's `ncc-tflite`, on the device.** The host
  build is a Linux ELF; the per-chip Android build runs from a shell.
  `tools/dla_compile.sh` drives it (deploy notes are in the script):
  `ncc-tflite --arch=mdla3.0 -d model.dla model.tflite`. `--arch=?` lists the
  targets; the tested MediaTek parts take `mdla3.0`.
* **A `.dla` is version-locked to the runtime's MAJOR version.** The default
  `dlopen` target is the device's own `libneuron_runtime.so`; if the compiler you
  used is newer, point `opt::kDlaLibrary` at the matching library from the same
  SDK, e.g. `neuron_sdk/<chip>/lib/libneuron_runtime.so.7.3.15`.
* **Element types are not exposed by the Neuron Runtime API** — it reports
  shapes and sizes only, and there is no getter for the dtype. `Tensor::type` is
  therefore reported as `FLOAT32`, which is what the compiler's default
  converting mode exchanges with the caller; `bytes` always comes from the
  runtime and stays exact. Quantized I/O would need the type to be carried
  out-of-band.
* **`ResizeInput` always returns false**, for the same reason as `kMtk`: input
  shapes are baked in at compile time (the runtime can only re-shape models
  compiled with `--runtime-dynamic-shape`, which this backend does not expose).
* **The backend uses the V2 request-based API — and that is what makes inputs
  live.** `kMtkDla` drives `NeuronRuntimeV2_create/createFromBuffer` +
  `NeuronRuntimeV2_run`, whose `SyncInferenceRequest` carries the IOBuffer
  descriptors on EVERY call; the runtime re-reads the staging buffers each
  run, so rewriting `Input(i).data` between runs is honored (verified on the
  device by `tools/dla_stale_probe.cc`: input 1.0 → 2.0 flips the output on
  `add.dla`). The earlier V1
  (`NeuronRuntime_*` set-once-then-infer) build was different: that family
  latches the input at the FIRST inference — re-calling setInput (same or
  different pointer), per-request V2 buffers driven through a swapped
  descriptor, clone-per-run, enqueue-trigger and the `suppress-input` option
  were all probed and none re-read host memory (`tools/dla_raw_probe.cc`
  holds the full 24-step matrix), and the ION route — the one path that must
  be read live — was unreachable (every `/dev/dma_heap/*` node refuses CPU
  mmap; `AHardwareBuffer` BLOB allocation fails EINVAL). Two quirks worth
  knowing: the descriptors must stay STABLE across runs (the engine's
  fixed staging buffers satisfy this; a runtime observed going stale when a
  request presented a brand-new descriptor each call), and
  `restoreFromCompiledNetwork` only accepts the Adapter API's own
  `storeCompiledNetwork` blobs — it rejects an ncc-tflite `.dla` with
  `NEURON_BAD_DATA(4)`, and the reverse pairing is equally incompatible.
* **Diagnose compile failures by their message.** `MDLA: unsupported operation`
  means an op the NPU cannot run at all (e.g. an SSD's TFLite-custom NMS tail —
  cut it out of the graph, or use `kMtk`/`kGpu` which can fall back to CPU).
  `DilationRate unsupported and cannot use SWDilated` / `(T:18,R:18,B:18,L:18) is
  too large` is a hardware limit on atrous convolutions: DeepLabV3's ASPP at
  rate 18 cannot be legalized even with `--use-sw-dilated-conv`.

## Verification status

| backend | compiles | links | runs on a device |
|---------|----------|-------|------------------|
| CPU / GPU / QNN | yes | yes | **yes** — Qualcomm device |
| QNN native | yes | yes | **yes** — Qualcomm device; the `add.bin` / `multi_add.bin` context binaries reproduce checksum `5cba39c5` |
| MTK | yes | yes | **yes** — MediaTek device, Android 16 |
| MTK DLA | yes | yes | **yes** — same device, `.dla` from `ncc-tflite` 7.3.15; V2 request API, per-run input tracking verified |
| RKNN | yes | yes | not yet — cross-compiled to aarch64 ELF; needs a Rockchip NPU device to confirm |

MTK is verified against the real `NeuroPilotTFLiteShim.h` (55,996 bytes, sha256
`AF772A00…`) and **runs on a real MediaTek device**. The vendor's
`/vendor/lib64/libtflite_mtk.so` provides the runtime; the delegate takes the
graph and the Neuron compiler emits for the `mtk-mdla` device. Measured on the
quantized SSD-MobileNet (`testdata/ssd/detect.tflite`): 63 of 64 nodes delegated
(the NMS tail stays on XNNPACK), **2.77 ms per inference vs 8.18 ms on the CPU
backend (2.95×)**, deterministic outputs, and the detection-count tensor
byte-identical to CPU; `cache_dir` takes init from 874.6 ms cold to 315.2 ms
warm. Two device-side caveats found on that phone: `accelerator_name=mtk-mdla` is
strongly recommended (some graphs otherwise reach the vendor's ArmNN/OpenCL
verification path, which aborts the process), and TF2-exported float32 models can
fail to compile with `NEURON_UNMAPPABLE` — that failure is clean and quantized
models are the reliable path. See the MTK note below.

MTK DLA is verified the same way, on the same phone: `.dla` files compiled there
by the SDK's `ncc-tflite` 7.3.15 (arm64) loaded and ran through the vendor
runtime, with `add.bin` reproducing the cross-backend checksum `5cba39c5`
byte-for-byte. Both the device's own runtime and the SDK's `libneuron_runtime.so.7.3.15`
(via `opt::kDlaLibrary`) load the same `.dla`.

## Environment quirks worth knowing

**NDK r16b has no `ld` next to clang.** Its
`toolchains/llvm/prebuilt/windows-x86_64/bin` holds only the `clang*`
executables — no `ld`, no `llvm-ar`, no `llvm-nm` — so clang's bare `ld` lookup
fails with `unable to execute command: program not executable`.
`cmake/android-r16b.cmake` selects the GNU 4.9 linker explicitly with
`-fuse-ld=<…>/aarch64-linux-android-ld.exe`. A complete NDK does not need this.

**Android refuses non-PIE executables**, and CMake will not add `-pie` when it
believes it is targeting plain Linux, so the toolchain file adds it.

## Notes

* **Out-of-range tensor indices are contained, not fatal.** `Input(i)` /
  `Output(i)` return a reference and so cannot report an error; an index outside
  `[0, NumInputs())` / `[0, NumOutputs())` is logged and yields an empty tensor
  (`bytes == 0`, `data == nullptr`) instead of reading past the end of the
  underlying array. libc++ container annotations are off in this NDK, so such a
  read would *not* have been caught by AddressSanitizer — the guard is the only
  defence.
* **Escape hatch.** Setting `Config::delegate_lib` loads that library through the
  external-delegate plugin ABI instead of the backend default. Only `kQnn` reads
  it; setting it with any other backend logs a warning rather than being ignored
  silently.
* **Backends fail loudly.** Asking for `kGpu` where it is unavailable is an
  error, not a silent CPU fallback — silent fallback hides a 20× regression.
* **One engine, one model, one thread.** An `Engine` is not thread-safe; give
  each worker its own, or serialize `Run()`.
* Vendor struct layouts were taken from the vendor headers current at the time
  of writing: QNN delegate v2.30, TFLite 2.18.0, and the NeuroPilot shim above.
  Re-check them on a vendor SDK bump.
* Not implemented: the GPU async API, custom op resolvers, multiple delegates on
  one interpreter.
