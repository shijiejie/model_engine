# Generating QNN context binaries: delegate cache vs. Qualcomm AI Hub

`kQnnNative` loads a **precompiled QNN context binary** (`.bin`) through the QNN
native C API (`QnnContext_createFromBinary`) — there is no on-device compile
step, so the `.bin` has to come from somewhere. There are three producers:

| # | Producer | Where it runs | Preserves names/order | Custom NMS tail |
|---|----------|---------------|-----------------------|-----------------|
| 1 | `qnn-context-binary-generator` (offline pipeline) | host, needs the Linux x86_64 QAIRT toolchain (e.g. WSL) | ✅ | ❌ fails (err 1002) |
| 2 | QNN **TFLite delegate cache** | **on the device** | ❌ | ✅ |
| 3 | **Qualcomm AI Hub** (cloud) | Qualcomm's cloud | ❌ | ✅ (once the custom op is gone) |

The offline pipeline is already covered in the main `README.md`
("Offline generation"). This document walks through **2 and 3** — both
"online" in the sense that no local QAIRT toolchain is required — with the
exact commands, every trap hit while validating them on a real Snapdragon 8
Elite (SM8750) device with QAIRT 2.50.0.260828, and the official references.

---

## 1. On-device: QNN TFLite delegate cache

The `kQnn` backend (`src/backend_qnn.cc`) drives the QNN TFLite delegate
(`libQnnTFLiteDelegate.so`). With a `cache_dir` (and optional `model_token`)
set, the delegate **compiles the graph on-device and serializes the finished
context binary into the cache** — that file is exactly what `kQnnNative`
consumes later.

### How to produce it (mie)

On the device, with the QNN libs and HTP skels deployed (see the main README
"Backend notes" for the deploy recipe — `ADSP_LIBRARY_PATH` must point at the
skels or the binary will not load, pitfall 2 below):

```bash
adb shell 'cd /data/local/tmp/mie && \
  LD_LIBRARY_PATH=/data/local/tmp/mie \
  ADSP_LIBRARY_PATH=/data/local/tmp/mie/skels \
  ./mie_run detect.tflite qnn detect_ctx 20 4'   # cache_dir = detect_ctx
```

On the device the cache then contains:

```
detect_ctx/
  ├── qnn_binary_16993899529729008237.bin   <- the context binary (4,543,864 bytes)
  └── be6eabc63936b6972.39.0_*.bin          <- delegate bookkeeping files
```

The name is always `qnn_binary_<token>.bin`, where `<token>` is the FNV-1a
hash of the model bytes (`src/fingerprint.{h,cc}`) unless `model_token` is
overridden.

### How to consume it (mie)

```bash
./mie_run detect_ctx/qnn_binary_16993899529729008237.bin qnnnative "" 20 4
```

### What we verified on the device

Running the same `detect_ctx/qnn_binary_*.bin`:

| run | init | out[0] boxes | out[1] | out[2] | out[3] count |
|-----|------|--------------|--------|--------|--------------|
| `qnn` cold compile (`cache_dir ""`) | 1329 ms | `c51ef85e` | `50deabbe` | `df0e04b5` | `4ae5a562` |
| `qnn` with `cache_dir detect_ctx` (cache hit) | 182.6 ms | `c51ef85e` | `50deabbe` | `df0e04b5` | `4ae5a562` |
| `qnnnative` on the same `.bin` | 176 ms | `c51ef85e` | **`df0e04b5`** | **`50deabbe`** | `4ae5a562` |

### Pitfalls (all hit, all explained)

1. **The delegate cache drops source tensor names and may reorder same-shape
   outputs.** Introspected names are QNN-internal ids (`"167"`, `"169"`, …), not
   the `.tflite` names. For the two `[1,10]` outputs of `detect.tflite`, the
   `qnn` delegate reports them in TFLite order while `qnnnative` reports them
   swapped (`df0e04b5`/`50deabbe` exchanged) — **the same binary, two different
   tensor mappings** (TFLite output-tensor table vs. the binary's internal id
   order). Do not rely on output identity/order from cache binaries; use the
   offline pipeline (or AI Hub) when order matters.
2. **`ADSP_LIBRARY_PATH` must point at the HTP skels**, otherwise
   `QnnContext_createFromBinary` fails — either with "skel load err 1002" or a
   misleading "must match this device's SoC" (see the main README's
   "Device-side errors are not masked" note: real cause is exposed via
   `errorGetMessage`, e.g. `error 0x36b1 (QNN_DEVICE_ERROR_INVALID_CONFIG)`).
3. **Cold compile is expensive; the cache is the point.** First init ~1329 ms
   (compile) vs. cache hit ~182 ms.
4. **Custom NMS tails only pass via the cache.** `TFLite_Detection_PostProcess`
   lowers to `DetectionOutput`/`q::MultiClassNms`, which HTP cannot prepare
   (err 1002) — the offline pipeline rejects the whole model, but the delegate
   cache still works because the NMS tail stays on the CPU side.
5. **The `nocache`-style "Failed to open for writing" warnings are harmless**
   if you point `cache_dir` at a nonexistent dir: the delegate still runs,
   it just cannot persist the binary.

---

## 2. Cloud: Qualcomm AI Hub

Qualcomm AI Hub compiles ONNX / PyTorch models on Qualcomm's servers and hands
back the context binary — no local QAIRT toolchain, no WSL. Free accounts can
submit compile jobs.

### Prerequisites

```bash
pip install qai-hub
```

Get an API token from the AI Hub console: log in at
<https://aihub.qualcomm.com/>, avatar menu → Settings → API token. Pass it to
the SDK without writing it to disk, e.g. via an environment variable
(`QAI_HUB_TOKEN`).

### The minimal flow (verified against qai-hub 0.56)

```python
import os
import qai_hub as hub

hub.set_session_token(os.environ["QAI_HUB_TOKEN"])   # 0.56 API; see pitfall 1

# Pick a device that matches the target SoC. SM8750 == Snapdragon 8 Elite
# (Hexagon V79, soc-model 69). The Samsung Galaxy S25 family is SM8750.
device = next(d for d in hub.get_devices()
              if any("8750" in a for a in d.attributes))

# Step 1+2 in one call: compile to a context binary, then "link" it.
# Returns (list[CompileJob], LinkJob).
_, link_job = hub.submit_compile_and_link_jobs(
    models="model.onnx",       # PyTorch (.pt/.pt2) or ONNX
    device=device,
)
link_job.wait()

# The LinkJob's target model IS the raw context binary.
qnn_bin = link_job.get_target_model()
qnn_bin.download("model_aihub.bin")
```

Consume it on the device exactly like any other `.bin`:

```bash
adb push model_aihub.bin /data/local/tmp/mie/
adb shell 'cd /data/local/tmp/mie && \
  LD_LIBRARY_PATH=/data/local/tmp/mie \
  ADSP_LIBRARY_PATH=/data/local/tmp/mie/skels \
  ./mie_run model_aihub.bin qnnnative "" 20 4'
```

### Repository scripts

The three scripts used for this document live in
[`tools/aihub/`](../tools/aihub/), token-free (the token is read from the
`QAI_HUB_TOKEN` environment variable only, never from disk):

| script | purpose |
|--------|---------|
| `tools/aihub/compile_to_qnn.py` | `python compile_to_qnn.py <model.onnx|model.pt2> [out.bin]` — pick the SM8750 device, compile+link, download the context binary; on failure it saves `compile.log` |
| `tools/aihub/strip_nms.py` | `python strip_nms.py <in.tflite> <out.tflite>` — cut a trailing CUSTOM NMS op from a `.tflite` (needed before any TFLite→ONNX conversion, see §3) |
| `tools/aihub/make_multi_add_onnx.py` | build the `multi_add.onnx` probe (4 inputs → 2 outputs) used for cross-backend checksum verification |

### Pitfalls (all hit, all fixed)

1. **qai_hub 0.56 API drift.** `set_config_with_token` no longer exists and
   `set_session_token` is deprecated (still works, prints a warning). Modern
   spelling: `hub.Client(config=qai_hub.ClientConfig("<token>"))`.
2. **`submit_compile_and_link_jobs` returns a tuple
   `(List[CompileJob], LinkJob)`** — index `[0]` is a *list* of compile jobs
   (one per model), `[1]` is the link job. Don't call `.wait()` on the list.
3. **`--qairt_version 2.50` is rejected for ONNX → context binary** with
   `QAIRT SDK version is not applicable to selected runtime.` Omit it; the
   default version compiled binaries that loaded fine on device QAIRT 2.50.
4. **`--target_runtime qnn_context_binary` is not a valid value.** The valid
   runtimes are `onnx | precompiled_qnn_onnx | qnn_dlc | tflite`. A plain
   `submit_compile_job` defaults to **tflite** (you get a `.tflite` out). The
   context binary comes from the compile+link flow above (or by wrapping with
   `--target_runtime precompiled_qnn_onnx` for ONNX Runtime).
5. **AI Hub does not accept TFLite.** Only PyTorch / ONNX / AIMET-quantized
   / TensorFlow-via-ONNX. A `.tflite` must first be converted to ONNX — and
   TFLite **custom ops block the converters** (see §3 for the strip trick).
6. **tflite2onnx emits an ONNX that AI Hub rejects** with
   `Tensors {…} occur in value_info but also in model IO` (an IR violation per
   <https://github.com/onnx/onnx/blob/main/docs/IR.md>). Fix: drop the
   `value_info` entries whose names duplicate the graph inputs/outputs before
   submitting (onnx.checker does *not* catch this locally).
7. **Device selection.** Filter the pool by chipset attribute (`"8750"`),
   not by name — there is no literal "8 Elite QRD" in the pool; `Samsung
   Galaxy S25 (Family)` carries `chipset:sm8750-ac`, `hexagon:v79`,
   `soc-model:69`.
8. **Input layout flips to NCHW.** The AI Hub binary for the detect backbone
   reports input `[1,3,300,300]` (NCHW) where the source `.tflite` had
   `[1,300,300,3]` (NHWC). All-`1.0` fill is unaffected; real inputs must be
   converted NHWC→NCHW before `Run()`.
9. **Cloud binaries drop source output names too** (observed
   `output_0`/`output_1` for the two backbone outputs). Inputs kept theirs
   (`a`,`b`,`c`,`d` on the multi_add probe). Same ordering caveat as §1.1.
10. **The context binary is SoC-specific but OS-agnostic** and only runs on
    the NPU (official wording). Compile per target device family.
11. **When a job fails, read the logs.** `cjobs[0].get_status()` gives the
    API-level message (e.g. the `value_info` IR violation of pitfall 6), and
    `cjobs[0].download_job_logs("compile.log")` fetches the converter log.
    Both are wired into `tools/aihub/compile_to_qnn.py`.

### TFLite → ONNX: stripping a custom NMS tail (detect.tflite example)

`tflite2onnx` aborts on custom ops:

```
NotImplementedError: Unsupported TFLite OP: 32 CUSTOM!
```

`TFLite_Detection_PostProcess` is the **last** node of the main subgraph, so
the fix is to cut it in place — repoint the subgraph outputs to its input
tensors (`Squeeze` = box encodings, `convert_scores` = class scores) and drop
the node. The surgery is pure byte patching, no rebuild (`tflite` package is
read-only, so vector positions are located by probing vtable slots):

```bash
python tools/aihub/strip_nms.py detect.tflite detect_backbone.tflite
```

(`tools/aihub/strip_nms.py` prints the CUSTOM op's inputs, e.g.
`[(165,'Squeeze'), (174,'convert_scores'), (171,'anchors')]`, and writes the
patched file — the same probe logic is what found the positions above. Verify
the strip with `mie_run detect_backbone.tflite cpu` on the device: the
original names `Squeeze`/`convert_scores` should survive.)

Then `tflite2onnx detect_backbone.tflite detect_backbone.onnx`, apply pitfall
6's `value_info` cleanup, and submit. Result: 63 ops, 1 input
`[1,300,300,3]`, outputs `[1,1917,4]` + `[1,1917,91]`.

### What we verified on the device

| model | AI Hub `.bin` | device result |
|-------|---------------|---------------|
| multi_add (4 fp32 inputs → 2 outputs) | 46,120 bytes | init 182.7 ms, both outputs all-3.0, checksum `5cba39c5` — **byte-identical to the offline-pipeline binary** |
| detect backbone (real quantized SSD-MobileNet CNN) | 4,524,432 bytes | init 152 ms, outputs `[1,1917,4]` + `[1,1917,91]`, 2.67 ms/infer vs. 5.31 ms CPU on the stripped tflite |

---

## Comparison: delegate cache vs. AI Hub vs. offline pipeline

| | delegate cache | AI Hub | offline pipeline |
|---|---|---|---|
| Where it runs | device | cloud | host (Linux x86_64 / WSL) |
| Input format | `.tflite` | ONNX / PyTorch | `.tflite` |
| QAIRT toolchain needed locally | no | no | yes |
| Custom NMS tail | ✅ | ✅ (after strip) | ❌ |
| Output names/order | ❌ lost | ❌ lost | ✅ preserved |
| Typical use | quick on-device test | no-local-toolchain release path | canonical, order-critical path |

## Official references

- Qualcomm AI Hub — Compiling Models (the `submit_compile_and_link_jobs`
  example): <https://workbench.aihub.qualcomm.com/docs/hub/compile_examples.html>
- Qualcomm AI Hub — API / common options:
  <https://workbench.aihub.qualcomm.com/docs/hub/api.html>
- Qualcomm AI Hub — FAQ (token, security, fees):
  <https://workbench.aihub.qualcomm.com/docs/hub/faq.html>
- QAIRT — HTP API usage guidelines (Online Prepare / `QnnContext_getBinary`):
  <https://docs.qualcomm.com/bundle/publicresource/80-63442-10/topics/htp_api_usage_guidelines.md>
- QAIRT — supported API matrix (`QnnContext_getBinary` on HTP aarch64-android):
  <https://docs.qualcomm.com/bundle/publicresource/80-63442-10/topics/supported_api.md>
- QAIRT — C++ application sample (`contextGetBinarySize`):
  <https://docs.qualcomm.com/bundle/publicresource/80-80020-15B/topics/develop-your-own-application-qairt-cpp.md>
- ONNX Runtime — QNN Execution Provider (precompiled QNN ONNX):
  <https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html>
- ONNX IR (why `value_info` must not duplicate model IO):
  <https://github.com/onnx/onnx/blob/main/docs/IR.md>
- QAIRT SDK download: <https://softwarecenter.qualcomm.com/#/catalog/item/Qualcomm_AI_Runtime_Community>
