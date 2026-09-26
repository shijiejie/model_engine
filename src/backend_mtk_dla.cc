#include "src/backend_mtk_dla.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "src/dl.h"
#include "src/options.h"
#include "src/tensor_view.h"

#if MIE_ENABLE_MTK_DLA
// The Neuron Runtime V2 API is plain C (__BEGIN_DECLS) and comes from the
// SDK's per-chip include directory. The struct that crosses the ABI is
// IOBuffer (plus the SyncInferenceRequest that carries it) — small enough that
// reproducing them by hand looked tempting, but a vendor struct that grows a
// field would then fail silently, which is exactly why the MTK shim header is
// taken from the SDK too.
//
// V2 rather than V1 (NeuronRuntime_*): the request carries the buffer
// descriptors on EVERY call, which is the contract-correct shape for inputs
// that change between runs, and the variant already validated in production
// on this device family. V1's set-once-then-infer latched the input at the
// first inference on runtime 7.3.15 (tools/dla_raw_probe.cc).
#include "neuron/api/RuntimeV2.h"
#endif

namespace mie {
namespace internal {
namespace {

void SetError(std::string* error, const std::string& message) {
  if (error != nullptr) *error = message;
}

#if MIE_ENABLE_MTK_DLA

// Every entry point this backend uses, resolved once. dlopen rather than link,
// for the same two reasons as QNN: a host build must not need a vendor .so,
// and a modern vendor library expects newer libc symbols than NDK r16b's stub
// can express. All are NeuronRuntimeV2_* exports of the same library the V1
// family lives in.
struct DlaApi {
  int (*create)(const char*, std::size_t, void**, std::size_t);
  int (*create_from_buffer)(const void*, std::size_t, std::size_t, void**,
                            std::size_t);
  int (*input_count)(void*, std::size_t*);
  int (*output_count)(void*, std::size_t*);
  int (*input_size)(void*, uint64_t, std::size_t*);
  int (*output_size)(void*, uint64_t, std::size_t*);
  int (*input_rank)(void*, uint64_t, uint32_t*);
  int (*output_rank)(void*, uint64_t, uint32_t*);
  int (*input_dims)(void*, uint64_t, RuntimeAPIDimensions*);
  int (*output_dims)(void*, uint64_t, RuntimeAPIDimensions*);
  int (*run)(void*, SyncInferenceRequest);
  void (*release)(void*);
};

// Resolves one symbol and records the first miss. Casting the void* through
// the destination member's own type keeps the signatures in one place.
struct SymbolLoader {
  LibHandle lib;
  std::string* error;
  bool ok;

  template <typename Fn>
  Fn Get(const char* name) {
    void* symbol = DlSym(lib, name);
    if (symbol == nullptr) {
      ok = false;
      if (error != nullptr) {
        *error = std::string("Neuron Runtime does not export ") + name;
      }
    }
    return reinterpret_cast<Fn>(symbol);
  }
};

// Runs a .dla through the Neuron Runtime V2. Like kMtk — and unlike
// CPU/GPU/QNN — the tensor API is not a raw tensor view: every request
// carries an IOBuffer (pointer + length) per tensor, so Input()/Output() are
// backed by engine-owned staging buffers.
class MtkDlaEngine : public Engine {
 public:
  ~MtkDlaEngine() override {
    if (runtime_ != nullptr) api_.release(runtime_);
    DlClose(lib_);
  }

  static std::unique_ptr<MtkDlaEngine> Build(const Config& cfg,
                                             const ModelSource& model,
                                             std::string* error) {
    std::unique_ptr<MtkDlaEngine> engine(new MtkDlaEngine());
    if (!engine->Init(cfg, model, error)) return nullptr;
    return engine;
  }

  int NumInputs() const override { return static_cast<int>(inputs_.size()); }
  int NumOutputs() const override { return static_cast<int>(outputs_.size()); }
  const Tensor& Input(int index) const override {
    return TensorAt(inputs_, index, "Input");
  }
  const Tensor& Output(int index) const override {
    return TensorAt(outputs_, index, "Output");
  }

  // Resizing is a compile-time property of a .dla: the shapes were baked in by
  // ncc-tflite. The runtime can only re-shape models compiled with
  // --runtime-dynamic-shape, which this backend does not expose.
  bool ResizeInput(int index, const std::vector<int>& shape) override {
    (void)index;
    (void)shape;
    return false;
  }

  // V2's request carries the IOBuffer descriptors on every call, so the
  // buffers the runtime reads are whatever this request points at — the
  // contract-correct shape for inputs that change between runs. The
  // descriptors point at the engine-owned staging buffers, which never move;
  // the caller rewrites Input(i).data and the next Run() submits it again.
  // tools/dla_stale_probe.cc is the standing check that the runtime actually
  // honors a re-submitted buffer (the 7.3.15 V1 family did not; see the
  // README's MediaTek DLA section).
  bool Run() override {
    SyncInferenceRequest request;
    request.inputs = input_buffers_.empty() ? nullptr : input_buffers_.data();
    request.outputs = output_buffers_.empty() ? nullptr : output_buffers_.data();
    return api_.run(runtime_, request) == NEURONRUNTIME_NO_ERROR;
  }

 private:
  MtkDlaEngine() = default;

  bool Init(const Config& cfg, const ModelSource& model, std::string* error) {
    const std::string* override_lib = FindOption(cfg.options, opt::kDlaLibrary);
    library_ = override_lib != nullptr ? *override_lib : "libneuron_runtime.so";

    lib_ = DlOpen(library_.c_str());
    if (lib_ == nullptr) {
      SetError(error, "failed to dlopen " + library_ +
                          " (set " + std::string(opt::kDlaLibrary) +
                          " to the SDK's libneuron_runtime.so if the device "
                          "ships a different one)");
      return false;
    }

    SymbolLoader loader;
    loader.lib = lib_;
    loader.error = error;
    loader.ok = true;
    api_.create =
        loader.Get<decltype(api_.create)>("NeuronRuntimeV2_create");
    api_.create_from_buffer = loader.Get<decltype(api_.create_from_buffer)>(
        "NeuronRuntimeV2_createFromBuffer");
    api_.input_count = loader.Get<decltype(api_.input_count)>(
        "NeuronRuntimeV2_getInputNumber");
    api_.output_count = loader.Get<decltype(api_.output_count)>(
        "NeuronRuntimeV2_getOutputNumber");
    api_.input_size =
        loader.Get<decltype(api_.input_size)>("NeuronRuntimeV2_getInputSize");
    api_.output_size = loader.Get<decltype(api_.output_size)>(
        "NeuronRuntimeV2_getOutputSize");
    api_.input_rank =
        loader.Get<decltype(api_.input_rank)>("NeuronRuntimeV2_getInputRank");
    api_.output_rank = loader.Get<decltype(api_.output_rank)>(
        "NeuronRuntimeV2_getOutputRank");
    api_.input_dims = loader.Get<decltype(api_.input_dims)>(
        "NeuronRuntimeV2_getInputPaddedDimensions");
    api_.output_dims = loader.Get<decltype(api_.output_dims)>(
        "NeuronRuntimeV2_getOutputPaddedDimensions");
    api_.run = loader.Get<decltype(api_.run)>("NeuronRuntimeV2_run");
    api_.release =
        loader.Get<decltype(api_.release)>("NeuronRuntimeV2_release");
    if (!loader.ok) return false;

    // One worker thread: Run() is synchronous, and each thread owns its own
    // working buffer, so more threads would only add footprint. The backlog is
    // the runtime's ring-buffer capacity; the header recommends 2048.
    const std::size_t kThreads = 1;
    const std::size_t kBacklog = 2048;
    int status = model.is_buffer()
                     ? api_.create_from_buffer(model.data(), model.size(),
                                               kThreads, &runtime_, kBacklog)
                     : api_.create(model.path().c_str(), kThreads, &runtime_,
                                   kBacklog);
    if (status != NEURONRUNTIME_NO_ERROR || runtime_ == nullptr) {
      SetError(error, "NeuronRuntimeV2_create failed (status " +
                          std::to_string(status) +
                          "); the .dla must come from an ncc-tflite with the "
                          "same MAJOR version as this runtime, and the device "
                          "needs its APU enabled");
      return false;
    }

    if (!ReadTensors(true, &inputs_, &input_storage_, error)) return false;
    if (!ReadTensors(false, &outputs_, &output_storage_, error)) return false;

    // The request descriptors are built once and point at the staging
    // buffers, which never move (ResizeInput is not supported). fd = -1 marks
    // a plain host buffer, per the IOBuffer contract. IOBuffer has no default
    // constructor, so the vectors grow by push_back rather than resize.
    input_buffers_.reserve(inputs_.size());
    for (std::size_t i = 0; i < inputs_.size(); ++i) {
      input_buffers_.push_back(
          IOBuffer(inputs_[i].data, inputs_[i].bytes, NON_ION_FD));
    }
    output_buffers_.reserve(outputs_.size());
    for (std::size_t i = 0; i < outputs_.size(); ++i) {
      output_buffers_.push_back(
          IOBuffer(outputs_[i].data, outputs_[i].bytes, NON_ION_FD));
    }
    return true;
  }

  bool ReadTensors(bool is_input, std::vector<Tensor>* tensors,
                   std::vector<std::vector<uint8_t> >* storage,
                   std::string* error) {
    std::size_t count = 0;
    int status = is_input ? api_.input_count(runtime_, &count)
                          : api_.output_count(runtime_, &count);
    if (status != NEURONRUNTIME_NO_ERROR) {
      SetError(error, is_input ? "failed to read the input count"
                               : "failed to read the output count");
      return false;
    }
    tensors->resize(count);
    storage->resize(count);

    for (std::size_t i = 0; i < count; ++i) {
      const uint64_t handle = static_cast<uint64_t>(i);
      Tensor& tensor = (*tensors)[i];

      uint32_t rank = 0;
      status = is_input ? api_.input_rank(runtime_, handle, &rank)
                        : api_.output_rank(runtime_, handle, &rank);
      if (status != NEURONRUNTIME_NO_ERROR) {
        SetError(error, "failed to read a tensor rank");
        return false;
      }
      RuntimeAPIDimensions dimensions;
      std::memset(&dimensions, 0, sizeof(dimensions));
      status = is_input ? api_.input_dims(runtime_, handle, &dimensions)
                        : api_.output_dims(runtime_, handle, &dimensions);
      if (status != NEURONRUNTIME_NO_ERROR) {
        SetError(error, "failed to read tensor dimensions");
        return false;
      }
      if (rank > kDimensionSize) rank = kDimensionSize;
      tensor.shape.assign(dimensions.dimensions,
                          dimensions.dimensions + rank);

      std::size_t bytes = 0;
      status = is_input ? api_.input_size(runtime_, handle, &bytes)
                        : api_.output_size(runtime_, handle, &bytes);
      if (status != NEURONRUNTIME_NO_ERROR) {
        SetError(error, "failed to read a tensor size");
        return false;
      }

      // The runtime reports shapes and sizes but not element types, and the
      // Neuron Runtime API has no getter for them. All models the offline
      // compiler accepts in the default (converting) mode exchange FLOAT32
      // with the caller, which is what is reported here. bytes always comes
      // from the runtime, so it stays exact regardless.
      tensor.type = Type::kFloat32;
      tensor.bytes = bytes;

      std::vector<uint8_t>& buffer = (*storage)[i];
      buffer.assign(bytes, 0);
      tensor.data = buffer.empty() ? nullptr : buffer.data();
    }
    return true;
  }

  // Declared first so it is closed last: the function pointers and the runtime
  // handle both outlive it, and the destructor body releases the runtime before
  // any member dies.
  LibHandle lib_ = nullptr;
  std::string library_;
  DlaApi api_ = {};
  void* runtime_ = nullptr;

  // Request descriptors, pointing into the storage below.
  std::vector<IOBuffer> input_buffers_;
  std::vector<IOBuffer> output_buffers_;

  // Storage is declared before the views that point into it.
  std::vector<std::vector<uint8_t> > input_storage_;
  std::vector<std::vector<uint8_t> > output_storage_;
  std::vector<Tensor> inputs_;
  std::vector<Tensor> outputs_;
};

#endif  // MIE_ENABLE_MTK_DLA

}  // namespace

std::unique_ptr<Engine> BuildMtkDla(const Config& cfg, const ModelSource& model,
                                    std::string* error) {
#if !MIE_ENABLE_MTK_DLA
  (void)cfg;
  (void)model;
  SetError(error,
           "MediaTek DLA backend not compiled in (configure with "
           "-DMIE_ENABLE_MTK_DLA=ON and point MIE_MTK_SDK_INCLUDE_DIR at the "
           "directory holding neuron/api/RuntimeV2.h)");
  return nullptr;
#else
  return MtkDlaEngine::Build(cfg, model, error);
#endif
}

}  // namespace internal
}  // namespace mie
