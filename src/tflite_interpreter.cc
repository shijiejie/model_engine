// The one concrete engine behind kCpu / kGpu / kQnn: a TFLite interpreter plus
// an optional delegate.
//
// The class lives entirely in this translation unit — backend_cpu/gpu/qnn only
// see BuildTfliteEngine(). The TFLite C API is used deliberately: its headers
// are plain C (so no C++17 requirement leaks in, and this builds with NDK r16b),
// it is ABI-stable (so a runtime built by any NDK links), and it still exposes
// the interpreter's own tensor buffers, which is what keeps the hot path
// zero-copy.
#include "src/tflite_interpreter.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "tensorflow/lite/c/c_api.h"

#include "src/tensor_view.h"

namespace mie {
namespace internal {
namespace {

void SetError(std::string* error, std::string message) {
  if (error != nullptr) *error = std::move(message);
}

struct ModelDeleter {
  void operator()(TfLiteModel* model) const { TfLiteModelDelete(model); }
};

struct InterpreterDeleter {
  void operator()(TfLiteInterpreter* interpreter) const {
    TfLiteInterpreterDelete(interpreter);
  }
};

Type MapType(TfLiteType type) {
  switch (type) {
    case kTfLiteFloat32:
      return Type::kFloat32;
    case kTfLiteFloat16:
      return Type::kFloat16;
    case kTfLiteInt32:
      return Type::kInt32;
    case kTfLiteInt64:
      return Type::kInt64;
    case kTfLiteUInt8:
      return Type::kUInt8;
    case kTfLiteInt8:
      return Type::kInt8;
    case kTfLiteBool:
      return Type::kBool;
    default:
      return Type::kUnknown;
  }
}

Tensor MakeTensor(const TfLiteTensor* tensor) {
  Tensor out;
  if (tensor == nullptr) return out;

  const char* name = TfLiteTensorName(tensor);
  if (name != nullptr) out.name = name;

  out.type = MapType(TfLiteTensorType(tensor));

  const int32_t dims = TfLiteTensorNumDims(tensor);
  if (dims > 0) out.shape.reserve(static_cast<std::size_t>(dims));
  for (int32_t i = 0; i < dims; ++i) {
    out.shape.push_back(TfLiteTensorDim(tensor, i));
  }

  out.bytes = TfLiteTensorByteSize(tensor);
  // Points straight at interpreter-owned storage: no copy on the hot path.
  out.data = TfLiteTensorData(tensor);
  return out;
}

class TfliteEngine : public Engine {
 public:
  static std::unique_ptr<TfliteEngine> Build(const Config& cfg,
                                             const ModelSource& source,
                                             DelegateHandle delegate,
                                             std::string* error);

  int NumInputs() const override { return static_cast<int>(inputs_.size()); }
  int NumOutputs() const override { return static_cast<int>(outputs_.size()); }
  const Tensor& Input(int index) const override {
    return TensorAt(inputs_, index, "Input");
  }
  const Tensor& Output(int index) const override {
    return TensorAt(outputs_, index, "Output");
  }
  bool ResizeInput(int index, const std::vector<int>& shape) override;
  bool Run() override {
    return TfLiteInterpreterInvoke(interpreter_.get()) == kTfLiteOk;
  }

 private:
  TfliteEngine() = default;

  // Rebuilds the zero-copy input/output views from the interpreter.
  void RefreshTensors();

  // Declaration order is load-bearing: members are destroyed in reverse, which
  // yields interpreter -> delegate -> model. The interpreter must die before
  // the delegate it borrowed, and before the model buffer it points into.
  std::unique_ptr<TfLiteModel, ModelDeleter> model_;
  DelegateHandle delegate_;
  std::unique_ptr<TfLiteInterpreter, InterpreterDeleter> interpreter_;

  std::vector<Tensor> inputs_;
  std::vector<Tensor> outputs_;
};

std::unique_ptr<TfliteEngine> TfliteEngine::Build(const Config& cfg,
                                                  const ModelSource& source,
                                                  DelegateHandle delegate,
                                                  std::string* error) {
  if (source.empty()) {
    SetError(error, "no model given: set Config::model_path or Config::model_data");
    return nullptr;
  }

  // TfLiteModelCreate does NOT copy; it keeps pointing into the caller's bytes,
  // which is why Config::model_data must outlive the Engine.
  TfLiteModel* model =
      source.is_buffer()
          ? TfLiteModelCreate(source.data(), source.size())
          : TfLiteModelCreateFromFile(source.path().c_str());
  if (model == nullptr) {
    SetError(error, source.is_buffer()
                        ? std::string("failed to parse the in-memory model buffer")
                        : "failed to load model: " + source.path());
    return nullptr;
  }

  std::unique_ptr<TfliteEngine> engine(new TfliteEngine());
  engine->model_.reset(model);
  engine->delegate_ = std::move(delegate);

  TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
  if (options == nullptr) {
    SetError(error, "TfLiteInterpreterOptionsCreate failed");
    return nullptr;
  }
  if (cfg.backend == Backend::kCpu) {
    TfLiteInterpreterOptionsSetNumThreads(options, cfg.num_threads);
  }
  if (!engine->delegate_.empty()) {
    // Delegates are attached here; the C API applies them while creating the
    // interpreter, so there is no separate ModifyGraphWithDelegate step.
    TfLiteInterpreterOptionsAddDelegate(options, engine->delegate_.get());
  }

  TfLiteInterpreter* interpreter =
      TfLiteInterpreterCreate(engine->model_.get(), options);
  // Per the C API contract the options are only needed for creation.
  TfLiteInterpreterOptionsDelete(options);
  if (interpreter == nullptr) {
    SetError(error, "TfLiteInterpreterCreate failed");
    return nullptr;
  }
  engine->interpreter_.reset(interpreter);

  if (TfLiteInterpreterAllocateTensors(engine->interpreter_.get()) !=
      kTfLiteOk) {
    SetError(error, "TfLiteInterpreterAllocateTensors failed");
    return nullptr;
  }

  engine->RefreshTensors();
  return engine;
}

void TfliteEngine::RefreshTensors() {
  inputs_.clear();
  outputs_.clear();

  const int32_t num_inputs =
      TfLiteInterpreterGetInputTensorCount(interpreter_.get());
  const int32_t num_outputs =
      TfLiteInterpreterGetOutputTensorCount(interpreter_.get());
  if (num_inputs > 0) inputs_.reserve(static_cast<std::size_t>(num_inputs));
  if (num_outputs > 0) outputs_.reserve(static_cast<std::size_t>(num_outputs));

  for (int32_t i = 0; i < num_inputs; ++i) {
    inputs_.push_back(
        MakeTensor(TfLiteInterpreterGetInputTensor(interpreter_.get(), i)));
  }
  for (int32_t i = 0; i < num_outputs; ++i) {
    outputs_.push_back(
        MakeTensor(TfLiteInterpreterGetOutputTensor(interpreter_.get(), i)));
  }
}

bool TfliteEngine::ResizeInput(int index, const std::vector<int>& shape) {
  if (index < 0 || index >= static_cast<int>(inputs_.size())) return false;
  if (shape.empty()) return false;

  if (TfLiteInterpreterResizeInputTensor(interpreter_.get(), index, shape.data(),
                                         static_cast<int32_t>(shape.size())) !=
      kTfLiteOk) {
    return false;
  }
  if (TfLiteInterpreterAllocateTensors(interpreter_.get()) != kTfLiteOk) {
    return false;
  }
  RefreshTensors();
  return true;
}

}  // namespace

std::unique_ptr<Engine> BuildTfliteEngine(const Config& cfg,
                                          const ModelSource& model,
                                          DelegateHandle delegate,
                                          std::string* error) {
  return TfliteEngine::Build(cfg, model, std::move(delegate), error);
}

}  // namespace internal
}  // namespace mie
