#include "src/backend_mtk.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "src/fingerprint.h"
#include "src/options.h"
#include "src/tensor_view.h"

#if MIE_ENABLE_MTK
// NeuroPilotTFLiteShim.h uses NNAPI's constants (ANEURALNETWORKS_*) and needs
// ANeuralNetworksModel declared, but does NOT include the NNAPI header itself —
// the consumer must, and before this include.
//
// That header exists in the unified sysroot at every API level, but its CONTENTS
// are gated on __ANDROID_API__, so below 27 it is an empty file and the vendor
// header then fails with "unknown type name 'ANeuralNetworksModel'" — an error
// that points nowhere useful. Catch the real cause here, in a way that works
// under ndk-build and CMake alike (and on any host OS, unlike a $(shell) test).
#if !defined(__ANDROID_API__) || (__ANDROID_API__ < 27)
#error \
    "MIE_ENABLE_MTK=1 requires API level 27 or newer: build with APP_PLATFORM=android-27 (ndk-build) or -DANDROID_API=27 (CMake)."
#endif
#include <android/NeuralNetworks.h>
#include "NeuroPilotTFLiteShim.h"
#endif

namespace mie {
namespace internal {
namespace {

void SetError(std::string* error, const std::string& message) {
  if (error != nullptr) *error = message;
}

#if MIE_ENABLE_MTK

Type MapType(TFLiteTensorType type) {
  switch (type) {
    case TFLITE_TENSOR_TYPE_FLOAT:
      return Type::kFloat32;
    case TFLITE_TENSOR_TYPE_FLOAT16:
      return Type::kFloat16;
    case TFLITE_TENSOR_TYPE_INT32:
      return Type::kInt32;
    case TFLITE_TENSOR_TYPE_INT64:
      return Type::kInt64;
    case TFLITE_TENSOR_TYPE_UINT8:
      return Type::kUInt8;
    case TFLITE_TENSOR_TYPE_INT8:
      return Type::kInt8;
    case TFLITE_TENSOR_TYPE_BOOL:
      return Type::kBool;
    default:
      return Type::kUnknown;
  }
}

// The shim only accepts copied buffers (setInputTensorData / getOutputTensorData
// both take data+size), so the zero-copy view the public API promises is backed
// by an engine-owned staging buffer here. Run() copies in and out. That is a
// property of the NeuroPilot API, not a choice made in this file.
class MtkEngine : public Engine {
 public:
  ~MtkEngine() override {
    if (tflite_ != nullptr) ANeuroPilotTFLiteWrapper_free(tflite_);
    if (options_ != nullptr) ANeuralNetworksTFLiteOptions_free(options_);
  }

  static std::unique_ptr<MtkEngine> Build(const Config& cfg,
                                          const ModelSource& model,
                                          std::string* error) {
    std::unique_ptr<MtkEngine> engine(new MtkEngine());
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

  // The shim's resize is a creation-time option, not a runtime call
  // (ANeuralNetworksTFLiteOptions_resizeInputTensor), so there is nothing safe
  // to do here. Rebuild the engine with the shape applied as an option.
  bool ResizeInput(int index, const std::vector<int>& shape) override {
    (void)index;
    (void)shape;
    return false;
  }

  bool Run() override {
    for (std::size_t i = 0; i < inputs_.size(); ++i) {
      if (ANeuroPilotTFLiteWrapper_setInputTensorData(
              tflite_, static_cast<int>(i), inputs_[i].data,
              inputs_[i].bytes) != ANEURALNETWORKS_NO_ERROR) {
        return false;
      }
    }
    if (ANeuroPilotTFLiteWrapper_invoke(tflite_) != ANEURALNETWORKS_NO_ERROR) {
      return false;
    }
    for (std::size_t i = 0; i < outputs_.size(); ++i) {
      if (ANeuroPilotTFLiteWrapper_getOutputTensorData(
              tflite_, static_cast<int>(i), outputs_[i].data,
              outputs_[i].bytes) != ANEURALNETWORKS_NO_ERROR) {
        return false;
      }
    }
    return true;
  }

 private:
  MtkEngine() = default;

  bool Init(const Config& cfg, const ModelSource& model, std::string* error) {
    if (ANeuralNetworksTFLiteOptions_create(&options_) !=
            ANEURALNETWORKS_NO_ERROR ||
        options_ == nullptr) {
      SetError(error, "ANeuralNetworksTFLiteOptions_create failed");
      return false;
    }

    // kMtk means "the MediaTek NPU", so require the Neuron accelerator instead
    // of letting the runtime quietly fall back to CPU or NNAPI.
    ANeuralNetworksTFLiteOptions_setAccelerationMode(options_,
                                                     NP_ACCELERATION_NEURON);

    if (!cfg.cache_dir.empty()) {
      // Copy first, then hand out the pointer: the shim's setters take a
      // `const char*` and we cannot see whether they retain it, so backing every
      // one of them with engine-owned storage removes the question. Passing
      // cfg.cache_dir / cfg.options straight through would dangle as soon as the
      // caller's Config died.
      cache_dir_ = cfg.cache_dir;
      ANeuralNetworksTFLiteOptions_setCacheDir(options_, cache_dir_.c_str());
    }
    ApplyOptions(cfg);

    // The shim duplicates the buffer internally, so unlike the TFLite backends
    // nothing here depends on the caller keeping model.data() alive — but the
    // documented contract is the same for every backend, so callers can switch
    // backends without re-reading the rules.
    const int status =
        model.is_buffer()
            ? ANeuroPilotTFLiteWrapper_makeAdvTFLiteWithBuffer(
                  &tflite_, static_cast<const char*>(model.data()),
                  model.size(), options_)
            : ANeuroPilotTFLiteWrapper_makeAdvTFLite(&tflite_,
                                                     model.path().c_str(),
                                                     options_);
    if (status != ANEURALNETWORKS_NO_ERROR || tflite_ == nullptr) {
      SetError(error,
               "NeuroPilot could not create an interpreter (is this a MediaTek "
               "device with libtflite_mtk.so?)");
      return false;
    }

    if (!ReadTensors(TFLITE_BUFFER_TYPE_INPUT, &inputs_, &input_storage_)) {
      SetError(error, "failed to read input tensor metadata");
      return false;
    }
    if (!ReadTensors(TFLITE_BUFFER_TYPE_OUTPUT, &outputs_, &output_storage_)) {
      SetError(error, "failed to read output tensor metadata");
      return false;
    }
    return true;
  }

  void ApplyOptions(const Config& cfg) {
    if (const std::string* v =
            FindOption(cfg.options, opt::kExecutionPreference)) {
      if (*v == "low_power") {
        ANeuralNetworksTFLiteOptions_setPreference(options_, kLowPower);
      } else if (*v == "fast_single_answer") {
        ANeuralNetworksTFLiteOptions_setPreference(options_, kFastSingleAnswer);
      } else if (*v == "sustained_speed") {
        ANeuralNetworksTFLiteOptions_setPreference(options_, kSustainedSpeed);
      }
    }
    if (const std::string* v = FindOption(cfg.options, opt::kExecutionPriority)) {
      ANeuralNetworksTFLiteOptions_setExecutionPriority(options_,
                                                        std::atoi(v->c_str()));
    }
    if (const std::string* v = FindOption(cfg.options, opt::kAllowFp16)) {
      ANeuralNetworksTFLiteOptions_setAllowFp16PrecisionForFp32(options_,
                                                                IsTruthy(*v));
    }
    if (const std::string* v = FindOption(cfg.options, opt::kAcceleratorName)) {
      accelerator_name_ = *v;
      ANeuralNetworksTFLiteOptions_setAcceleratorName(options_,
                                                      accelerator_name_.c_str());
    }
    if (const std::string* v = FindOption(cfg.options, opt::kLowLatency)) {
      ANeuralNetworksTFLiteOptions_setLowLatency(options_, IsTruthy(*v));
    }
    if (const std::string* v = FindOption(cfg.options, opt::kDeepFusion)) {
      ANeuralNetworksTFLiteOptions_setDeepFusion(options_, IsTruthy(*v));
    }
    if (const std::string* v = FindOption(cfg.options, opt::kBatchProcessing)) {
      ANeuralNetworksTFLiteOptions_setBatchProcessing(options_, IsTruthy(*v));
    }
    if (const std::string* v = FindOption(cfg.options, opt::kUseIon)) {
      ANeuralNetworksTFLiteOptions_setUseIon(options_, IsTruthy(*v));
    }
  }

  bool ReadTensors(TFLiteBufferType btype, std::vector<Tensor>* tensors,
                   std::vector<std::vector<uint8_t> >* storage) {
    int32_t count = 0;
    if (ANeuroPilotTFLiteWrapper_getTensorCount(tflite_, btype, &count) !=
        ANEURALNETWORKS_NO_ERROR) {
      return false;
    }
    if (count < 0) return false;
    tensors->resize(static_cast<std::size_t>(count));
    storage->resize(static_cast<std::size_t>(count));

    for (int32_t i = 0; i < count; ++i) {
      const std::size_t idx = static_cast<std::size_t>(i);
      Tensor& tensor = (*tensors)[idx];

      int rank = 0;
      if (ANeuroPilotTFLiteWrapper_getTensorRank(tflite_, btype, i, &rank) !=
          ANEURALNETWORKS_NO_ERROR) {
        return false;
      }
      if (rank > 0) {
        tensor.shape.assign(static_cast<std::size_t>(rank), 0);
        if (ANeuroPilotTFLiteWrapper_getTensorDimensions(
                tflite_, btype, i, tensor.shape.data()) !=
            ANEURALNETWORKS_NO_ERROR) {
          return false;
        }
      }

      std::size_t bytes = 0;
      if (ANeuroPilotTFLiteWrapper_getTensorByteSize(tflite_, btype, i,
                                                     &bytes) !=
          ANEURALNETWORKS_NO_ERROR) {
        return false;
      }

      TFLiteTensorType shim_type = TFLITE_TENSOR_TYPE_NONE;
      if (ANeuroPilotTFLiteWrapper_getTensorType(tflite_, btype, i,
                                                 &shim_type) !=
          ANEURALNETWORKS_NO_ERROR) {
        return false;
      }

      tensor.type = MapType(shim_type);
      tensor.bytes = bytes;

      // Staging buffer. storage is sized up-front and never grows again, so the
      // pointers handed out below stay valid for the engine's lifetime.
      std::vector<uint8_t>& buffer = (*storage)[idx];
      buffer.assign(bytes, 0);
      tensor.data = buffer.empty() ? nullptr : buffer.data();
    }
    return true;
  }

  ANeuralNetworksTFLite* tflite_ = nullptr;
  ANeuralNetworksTFLiteOptions* options_ = nullptr;

  // Engine-owned storage for every `const char*` handed to the shim's options,
  // declared before the options handle is used so it outlives it. Do not pass
  // pointers that belong to the caller's Config to a vendor setter.
  std::string cache_dir_;
  std::string accelerator_name_;

  // Storage is declared before the views that point into it.
  std::vector<std::vector<uint8_t> > input_storage_;
  std::vector<std::vector<uint8_t> > output_storage_;
  std::vector<Tensor> inputs_;
  std::vector<Tensor> outputs_;
};

#endif  // MIE_ENABLE_MTK

}  // namespace

std::unique_ptr<Engine> BuildMtk(const Config& cfg, const ModelSource& model,
                                 std::string* error) {
#if !MIE_ENABLE_MTK
  (void)cfg;
  (void)model;
  SetError(error,
           "MediaTek backend not compiled in (configure with -DMIE_ENABLE_MTK=ON "
           "and point MIE_MTK_INCLUDE_DIR at the directory holding "
           "NeuroPilotTFLiteShim.h)");
  return nullptr;
#else
  return MtkEngine::Build(cfg, model, error);
#endif
}

}  // namespace internal
}  // namespace mie
