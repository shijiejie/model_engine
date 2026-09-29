#include "src/backend_rknn.h"

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#ifndef MIE_ENABLE_RKNN
#define MIE_ENABLE_RKNN 0
#endif

#if MIE_ENABLE_RKNN
#include "rknn_api.h"
#endif

#include "src/log.h"
#include "src/options.h"
#include "src/tensor_view.h"

namespace mie {
namespace internal {
namespace {

void SetError(std::string* error, const std::string& message) {
  if (error != nullptr) *error = message;
}

#if MIE_ENABLE_RKNN

std::size_t TypeSize(rknn_tensor_type type) {
  switch (type) {
    case RKNN_TENSOR_FLOAT32:
    case RKNN_TENSOR_INT32:
    case RKNN_TENSOR_UINT32:
      return 4;
    case RKNN_TENSOR_FLOAT16:
    case RKNN_TENSOR_INT16:
    case RKNN_TENSOR_UINT16:
    case RKNN_TENSOR_BFLOAT16:
      return 2;
    case RKNN_TENSOR_INT8:
    case RKNN_TENSOR_UINT8:
    case RKNN_TENSOR_BOOL:
      return 1;
    case RKNN_TENSOR_INT64:
      return 8;
    default:
      return 0;
  }
}

Type MapType(rknn_tensor_type type) {
  switch (type) {
    case RKNN_TENSOR_FLOAT32:
      return Type::kFloat32;
    case RKNN_TENSOR_FLOAT16:
      return Type::kFloat16;
    case RKNN_TENSOR_INT32:
      return Type::kInt32;
    case RKNN_TENSOR_INT64:
      return Type::kInt64;
    case RKNN_TENSOR_UINT8:
      return Type::kUInt8;
    case RKNN_TENSOR_INT8:
      return Type::kInt8;
    case RKNN_TENSOR_BOOL:
      return Type::kBool;
    default:
      return Type::kUnknown;
  }
}

bool IsValidCoreMask(unsigned long value) {
  return value == 0 || value == RKNN_NPU_CORE_0 || value == RKNN_NPU_CORE_1 ||
         value == RKNN_NPU_CORE_2 || value == RKNN_NPU_CORE_0_1 ||
         value == RKNN_NPU_CORE_0_1_2 || value == RKNN_NPU_CORE_ALL;
}

bool ParseCoreMask(const std::string& text, rknn_core_mask* mask) {
  if (text.empty() || mask == nullptr) return false;
  unsigned long value = 0;
  if (text == "auto") {
    value = RKNN_NPU_CORE_AUTO;
  } else if (text == "core0") {
    value = RKNN_NPU_CORE_0;
  } else if (text == "core1") {
    value = RKNN_NPU_CORE_1;
  } else if (text == "core2") {
    value = RKNN_NPU_CORE_2;
  } else if (text == "core0_1") {
    value = RKNN_NPU_CORE_0_1;
  } else if (text == "core0_1_2") {
    value = RKNN_NPU_CORE_0_1_2;
  } else if (text == "all") {
    value = RKNN_NPU_CORE_ALL;
  } else {
    errno = 0;
    char* end = nullptr;
    value = std::strtoul(text.c_str(), &end, 0);
    if (errno != 0 || end == text.c_str() || *end != '\0') return false;
  }
  if (!IsValidCoreMask(value)) return false;
  *mask = static_cast<rknn_core_mask>(value);
  return true;
}

bool ParseBool(const std::string& text, bool* value) {
  if (value == nullptr) return false;
  if (text == "true" || text == "1") {
    *value = true;
    return true;
  }
  if (text == "false" || text == "0") {
    *value = false;
    return true;
  }
  return false;
}

bool ReadBoolOption(const Config& cfg, const char* key, bool default_value,
                    bool* value, std::string* error) {
  *value = default_value;
  const std::string* option = FindOption(cfg.options, key);
  if (option == nullptr) return true;
  if (ParseBool(*option, value)) return true;
  SetError(error, std::string("invalid ") + key + ": " + *option);
  return false;
}

std::string TensorName(const rknn_tensor_attr& attr) {
  const void* terminator = std::memchr(attr.name, '\0', RKNN_MAX_NAME_LEN);
  const std::size_t length =
      terminator == nullptr
          ? static_cast<std::size_t>(RKNN_MAX_NAME_LEN)
          : static_cast<const char*>(terminator) - attr.name;
  return std::string(attr.name, length);
}

bool ReadFile(const std::string& path, std::vector<unsigned char>* bytes,
              std::string* error) {
  std::FILE* file = std::fopen(path.c_str(), "rb");
  if (file == nullptr) {
    SetError(error, "cannot open RKNN model: " + path);
    return false;
  }
  if (std::fseek(file, 0, SEEK_END) != 0) {
    std::fclose(file);
    SetError(error, "cannot seek RKNN model: " + path);
    return false;
  }
  const long size = std::ftell(file);
  if (size <= 0 ||
      static_cast<unsigned long long>(size) >
          std::numeric_limits<uint32_t>::max() ||
      std::fseek(file, 0, SEEK_SET) != 0) {
    std::fclose(file);
    SetError(error, "invalid RKNN model size: " + path);
    return false;
  }
  bytes->resize(static_cast<std::size_t>(size));
  const std::size_t got = std::fread(bytes->data(), 1, bytes->size(), file);
  const bool ok = got == bytes->size() && std::ferror(file) == 0;
  std::fclose(file);
  if (!ok) {
    SetError(error, "failed to read RKNN model: " + path);
    return false;
  }
  return true;
}

bool ReadElements(const rknn_tensor_attr& attr, std::size_t* elements,
                  std::string* error) {
  if (attr.n_dims == 0 || attr.n_dims > RKNN_MAX_DIMS) {
    SetError(error, "RKNN tensor has an invalid rank");
    return false;
  }
  std::size_t product = 1;
  for (uint32_t i = 0; i < attr.n_dims; ++i) {
    if (attr.dims[i] == 0 ||
        product > std::numeric_limits<std::size_t>::max() / attr.dims[i]) {
      SetError(error, "RKNN tensor has an invalid or overflowing dimension");
      return false;
    }
    product *= attr.dims[i];
  }
  // n_elems is the runtime's authoritative logical element count. It can
  // differ from the raw dimension product for packed/native formats; use the
  // product only for older runtimes that leave n_elems unset.
  *elements = attr.n_elems != 0 ? static_cast<std::size_t>(attr.n_elems)
                                : product;
  return true;
}

bool ReadTensorBytes(const rknn_tensor_attr& attr, std::size_t elements,
                     std::size_t type_size, bool prefer_stride,
                     std::size_t* bytes, std::string* error) {
  const uint32_t reported_size =
      prefer_stride && attr.size_with_stride != 0 ? attr.size_with_stride
                                                  : attr.size;
  if (type_size == 0 ||
      elements > std::numeric_limits<std::size_t>::max() / type_size) {
    SetError(error, "RKNN tensor byte size overflows host size_t");
    return false;
  }
  const std::size_t logical_bytes = elements * type_size;
  if (reported_size != 0) {
    if (static_cast<std::size_t>(reported_size) < logical_bytes) {
      SetError(error, "RKNN tensor byte metadata is smaller than its shape");
      return false;
    }
    *bytes = static_cast<std::size_t>(reported_size);
    return true;
  }
  *bytes = logical_bytes;
  return true;
}

bool ReadInputShape(const rknn_tensor_attr& attr, bool pass_through,
                    std::vector<int>* shape, std::size_t* elements,
                    std::string* error) {
  if (!ReadElements(attr, elements, error)) return false;
  if (!pass_through &&
      (attr.n_dims != 4 ||
       (attr.fmt != RKNN_TENSOR_NCHW && attr.fmt != RKNN_TENSOR_NHWC))) {
    SetError(error, "RKNN conversion mode requires rank-4 NCHW/NHWC inputs");
    return false;
  }
  shape->clear();
  shape->reserve(attr.n_dims);
  for (uint32_t i = 0; i < attr.n_dims; ++i) {
    if (attr.dims[i] > static_cast<uint32_t>(std::numeric_limits<int>::max())) {
      SetError(error, "RKNN input dimension exceeds MIE shape range");
      return false;
    }
  }
  if (pass_through || attr.fmt == RKNN_TENSOR_NHWC) {
    for (uint32_t i = 0; i < attr.n_dims; ++i) {
      shape->push_back(static_cast<int>(attr.dims[i]));
    }
  } else {
    shape->push_back(static_cast<int>(attr.dims[0]));
    shape->push_back(static_cast<int>(attr.dims[2]));
    shape->push_back(static_cast<int>(attr.dims[3]));
    shape->push_back(static_cast<int>(attr.dims[1]));
  }
  return true;
}

bool ReadOutputShape(const rknn_tensor_attr& attr, std::vector<int>* shape,
                     std::size_t* elements, std::string* error) {
  if (!ReadElements(attr, elements, error)) return false;
  shape->clear();
  shape->reserve(attr.n_dims);
  for (uint32_t i = 0; i < attr.n_dims; ++i) {
    if (attr.dims[i] > static_cast<uint32_t>(std::numeric_limits<int>::max())) {
      SetError(error, "RKNN output dimension exceeds MIE shape range");
      return false;
    }
    shape->push_back(static_cast<int>(attr.dims[i]));
  }
  return true;
}

struct TensorInfo {
  rknn_tensor_attr attr = {};
  std::size_t elements = 0;
};

class RknnEngine : public Engine {
 public:
  ~RknnEngine() override {
    if (ctx_ != 0) rknn_destroy(ctx_);
  }

  static std::unique_ptr<RknnEngine> Build(const Config& cfg,
                                           const ModelSource& model,
                                           std::string* error) {
    try {
      std::unique_ptr<RknnEngine> engine(new RknnEngine());
      if (!engine->Init(cfg, model, error)) return nullptr;
      return engine;
    } catch (const std::bad_alloc&) {
      SetError(error, "out of memory while creating RKNN engine");
      return nullptr;
    } catch (const std::exception& exception) {
      SetError(error, std::string("exception while creating RKNN engine: ") +
                           exception.what());
      return nullptr;
    }
  }

  int NumInputs() const override { return static_cast<int>(inputs_.size()); }
  int NumOutputs() const override { return static_cast<int>(outputs_.size()); }
  const Tensor& Input(int index) const override {
    return TensorAt(inputs_, index, "Input");
  }
  const Tensor& Output(int index) const override {
    return TensorAt(outputs_, index, "Output");
  }

  bool ResizeInput(int index, const std::vector<int>& shape) override {
    (void)index;
    (void)shape;
    return false;
  }

  bool Run() override {
    if (ctx_ == 0) return false;

    for (std::size_t i = 0; i < input_infos_.size(); ++i) {
      rknn_input& input = rknn_inputs_[i];
      std::memset(&input, 0, sizeof(input));
      input.index = static_cast<uint32_t>(i);
      input.buf = input_storage_[i].empty() ? nullptr : input_storage_[i].data();
      input.size = static_cast<uint32_t>(input_storage_[i].size());
      input.pass_through = pass_through_ ? 1 : 0;
      input.type = pass_through_ ? input_infos_[i].attr.type
                                 : RKNN_TENSOR_UINT8;
      input.fmt = pass_through_ ? input_infos_[i].attr.fmt
                                : RKNN_TENSOR_NHWC;
    }

    int ret = rknn_inputs_set(ctx_, static_cast<uint32_t>(rknn_inputs_.size()),
                              rknn_inputs_.data());
    if (ret != RKNN_SUCC) {
      MIE_LOGW("rknn_inputs_set failed: %d", ret);
      return false;
    }
    ret = rknn_run(ctx_, nullptr);
    if (ret != RKNN_SUCC) {
      MIE_LOGW("rknn_run failed: %d", ret);
      return false;
    }

    for (std::size_t i = 0; i < rknn_outputs_.size(); ++i) {
      std::memset(&rknn_outputs_[i], 0, sizeof(rknn_outputs_[i]));
      rknn_outputs_[i].index = static_cast<uint32_t>(i);
      rknn_outputs_[i].want_float = want_float_ ? 1 : 0;
      rknn_outputs_[i].is_prealloc = 0;
    }
    ret = rknn_outputs_get(ctx_, static_cast<uint32_t>(rknn_outputs_.size()),
                           rknn_outputs_.data(), nullptr);
    if (ret != RKNN_SUCC) {
      MIE_LOGW("rknn_outputs_get failed: %d", ret);
      // The RKNN contract only permits releasing buffers returned by a
      // successful rknn_outputs_get call. On failure the descriptors may be
      // only partially initialized, so calling release here can turn an
      // inference error into a runtime crash on some SDK versions.
      return false;
    }

    bool ok = true;
    for (std::size_t i = 0; i < rknn_outputs_.size(); ++i) {
      if (rknn_outputs_[i].buf == nullptr ||
          rknn_outputs_[i].size < output_storage_[i].size()) {
        MIE_LOGW("RKNN output[%d] is too small: got %u need %zu",
                 static_cast<int>(i), rknn_outputs_[i].size,
                 output_storage_[i].size());
        ok = false;
        continue;
      }
      if (!output_storage_[i].empty()) {
        std::memcpy(output_storage_[i].data(), rknn_outputs_[i].buf,
                    output_storage_[i].size());
      }
    }
    if (rknn_outputs_release(ctx_, static_cast<uint32_t>(rknn_outputs_.size()),
                             rknn_outputs_.data()) != RKNN_SUCC) {
      ok = false;
    }
    return ok;
  }

 private:
  RknnEngine() = default;

  bool Init(const Config& cfg, const ModelSource& model, std::string* error) {
    if (model.empty()) {
      SetError(error, "no RKNN model given");
      return false;
    }

    std::vector<unsigned char> file_bytes;
    const void* data = model.data();
    std::size_t size = model.size();
    if (!model.is_buffer()) {
      if (!ReadFile(model.path(), &file_bytes, error)) return false;
      data = file_bytes.data();
      size = file_bytes.size();
    }
    if (data == nullptr || size == 0 ||
        size > std::numeric_limits<uint32_t>::max()) {
      SetError(error, "RKNN model is empty or too large");
      return false;
    }

    // RKNN may free/munmap the model pointer. Ownership transfers at the
    // rknn_init call, so caller-owned or temporary-vector storage is forbidden.
    void* model_copy = std::malloc(size);
    if (model_copy == nullptr) {
      SetError(error, "failed to allocate RKNN model buffer");
      return false;
    }
    std::memcpy(model_copy, data, size);
    const int init_ret = rknn_init(&ctx_, model_copy,
                                   static_cast<uint32_t>(size), 0, nullptr);
    if (init_ret != RKNN_SUCC || ctx_ == 0) {
      // A failed rknn_init does not promise that a nonzero value in the
      // output handle is a destroyable context. Match the RKNN wrappers and
      // do not call rknn_destroy on a failed initialization.
      ctx_ = 0;
      SetError(error, "rknn_init failed: " + std::to_string(init_ret));
      return false;
    }

    if (const std::string* value = FindOption(cfg.options, opt::kRknnCoreMask)) {
      rknn_core_mask mask;
      if (!ParseCoreMask(*value, &mask)) {
        SetError(error, "invalid rknn_core_mask: " + *value);
        return false;
      }
      if (mask != RKNN_NPU_CORE_AUTO &&
          rknn_set_core_mask(ctx_, mask) != RKNN_SUCC) {
        SetError(error, "rknn_set_core_mask failed");
        return false;
      }
    }
    if (!ReadBoolOption(cfg, opt::kRknnPassThrough, false, &pass_through_,
                        error) ||
        !ReadBoolOption(cfg, opt::kRknnWantFloat, true, &want_float_, error)) {
      return false;
    }

    rknn_input_output_num io_num;
    std::memset(&io_num, 0, sizeof(io_num));
    int ret = rknn_query(ctx_, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC || io_num.n_input == 0 || io_num.n_output == 0) {
      SetError(error, "rknn_query IN_OUT_NUM failed: " + std::to_string(ret));
      return false;
    }

    input_infos_.resize(io_num.n_input);
    output_infos_.resize(io_num.n_output);
    input_storage_.resize(io_num.n_input);
    output_storage_.resize(io_num.n_output);
    rknn_inputs_.resize(io_num.n_input);
    rknn_outputs_.resize(io_num.n_output);
    inputs_.resize(io_num.n_input);
    outputs_.resize(io_num.n_output);

    for (uint32_t i = 0; i < io_num.n_input; ++i) {
      input_infos_[i].attr.index = i;
      ret = rknn_query(ctx_, RKNN_QUERY_INPUT_ATTR, &input_infos_[i].attr,
                       sizeof(input_infos_[i].attr));
      if (ret != RKNN_SUCC) {
        SetError(error, "rknn_query INPUT_ATTR[" + std::to_string(i) +
                           "] failed: " + std::to_string(ret));
        return false;
      }
      if (!PrepareInput(i, error)) return false;
    }
    for (uint32_t i = 0; i < io_num.n_output; ++i) {
      output_infos_[i].attr.index = i;
      ret = rknn_query(ctx_, RKNN_QUERY_OUTPUT_ATTR, &output_infos_[i].attr,
                       sizeof(output_infos_[i].attr));
      if (ret != RKNN_SUCC) {
        SetError(error, "rknn_query OUTPUT_ATTR[" + std::to_string(i) +
                           "] failed: " + std::to_string(ret));
        return false;
      }
      if (!PrepareOutput(i, error)) return false;
    }
    return true;
  }

  bool PrepareInput(uint32_t index, std::string* error) {
    std::vector<int> shape;
    if (!ReadInputShape(input_infos_[index].attr, pass_through_, &shape,
                        &input_infos_[index].elements, error)) {
      return false;
    }
    Tensor& tensor = inputs_[index];
    tensor.name = TensorName(input_infos_[index].attr);
    tensor.shape = shape;
    if (pass_through_) {
      tensor.shape.clear();
      tensor.shape.reserve(input_infos_[index].attr.n_dims);
      for (uint32_t i = 0; i < input_infos_[index].attr.n_dims; ++i) {
        if (input_infos_[index].attr.dims[i] >
            static_cast<uint32_t>(std::numeric_limits<int>::max())) {
          SetError(error, "RKNN input dimension exceeds MIE shape range");
          return false;
        }
        tensor.shape.push_back(
            static_cast<int>(input_infos_[index].attr.dims[i]));
      }
      tensor.type = MapType(input_infos_[index].attr.type);
      if (tensor.type == Type::kUnknown) {
        SetError(error, "RKNN pass-through input type is not exposed by MIE");
        return false;
      }
      const std::size_t type_size = TypeSize(input_infos_[index].attr.type);
      if (!ReadTensorBytes(input_infos_[index].attr,
                           input_infos_[index].elements, type_size, true,
                           &tensor.bytes, error)) {
        return false;
      }
    } else {
      tensor.type = Type::kUInt8;
      tensor.bytes = input_infos_[index].elements;
    }
    if (tensor.bytes > std::numeric_limits<uint32_t>::max()) {
      SetError(error, "RKNN input is too large for the runtime API");
      return false;
    }
    input_storage_[index].assign(tensor.bytes, 0);
    tensor.data = input_storage_[index].empty() ? nullptr
                                                : input_storage_[index].data();
    return true;
  }

  bool PrepareOutput(uint32_t index, std::string* error) {
    std::vector<int> shape;
    if (!ReadOutputShape(output_infos_[index].attr, &shape,
                         &output_infos_[index].elements, error)) {
      return false;
    }
    Tensor& tensor = outputs_[index];
    tensor.name = TensorName(output_infos_[index].attr);
    tensor.shape = shape;
    if (want_float_) {
      tensor.type = Type::kFloat32;
      if (output_infos_[index].elements >
          std::numeric_limits<std::size_t>::max() / sizeof(float)) {
        SetError(error, "RKNN output byte size overflows host size_t");
        return false;
      }
      tensor.bytes = output_infos_[index].elements * sizeof(float);
    } else {
      tensor.type = MapType(output_infos_[index].attr.type);
      const std::size_t type_size = TypeSize(output_infos_[index].attr.type);
      if (tensor.type == Type::kUnknown || type_size == 0 ||
          !ReadTensorBytes(output_infos_[index].attr,
                           output_infos_[index].elements, type_size, false,
                           &tensor.bytes, error)) {
        if (tensor.type == Type::kUnknown || type_size == 0) {
          SetError(error, "RKNN output type is not exposed by MIE");
        }
        return false;
      }
    }
    if (tensor.bytes > std::numeric_limits<uint32_t>::max()) {
      SetError(error, "RKNN output is too large for the runtime API");
      return false;
    }
    output_storage_[index].assign(tensor.bytes, 0);
    tensor.data = output_storage_[index].empty() ? nullptr
                                                 : output_storage_[index].data();
    return true;
  }

  rknn_context ctx_ = 0;
  bool pass_through_ = false;
  bool want_float_ = true;
  std::vector<TensorInfo> input_infos_;
  std::vector<TensorInfo> output_infos_;
  std::vector<std::vector<unsigned char> > input_storage_;
  std::vector<std::vector<unsigned char> > output_storage_;
  std::vector<rknn_input> rknn_inputs_;
  std::vector<rknn_output> rknn_outputs_;
  std::vector<Tensor> inputs_;
  std::vector<Tensor> outputs_;
};

#endif  // MIE_ENABLE_RKNN

}  // namespace

std::unique_ptr<Engine> BuildRknn(const Config& cfg, const ModelSource& model,
                                  std::string* error) {
#if !MIE_ENABLE_RKNN
  (void)cfg;
  (void)model;
  SetError(error, "RKNN backend not compiled; it requires ARM64 Linux and MIE_ENABLE_RKNN=ON");
  return nullptr;
#else
  return RknnEngine::Build(cfg, model, error);
#endif
}

}  // namespace internal
}  // namespace mie
