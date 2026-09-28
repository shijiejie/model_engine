#include "src/backend_qnn_native.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "src/dl.h"
#include "src/options.h"
#include "src/tensor_view.h"

#if MIE_ENABLE_QNN_NATIVE
// The QNN SDK headers are plain C (extern "C") and come from the SDK include
// directory. Unlike the Neuron Runtime in backend_mtk_dla.cc, the QNN native
// API is not a flat list of dlopen'd symbols: each backend library exports a
// single QnnInterface_getProviders() that hands back a QnnInterface_t carrying
// a large table of function pointers (QNN_INTERFACE_VER_TYPE). Tensor discovery
// is not part of that table at all — it lives in the separate System Context
// API, and since QNN 2.x libQnnSystem.so no longer exports the
// QnnSystemContext_* symbols directly: it exports QnnSystemInterface_getProviders()
// instead, whose QnnSystemInterface_t hands back the system function table.
#include "QnnInterface.h"
#include "System/QnnSystemContext.h"
#include "System/QnnSystemInterface.h"
#endif

namespace mie {
namespace internal {
namespace {

void SetError(std::string* error, const std::string& message) {
  if (error != nullptr) *error = message;
}

#if MIE_ENABLE_QNN_NATIVE

// Every entry point this backend uses, resolved at run time. The core handles
// come from libQnnHtp.so's QnnInterface_getProviders() table; the system
// handles come from libQnnSystem.so's QnnSystemInterface_getProviders() table
// (QNN 2.x no longer exports the QnnSystemContext_* symbols directly). Nothing
// QNN is linked.
struct QnnNativeApi {
  // QNN core (QnnInterface_t).
  QnnBackend_CreateFn_t backendCreate;
  QnnBackend_FreeFn_t backendFree;
  QnnContext_CreateFromBinaryFn_t contextCreateFromBinary;
  QnnContext_FreeFn_t contextFree;
  QnnGraph_RetrieveFn_t graphRetrieve;
  QnnGraph_ExecuteFn_t graphExecute;

  // QNN System (QnnSystemContext.h), from libQnnSystem.so's interface table.
  QnnSystemContext_CreateFn_t systemContextCreate;
  QnnSystemContext_GetBinaryInfoFn_t systemContextGetBinaryInfo;
  QnnSystemContext_FreeFn_t systemContextFree;
};

// Resolves one symbol and records the first miss, mirroring backend_mtk_dla.cc.
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
        *error = std::string("QNN library does not export ") + name;
      }
    }
    return reinterpret_cast<Fn>(symbol);
  }
};

// Width of a QNN data type in bytes, or 0 if unsupported. Sub-byte types
// (2/4-bit) are legal in QNN but never show up as graph I/O here.
std::size_t DataTypeSize(Qnn_DataType_t type) {
  switch (type) {
    case QNN_DATATYPE_UINT_8:
    case QNN_DATATYPE_INT_8:
    case QNN_DATATYPE_BOOL_8:
      return 1;
    case QNN_DATATYPE_UINT_16:
    case QNN_DATATYPE_INT_16:
    case QNN_DATATYPE_FLOAT_16:
    case QNN_DATATYPE_BFLOAT_16:
    case QNN_DATATYPE_SFIXED_POINT_16:
    case QNN_DATATYPE_UFIXED_POINT_16:
      return 2;
    case QNN_DATATYPE_UINT_32:
    case QNN_DATATYPE_INT_32:
    case QNN_DATATYPE_FLOAT_32:
    case QNN_DATATYPE_SFIXED_POINT_32:
    case QNN_DATATYPE_UFIXED_POINT_32:
      return 4;
    case QNN_DATATYPE_UINT_64:
    case QNN_DATATYPE_INT_64:
    case QNN_DATATYPE_FLOAT_64:
      return 8;
    default:
      return 0;
  }
}

Type MapType(Qnn_DataType_t type) {
  switch (type) {
    case QNN_DATATYPE_FLOAT_32:
      return Type::kFloat32;
    case QNN_DATATYPE_FLOAT_16:
      return Type::kFloat16;
    case QNN_DATATYPE_INT_32:
      return Type::kInt32;
    case QNN_DATATYPE_INT_64:
      return Type::kInt64;
    case QNN_DATATYPE_UINT_8:
      return Type::kUInt8;
    case QNN_DATATYPE_INT_8:
    case QNN_DATATYPE_SFIXED_POINT_8:
      return Type::kInt8;
    case QNN_DATATYPE_BOOL_8:
      return Type::kBool;
    default:
      return Type::kUnknown;
  }
}

// Reads the leading metadata shared by Qnn_TensorV1_t and Qnn_TensorV2_t (their
// first fields have identical layout: id/name/type/dataFormat/dataType/
// quantizeParams/rank/dimensions/memType).
struct TensorMeta {
  const char* name;
  Qnn_DataType_t dataType;
  uint32_t rank;
  uint32_t* dimensions;
};

bool GetTensorMeta(const Qnn_Tensor_t* tensor, TensorMeta* out) {
  switch (tensor->version) {
    case QNN_TENSOR_VERSION_1:
      out->name = tensor->v1.name;
      out->dataType = tensor->v1.dataType;
      out->rank = tensor->v1.rank;
      out->dimensions = tensor->v1.dimensions;
      return true;
    case QNN_TENSOR_VERSION_2:
      out->name = tensor->v2.name;
      out->dataType = tensor->v2.dataType;
      out->rank = tensor->v2.rank;
      out->dimensions = tensor->v2.dimensions;
      return true;
    default:
      return false;
  }
}

// The binary-info versions differ in the fields AFTER `graphs`, so read the
// graph list through the versioned union rather than assuming one layout.
bool ReadGraphList(const QnnSystemContext_BinaryInfo_t* info, uint32_t* num,
                   QnnSystemContext_GraphInfo_t** graphs) {
  switch (info->version) {
    case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1:
      *num = info->contextBinaryInfoV1.numGraphs;
      *graphs = info->contextBinaryInfoV1.graphs;
      return true;
    case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2:
      *num = info->contextBinaryInfoV2.numGraphs;
      *graphs = info->contextBinaryInfoV2.graphs;
      return true;
    case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3:
      *num = info->contextBinaryInfoV3.numGraphs;
      *graphs = info->contextBinaryInfoV3.graphs;
      return true;
    default:
      return false;
  }
}

bool ReadFile(const std::string& path, std::vector<uint8_t>* out,
              std::string* error) {
  // std::fopen/fread rather than <fstream>: the r16b NDK's clang 5.0 breaks on
  // <fstream> (locale pulls in strtof_l missing at API 21). Mirrors finger-
  // print.cc, the only other file reader in the project.
  std::FILE* file = std::fopen(path.c_str(), "rb");
  if (file == nullptr) {
    SetError(error, "cannot open " + path);
    return false;
  }
  std::fseek(file, 0, SEEK_END);
  const long size = std::ftell(file);
  std::fseek(file, 0, SEEK_SET);
  out->resize(static_cast<std::size_t>(size));
  if (size > 0) {
    std::fread(out->data(), 1, static_cast<std::size_t>(size), file);
  }
  const bool ok = std::ferror(file) == 0;
  std::fclose(file);
  if (!ok) {
    SetError(error, "read error on " + path);
    return false;
  }
  return true;
}

// Runs a QNN context binary on the HTP through the native C API. Like kMtk and
// kMtkDla — and unlike CPU/GPU/QNN — the tensor buffers are engine-owned
// staging buffers: the QNN runtime copies raw host buffers in and out rather
// than sharing them, so Input()/Output() return staging views and Run() hands
// the current contents to the device. (Zero-copy DMA-BUF registration is a
// possible follow-up; the prototype copies.)
class QnnNativeEngine : public Engine {
 public:
  ~QnnNativeEngine() override {
    if (context_ != nullptr) api_.contextFree(context_, nullptr);
    if (backend_ != nullptr) api_.backendFree(backend_);
    DlClose(lib_system_);
    DlClose(lib_htp_);
  }

  static std::unique_ptr<QnnNativeEngine> Build(const Config& cfg,
                                                const ModelSource& model,
                                                std::string* error) {
    std::unique_ptr<QnnNativeEngine> engine(new QnnNativeEngine());
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

  // Shapes are baked into the context binary by the offline compiler. QNN can
  // re-shape only graphs built with dynamic dimensions, which this backend does
  // not expose.
  bool ResizeInput(int index, const std::vector<int>& shape) override {
    (void)index;
    (void)shape;
    return false;
  }

  // The execution tensor descriptors were built once and point at the staging
  // buffers, which never move. The caller rewrites Input(i).data and the next
  // Run() submits the same descriptor with the new contents.
  bool Run() override {
    const Qnn_Tensor_t* inputs =
        qnn_inputs_.empty() ? nullptr : qnn_inputs_.data();
    Qnn_Tensor_t* outputs = qnn_outputs_.empty() ? nullptr : qnn_outputs_.data();
    return api_.graphExecute(graph_, inputs,
                             static_cast<uint32_t>(qnn_inputs_.size()), outputs,
                             static_cast<uint32_t>(qnn_outputs_.size()), nullptr,
                             nullptr) == QNN_SUCCESS;
  }

 private:
  QnnNativeEngine() = default;

  bool Init(const Config& cfg, const ModelSource& model, std::string* error) {
    // --- load the QNN core library and its interface ---
    const std::string* htp_override = FindOption(cfg.options, opt::kQnnNativeLibrary);
    htp_lib_ = htp_override != nullptr ? *htp_override : "libQnnHtp.so";
    lib_htp_ = DlOpen(htp_lib_.c_str());
    if (lib_htp_ == nullptr) {
      SetError(error, "failed to dlopen " + htp_lib_ + " (set " +
                          std::string(opt::kQnnNativeLibrary) +
                          " if the device ships a different one)");
      return false;
    }

    SymbolLoader loader;
    loader.lib = lib_htp_;
    loader.error = error;
    loader.ok = true;
    auto get_providers =
        loader.Get<Qnn_ErrorHandle_t (*)(const QnnInterface_t***, uint32_t*)>(
            "QnnInterface_getProviders");
    if (!loader.ok) return false;

    const QnnInterface_t** providers = nullptr;
    uint32_t num_providers = 0;
    if (get_providers == nullptr ||
        get_providers(&providers, &num_providers) != QNN_SUCCESS ||
        providers == nullptr || num_providers == 0) {
      SetError(error, "no QNN interface providers in " + htp_lib_);
      return false;
    }

    // One backend library is loaded, so its only provider is the one we want
    // (the HTP). Copy the function table out; the provider pointers stay valid
    // for the life of the library, which lib_htp_ keeps open.
    const QNN_INTERFACE_VER_TYPE* qnn = &providers[0]->QNN_INTERFACE_VER_NAME;
    api_.backendCreate = qnn->backendCreate;
    api_.backendFree = qnn->backendFree;
    api_.contextCreateFromBinary = qnn->contextCreateFromBinary;
    api_.contextFree = qnn->contextFree;
    api_.graphRetrieve = qnn->graphRetrieve;
    api_.graphExecute = qnn->graphExecute;
    if (api_.backendCreate == nullptr || api_.backendFree == nullptr ||
        api_.contextCreateFromBinary == nullptr || api_.contextFree == nullptr ||
        api_.graphRetrieve == nullptr || api_.graphExecute == nullptr) {
      SetError(error, "QNN HTP interface is missing required functions");
      return false;
    }

    // --- load the system library for context-binary introspection ---
    const std::string* sys_override = FindOption(cfg.options, opt::kQnnSystemLibrary);
    system_lib_ = sys_override != nullptr ? *sys_override : "libQnnSystem.so";
    lib_system_ = DlOpen(system_lib_.c_str());
    if (lib_system_ == nullptr) {
      SetError(error, "failed to dlopen " + system_lib_ + " (set " +
                          std::string(opt::kQnnSystemLibrary) +
                          " if the device ships a different one)");
      return false;
    }

    SymbolLoader sloader;
    sloader.lib = lib_system_;
    sloader.error = error;
    sloader.ok = true;
    auto sys_get_providers =
        sloader.Get<Qnn_ErrorHandle_t (*)(const QnnSystemInterface_t***,
                                          uint32_t*)>(
            "QnnSystemInterface_getProviders");
    if (!sloader.ok) return false;

    const QnnSystemInterface_t** sys_providers = nullptr;
    uint32_t sys_num_providers = 0;
    if (sys_get_providers == nullptr ||
        sys_get_providers(&sys_providers, &sys_num_providers) != QNN_SUCCESS ||
        sys_providers == nullptr || sys_num_providers == 0) {
      SetError(error, "no QNN system interface providers in " + system_lib_);
      return false;
    }
    const QNN_SYSTEM_INTERFACE_VER_TYPE* sysapi =
        &sys_providers[0]->QNN_SYSTEM_INTERFACE_VER_NAME;
    api_.systemContextCreate = sysapi->systemContextCreate;
    api_.systemContextGetBinaryInfo = sysapi->systemContextGetBinaryInfo;
    api_.systemContextFree = sysapi->systemContextFree;
    if (api_.systemContextCreate == nullptr ||
        api_.systemContextGetBinaryInfo == nullptr ||
        api_.systemContextFree == nullptr) {
      SetError(error, "QNN system interface is missing required functions");
      return false;
    }

    // --- the context binary itself ---
    const void* binary = nullptr;
    uint64_t binary_size = 0;
    if (model.is_buffer()) {
      binary = model.data();
      binary_size = model.size();
    } else {
      if (!ReadFile(model.path(), &model_storage_, error)) return false;
      binary = model_storage_.data();
      binary_size = model_storage_.size();
    }

    // --- introspect the binary for graph/tensor metadata ---
    QnnSystemContext_Handle_t sysctx = nullptr;
    if (api_.systemContextCreate(&sysctx) != QNN_SUCCESS || sysctx == nullptr) {
      SetError(error, "QnnSystemContext_create failed");
      return false;
    }
    const QnnSystemContext_BinaryInfo_t* bin_info = nullptr;
    Qnn_ContextBinarySize_t bin_info_size = 0;
    Qnn_ErrorHandle_t bin_err = api_.systemContextGetBinaryInfo(
        sysctx, const_cast<void*>(binary), binary_size, &bin_info,
        &bin_info_size);
    if (bin_err != QNN_SUCCESS || bin_info == nullptr) {
      api_.systemContextFree(sysctx);
      SetError(error,
               "QnnSystemContext_getBinaryInfo failed (is this a QNN context "
               "binary produced by the matching SDK?)");
      return false;
    }

    uint32_t num_graphs = 0;
    QnnSystemContext_GraphInfo_t* graphs = nullptr;
    if (!ReadGraphList(bin_info, &num_graphs, &graphs) || num_graphs == 0) {
      api_.systemContextFree(sysctx);
      SetError(error, "context binary holds no graphs");
      return false;
    }

    const std::string* want = FindOption(cfg.options, opt::kQnnGraphName);
    const QnnSystemContext_GraphInfoV1_t* gi = nullptr;
    for (uint32_t i = 0; i < num_graphs; ++i) {
      // V1/V2/V3 graphs share the leading fields, so graphInfoV1 is safe here.
      const bool match = (want == nullptr) ||
                         (graphs[i].graphInfoV1.graphName != nullptr &&
                          *want == graphs[i].graphInfoV1.graphName);
      if (match) {
        gi = &graphs[i].graphInfoV1;
        break;
      }
    }
    if (gi == nullptr) {
      api_.systemContextFree(sysctx);
      SetError(error, "no graph named '" + (want != nullptr ? *want : "") +
                          "' in the context binary");
      return false;
    }
    if (gi->graphName != nullptr) graph_name_ = gi->graphName;

    if (!BuildTensors(true, gi->numGraphInputs, gi->graphInputs, error)) {
      api_.systemContextFree(sysctx);
      return false;
    }
    if (!BuildTensors(false, gi->numGraphOutputs, gi->graphOutputs, error)) {
      api_.systemContextFree(sysctx);
      return false;
    }
    api_.systemContextFree(sysctx);

    // --- create the backend, context and graph ---
    if (api_.backendCreate(nullptr, nullptr, &backend_) != QNN_SUCCESS ||
        backend_ == nullptr) {
      SetError(error, "QnnBackend_create failed");
      return false;
    }
    if (api_.contextCreateFromBinary(backend_, nullptr, nullptr, binary,
                                     binary_size, &context_, nullptr) !=
            QNN_SUCCESS ||
        context_ == nullptr) {
      SetError(error,
               "QnnContext_createFromBinary failed (the context binary must "
               "match this device's SoC)");
      return false;
    }
    if (api_.graphRetrieve(context_, graph_name_.c_str(), &graph_) !=
            QNN_SUCCESS ||
        graph_ == nullptr) {
      SetError(error, "QnnGraph_retrieve failed for graph '" + graph_name_ + "'");
      return false;
    }

    // A context binary's graph tensors are already baked in; unlike a graph
    // built via graphCreate/graphAddNode there is nothing to register here. The
    // execution descriptors in qnn_inputs_/qnn_outputs_ carry the graph's own
    // ids/names and a raw clientBuf, and are handed straight to graphExecute.
    return true;
  }

  bool BuildTensors(bool is_input, uint32_t count, Qnn_Tensor_t* src,
                    std::string* error) {
    std::vector<Tensor>& tensors = is_input ? inputs_ : outputs_;
    std::vector<std::vector<uint8_t> >& storage =
        is_input ? input_storage_ : output_storage_;
    std::vector<std::string>& names = is_input ? input_names_ : output_names_;
    std::vector<std::vector<uint32_t> >& dims =
        is_input ? input_dims_ : output_dims_;
    std::vector<Qnn_Tensor_t>& qnn = is_input ? qnn_inputs_ : qnn_outputs_;

    // Sized up-front and never grown again, so the pointers handed to QNN
    // (name, dimensions, clientBuf) stay valid for the engine lifetime.
    tensors.resize(count);
    storage.resize(count);
    names.resize(count);
    dims.resize(count);
    qnn.resize(count);

    for (uint32_t i = 0; i < count; ++i) {
      TensorMeta meta;
      if (!GetTensorMeta(&src[i], &meta)) {
        SetError(error, "unsupported tensor metadata version");
        return false;
      }
      if (meta.name != nullptr) names[i] = meta.name;

      const std::size_t elem = DataTypeSize(meta.dataType);
      if (elem == 0) {
        SetError(error, std::string("unsupported tensor data type on ") +
                            (meta.name != nullptr ? meta.name : "?"));
        return false;
      }
      if (meta.rank > 0 && meta.dimensions == nullptr) {
        SetError(error, "tensor reports a rank but no dimensions");
        return false;
      }

      std::size_t num_elems = 1;
      dims[i].assign(meta.dimensions, meta.dimensions + meta.rank);
      for (uint32_t d = 0; d < meta.rank; ++d) num_elems *= dims[i][d];
      const std::size_t bytes = num_elems * elem;

      Tensor& tensor = tensors[i];
      tensor.name = names[i];
      tensor.type = MapType(meta.dataType);
      tensor.shape.assign(dims[i].begin(), dims[i].end());
      tensor.bytes = bytes;

      std::vector<uint8_t>& buffer = storage[i];
      buffer.assign(bytes, 0);
      tensor.data = buffer.empty() ? nullptr : buffer.data();

      // Execution tensor (V1). Start from the graph's own tensor descriptor
      // (from the context-binary introspection) so the id and any dataFormat/
      // quantizeParams match what the offline compiler baked in, then re-point
      // the runtime-owned name/dimensions at our long-lived copies and fill the
      // raw host buffer. APP_WRITE for inputs, APP_READ for outputs — the
      // runtime copies in/out of clientBuf on execute.
      Qnn_Tensor_t t = src[i];
      t.v1.name = names[i].c_str();
      t.v1.type = is_input ? QNN_TENSOR_TYPE_APP_WRITE : QNN_TENSOR_TYPE_APP_READ;
      t.v1.dimensions = dims[i].empty() ? nullptr : dims[i].data();
      t.v1.memType = QNN_TENSORMEMTYPE_RAW;
      t.v1.clientBuf.data = tensor.data;
      t.v1.clientBuf.dataSize = static_cast<uint32_t>(bytes);
      qnn[i] = t;
    }
    return true;
  }

  // Declared first so they are closed last: the api_ pointers and the QNN
  // handles outlive both, and the destructor body frees the context/backend
  // before any member dies.
  LibHandle lib_htp_ = nullptr;
  LibHandle lib_system_ = nullptr;
  QnnNativeApi api_ = {};
  Qnn_BackendHandle_t backend_ = nullptr;
  Qnn_ContextHandle_t context_ = nullptr;
  Qnn_GraphHandle_t graph_ = nullptr;

  std::string htp_lib_;
  std::string system_lib_;
  std::string graph_name_;

  // Owned context-binary bytes when the model came from a path.
  std::vector<uint8_t> model_storage_;

  // Names and dimensions backing the execution tensors' const char* / uint32_t*
  // pointers; declared before the Qnn_Tensor_t vectors that point into them.
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  std::vector<std::vector<uint32_t> > input_dims_;
  std::vector<std::vector<uint32_t> > output_dims_;

  // Storage declared before the views that point into it.
  std::vector<std::vector<uint8_t> > input_storage_;
  std::vector<std::vector<uint8_t> > output_storage_;
  std::vector<Qnn_Tensor_t> qnn_inputs_;
  std::vector<Qnn_Tensor_t> qnn_outputs_;
  std::vector<Tensor> inputs_;
  std::vector<Tensor> outputs_;
};

#endif  // MIE_ENABLE_QNN_NATIVE

}  // namespace

std::unique_ptr<Engine> BuildQnnNative(const Config& cfg, const ModelSource& model,
                                       std::string* error) {
#if !MIE_ENABLE_QNN_NATIVE
  (void)cfg;
  (void)model;
  SetError(error,
           "QNN native backend not compiled in (configure with "
           "-DMIE_ENABLE_QNN_NATIVE=ON and point MIE_QNN_SDK_INCLUDE_DIR at the "
           "directory holding QnnInterface.h)");
  return nullptr;
#else
  return QnnNativeEngine::Build(cfg, model, error);
#endif
}

}  // namespace internal
}  // namespace mie