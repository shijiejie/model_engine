#ifndef MIE_INTERNAL_MODEL_SOURCE_H_
#define MIE_INTERNAL_MODEL_SOURCE_H_

#include <cstddef>
#include <string>

#include "mie/engine.h"

namespace mie {
namespace internal {

// Where the model comes from: a file path, or a caller-owned in-memory .tflite.
//
// Non-owning by design. The TFLite backends (kCpu/kGpu/kQnn) hand the bytes
// straight to TfLiteModelCreate, which keeps a pointer into them rather than
// copying, so the buffer must outlive the Engine. kMtk happens to copy
// internally, but the same rule is documented for every backend so that the
// contract does not change underneath you when you switch backends.
class ModelSource {
 public:
  static ModelSource FromPath(const std::string& path) {
    ModelSource source;
    source.path_ = path;
    return source;
  }

  static ModelSource FromBuffer(const void* data, std::size_t size) {
    ModelSource source;
    source.data_ = data;
    source.size_ = size;
    return source;
  }

  // A caller-supplied buffer wins over a path when both are set.
  static ModelSource FromConfig(const Config& cfg) {
    if (cfg.model_data != nullptr && cfg.model_size > 0) {
      return FromBuffer(cfg.model_data, cfg.model_size);
    }
    return FromPath(cfg.model_path);
  }

  bool is_buffer() const { return data_ != nullptr; }
  bool empty() const { return data_ == nullptr && path_.empty(); }
  const std::string& path() const { return path_; }
  const void* data() const { return data_; }
  std::size_t size() const { return size_; }

 private:
  std::string path_;
  const void* data_ = nullptr;
  std::size_t size_ = 0;
};

}  // namespace internal
}  // namespace mie

#endif  // MIE_INTERNAL_MODEL_SOURCE_H_
