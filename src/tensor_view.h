#ifndef MIE_INTERNAL_TENSOR_VIEW_H_
#define MIE_INTERNAL_TENSOR_VIEW_H_

#include <vector>

#include "mie/engine.h"
#include "src/log.h"

namespace mie {
namespace internal {

// Input()/Output() return a reference and so cannot report an error, which
// makes an out-of-range index an out-of-bounds read on a std::vector — silent
// memory corruption rather than a diagnosable failure. Log and hand back an
// empty tensor instead: bytes==0 / data==nullptr is defined, and the log line
// names the mistake. See the class comment in mie/engine.h for the precondition
// callers are expected to satisfy.
inline const Tensor& TensorAt(const std::vector<Tensor>& tensors, int index,
                              const char* which) {
  // Function-local static in an inline function: one instance for the whole
  // program, and C++11 guarantees thread-safe initialisation.
  static const Tensor kEmpty;
  if (index < 0 || index >= static_cast<int>(tensors.size())) {
    MIE_LOGW("%s(%d) is out of range (count %d); returning an empty tensor",
             which, index, static_cast<int>(tensors.size()));
    return kEmpty;
  }
  return tensors[index];
}

}  // namespace internal
}  // namespace mie

#endif  // MIE_INTERNAL_TENSOR_VIEW_H_
