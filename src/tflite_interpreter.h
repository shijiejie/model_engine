#ifndef MIE_INTERNAL_TFLITE_INTERPRETER_H_
#define MIE_INTERNAL_TFLITE_INTERPRETER_H_

#include <memory>
#include <string>

#include "mie/engine.h"
#include "src/delegate_handle.h"
#include "src/model_source.h"

namespace mie {
namespace internal {

// Builds a TFLite-interpreter-backed engine. `delegate` may be empty, which is
// the CPU case (no delegation at all). Shared by backend_cpu, backend_gpu and
// backend_qnn — those three differ only in which delegate they hand over.
std::unique_ptr<Engine> BuildTfliteEngine(const Config& cfg,
                                          const ModelSource& model,
                                          DelegateHandle delegate,
                                          std::string* error);

}  // namespace internal
}  // namespace mie

#endif  // MIE_INTERNAL_TFLITE_INTERPRETER_H_
