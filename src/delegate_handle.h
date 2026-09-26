#ifndef MIE_INTERNAL_DELEGATE_HANDLE_H_
#define MIE_INTERNAL_DELEGATE_HANDLE_H_

#include "tensorflow/lite/c/c_api_types.h"

#include "src/dl.h"

namespace mie {
namespace internal {

// Owns a TfLiteDelegate and knows exactly how to destroy it: either a plain
// destroy function (a linked delegate), or a plugin's
// tflite_plugin_destroy_delegate plus the dlopen handle it came from.
//
// Bodies are inline so each backend_*.cc can adopt a delegate without pulling
// in a shared object file of its own.
class DelegateHandle {
 public:
  DelegateHandle() = default;
  ~DelegateHandle() { Reset(); }
  DelegateHandle(const DelegateHandle&) = delete;
  DelegateHandle& operator=(const DelegateHandle&) = delete;
  DelegateHandle(DelegateHandle&& other) noexcept { MoveFrom(&other); }
  DelegateHandle& operator=(DelegateHandle&& other) noexcept {
    if (this != &other) {
      Reset();
      MoveFrom(&other);
    }
    return *this;
  }

  void Adopt(TfLiteDelegate* delegate, void (*destroy)(TfLiteDelegate*)) {
    Reset();
    delegate_ = delegate;
    destroy_ = destroy;
  }

  void AdoptPlugin(TfLiteDelegate* delegate, void (*destroy)(TfLiteDelegate*),
                   LibHandle lib) {
    Reset();
    delegate_ = delegate;
    destroy_ = destroy;
    lib_ = lib;
  }

  TfLiteDelegate* get() const { return delegate_; }
  bool empty() const { return delegate_ == nullptr; }

 private:
  // Called by the destructor, the move assignment and both Adopt overloads;
  // nothing outside this class resets a handle.
  void Reset() {
    if (delegate_ != nullptr && destroy_ != nullptr) destroy_(delegate_);
    delegate_ = nullptr;
    destroy_ = nullptr;
    if (lib_ != nullptr) {
      DlClose(lib_);
      lib_ = nullptr;
    }
  }

  void MoveFrom(DelegateHandle* other) {
    delegate_ = other->delegate_;
    destroy_ = other->destroy_;
    lib_ = other->lib_;
    other->delegate_ = nullptr;
    other->destroy_ = nullptr;
    other->lib_ = nullptr;
  }

  TfLiteDelegate* delegate_ = nullptr;
  void (*destroy_)(TfLiteDelegate*) = nullptr;
  LibHandle lib_ = nullptr;  // non-null only for plugin delegates
};

}  // namespace internal
}  // namespace mie

#endif  // MIE_INTERNAL_DELEGATE_HANDLE_H_
