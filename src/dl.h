// Cross-platform dynamic-library shim (POSIX dlopen / Win32 LoadLibrary).
#ifndef MIE_INTERNAL_DL_H_
#define MIE_INTERNAL_DL_H_

#if defined(_WIN32)
#include <windows.h>
namespace mie {
namespace internal {
using LibHandle = HMODULE;
inline LibHandle DlOpen(const char* path) { return ::LoadLibraryA(path); }
inline void* DlSym(LibHandle h, const char* sym) {
  return reinterpret_cast<void*>(::GetProcAddress(h, sym));
}
inline void DlClose(LibHandle h) {
  if (h) ::FreeLibrary(h);
}
}  // namespace internal
}  // namespace mie
#else
#include <dlfcn.h>
namespace mie {
namespace internal {
using LibHandle = void*;
inline LibHandle DlOpen(const char* path) { return ::dlopen(path, RTLD_NOW | RTLD_LOCAL); }
inline void* DlSym(LibHandle h, const char* sym) { return ::dlsym(h, sym); }
inline void DlClose(LibHandle h) {
  if (h) ::dlclose(h);
}
}  // namespace internal
}  // namespace mie
#endif

#endif  // MIE_INTERNAL_DL_H_
