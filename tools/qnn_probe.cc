// Diagnostic for the Qualcomm QNN TFLite delegate.
//
// The engine reaches this delegate through the TFLite external-delegate plugin
// ABI. When that fails it is unclear whether the fault is the delegate/device
// pair or the engine's option plumbing. This probe drives the delegate's own C
// API instead, so the two can be told apart.
//
//   qnn_probe <lib_dir> [skel_dir]
//
// <lib_dir>  directory holding libQnnHtp.so / libQnnHtpV<NN>Stub.so
// <skel_dir> directory holding libQnnHtpV<NN>Skel.so

#include <cstdio>

#include "QNN/TFLiteDelegate/QnnTFLiteDelegate.h"

int main(int argc, char** argv) {
  const QnnDelegateApiVersion version = TfLiteQnnDelegateGetApiVersion();
  std::printf("delegate API version : %u.%u.%u\n", version.major, version.minor,
              version.patch);

  TfLiteQnnDelegateOptions options = TfLiteQnnDelegateOptionsDefault();
  std::printf("default backend_type : %d\n",
              static_cast<int>(options.backend_type));
  std::printf("default log_level    : %d\n",
              static_cast<int>(options.log_level));

  options.backend_type = kHtpBackend;
  options.log_level = kLogLevelVerbose;
  if (argc > 1) options.library_path = argv[1];
  if (argc > 2) options.skel_library_dir = argv[2];

  std::printf("library_path         : %s\n",
              options.library_path ? options.library_path : "(null)");
  std::printf("skel_library_dir     : %s\n",
              options.skel_library_dir ? options.skel_library_dir : "(null)");
  std::printf("calling TfLiteQnnDelegateCreate...\n");
  std::fflush(stdout);

  TfLiteDelegate* delegate = TfLiteQnnDelegateCreate(&options);
  std::printf("TfLiteQnnDelegateCreate -> %p\n",
              static_cast<void*>(delegate));

  if (delegate == nullptr) {
    std::printf("FAILED: delegate not created\n");
    return 1;
  }
  TfLiteQnnDelegateDelete(delegate);
  std::printf("OK\n");
  return 0;
}
