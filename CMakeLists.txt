cmake_minimum_required(VERSION 3.16)
project(mie CXX)

# C++11 on purpose: the Android NDK r16b ships clang 5.0, which has no C++17.
# Nothing here needs C++17 — the TFLite C API keeps the boundary plain C.
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

find_package(Threads REQUIRED)

option(MIE_ENABLE_GPU "Build the TFLite GPU backend" ON)
option(MIE_ENABLE_MTK "Build the MediaTek NeuroPilot backend" OFF)
option(MIE_ENABLE_MTK_DLA "Build the MediaTek DLA (Neuron Runtime) backend" OFF)
option(MIE_ENABLE_QNN_NATIVE "Build the Qualcomm QNN native C API backend" OFF)
option(MIE_BUILD_EXAMPLES "Build the examples" ON)

# ---------------------------------------------------------------------------
# TensorFlow Lite
#
# Defaults come from third_party/tflite-dist, a prebuilt TFLite distribution
# committed to the repo (headers + the C API runtime for Android and Windows,
# plus the GPU delegate lib for Android). Override MIE_TFLITE_INCLUDE_DIRS /
# MIE_TFLITE_LIBRARIES to use your own TFLite build.
# ---------------------------------------------------------------------------
set(MIE_TFLITE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/tflite-dist" CACHE PATH
    "Directory holding TFLite include/ and libs/ (android/<abi>, windows_x86_64)")
set(MIE_TFLITE_INCLUDE_DIRS "" CACHE STRING "TFLite include dirs")
set(MIE_TFLITE_LIBRARIES "" CACHE STRING "TFLite runtime library")
set(MIE_TFLITE_GPU_DELEGATE_LIB "" CACHE STRING "TFLite GPU delegate library")

if(NOT MIE_TFLITE_INCLUDE_DIRS AND EXISTS "${MIE_TFLITE_ROOT}/include")
  set(MIE_TFLITE_INCLUDE_DIRS "${MIE_TFLITE_ROOT}/include")
endif()

if(NOT MIE_TFLITE_LIBRARIES AND ANDROID)
  set(_mie_abi_dir "${MIE_TFLITE_ROOT}/libs/android/${ANDROID_ABI}")
  if(EXISTS "${_mie_abi_dir}/libtensorflowlite_c.so")
    set(MIE_TFLITE_LIBRARIES "${_mie_abi_dir}/libtensorflowlite_c.so")
  endif()
  if(NOT MIE_TFLITE_GPU_DELEGATE_LIB
     AND EXISTS "${_mie_abi_dir}/libtensorflowlite_gpu_delegate.so")
    set(MIE_TFLITE_GPU_DELEGATE_LIB
        "${_mie_abi_dir}/libtensorflowlite_gpu_delegate.so")
  endif()
elseif(NOT MIE_TFLITE_LIBRARIES AND WIN32)
  # Windows MSVC: the committed prebuilt C API dll and its import lib. The GPU
  # delegate is not shipped for Windows (build with -DMIE_ENABLE_GPU=OFF).
  set(_mie_win_dir "${MIE_TFLITE_ROOT}/libs/windows_x86_64")
  if(EXISTS "${_mie_win_dir}/tensorflowlite_c.dll.if.lib")
    set(MIE_TFLITE_LIBRARIES "${_mie_win_dir}/tensorflowlite_c.dll.if.lib")
  endif()
endif()

# One file per backend, plus the shared TFLite interpreter core they all build
# on. backend_mtk.cc and backend_mtk_dla.cc are self-contained: neither
# MediaTek path goes through a TFLite delegate.
add_library(mie STATIC
  src/engine.cc
  src/fingerprint.cc
  src/options.cc
  src/delegate_plugin.cc
  src/tflite_interpreter.cc
  src/backend_cpu.cc
  src/backend_gpu.cc
  src/backend_qnn.cc
  src/backend_qnn_native.cc
  src/backend_mtk.cc
  src/backend_mtk_dla.cc
)
target_include_directories(mie PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
)
set_target_properties(mie PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(NOT MIE_TFLITE_INCLUDE_DIRS)
  message(FATAL_ERROR
    "No TFLite headers. Point MIE_TFLITE_ROOT at a prebuilt TFLite "
    "distribution or set -DMIE_TFLITE_INCLUDE_DIRS=...")
endif()
target_include_directories(mie SYSTEM PUBLIC ${MIE_TFLITE_INCLUDE_DIRS})

if(NOT MIE_TFLITE_LIBRARIES)
  message(FATAL_ERROR
    "No TFLite runtime library. Point MIE_TFLITE_ROOT at a prebuilt TFLite "
    "distribution or set -DMIE_TFLITE_LIBRARIES=...")
endif()
target_link_libraries(mie PUBLIC ${MIE_TFLITE_LIBRARIES})

# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------
if(MIE_ENABLE_GPU)
  target_compile_definitions(mie PRIVATE MIE_ENABLE_GPU=1)
  if(MIE_TFLITE_GPU_DELEGATE_LIB)
    target_link_libraries(mie PUBLIC ${MIE_TFLITE_GPU_DELEGATE_LIB})
    # The prebuilt GPU delegate has EGL/GLESv3 as DT_NEEDED, so the linker
    # needs them on the command line to resolve that library's symbols.
    if(ANDROID)
      target_link_libraries(mie PUBLIC EGL GLESv3)
    endif()
  else()
    message(WARNING
      "MIE_ENABLE_GPU=ON but no GPU delegate library was configured; the kGpu "
      "backend will fail at runtime.")
  endif()
else()
  target_compile_definitions(mie PRIVATE MIE_ENABLE_GPU=0)
endif()

if(MIE_ENABLE_MTK)
  set(MIE_MTK_INCLUDE_DIR "" CACHE STRING
      "Directory holding NeuroPilotTFLiteShim.h")
  target_compile_definitions(mie PRIVATE MIE_ENABLE_MTK=1)

  # There is no MIE_MTK_LIB: the shim is header-only and resolves
  # libtflite_mtk.so with dlopen at runtime, so nothing is linked here.
  if(NOT MIE_MTK_INCLUDE_DIR)
    message(FATAL_ERROR
      "MIE_ENABLE_MTK=ON needs -DMIE_MTK_INCLUDE_DIR=<dir holding "
      "NeuroPilotTFLiteShim.h>. That header is confidential MediaTek material, "
      "so it is not fetched automatically.")
  endif()
  target_include_directories(mie SYSTEM PRIVATE ${MIE_MTK_INCLUDE_DIR})

  # NeuroPilotTFLiteShim.h uses NNAPI constants and types but does not include
  # <android/NeuralNetworks.h> itself; backend_mtk.cc does, and that header only
  # exists from API 27.
  if(DEFINED ANDROID_API AND ANDROID_API LESS 27)
    message(FATAL_ERROR
      "MIE_ENABLE_MTK=ON requires -DANDROID_API=27 or newer (the NeuroPilot shim "
      "depends on <android/NeuralNetworks.h>); got ANDROID_API=${ANDROID_API}")
  endif()
else()
  target_compile_definitions(mie PRIVATE MIE_ENABLE_MTK=0)
endif()

if(MIE_ENABLE_MTK_DLA)
  set(MIE_MTK_SDK_INCLUDE_DIR "" CACHE STRING
      "Directory holding neuron/api/RuntimeV2.h (from the NeuroPilot SDK)")
  target_compile_definitions(mie PRIVATE MIE_ENABLE_MTK_DLA=1)

  # The Neuron Runtime is dlopen'd at run time, so nothing is linked here — the
  # include directory is only needed for the C API declarations.
  if(NOT MIE_MTK_SDK_INCLUDE_DIR)
    message(FATAL_ERROR
      "MIE_ENABLE_MTK_DLA=ON needs -DMIE_MTK_SDK_INCLUDE_DIR=<the SDK's per-chip "
      "include dir, e.g. neuron_sdk/<chip>/include holding neuron/api/RuntimeV2.h>. "
      "That header is confidential "
      "MediaTek material, so it is not fetched automatically.")
  endif()
  target_include_directories(mie SYSTEM PRIVATE ${MIE_MTK_SDK_INCLUDE_DIR})
else()
  target_compile_definitions(mie PRIVATE MIE_ENABLE_MTK_DLA=0)
endif()

if(MIE_ENABLE_QNN_NATIVE)
  set(MIE_QNN_SDK_INCLUDE_DIR "" CACHE STRING
      "Directory holding QnnInterface.h (from the Qualcomm QNN SDK)")
  target_compile_definitions(mie PRIVATE MIE_ENABLE_QNN_NATIVE=1)

  # libQnnHtp.so and libQnnSystem.so are dlopen'd at run time, so nothing is
  # linked here — the include directory is only needed for the C API
  # declarations.
  if(NOT MIE_QNN_SDK_INCLUDE_DIR)
    message(FATAL_ERROR
      "MIE_ENABLE_QNN_NATIVE=ON needs -DMIE_QNN_SDK_INCLUDE_DIR=<the QNN SDK "
      "include dir, e.g. <sdk>/include/QNN holding QnnInterface.h and "
      "System/QnnSystemContext.h>. Those headers are confidential Qualcomm "
      "material, so they are not fetched automatically.")
  endif()
  target_include_directories(mie SYSTEM PRIVATE ${MIE_QNN_SDK_INCLUDE_DIR})
else()
  target_compile_definitions(mie PRIVATE MIE_ENABLE_QNN_NATIVE=0)
endif()

# QNN (delegate) is dlopen'd, so it needs nothing at build time.
if(ANDROID)
  target_link_libraries(mie PUBLIC log)
endif()
target_link_libraries(mie PUBLIC Threads::Threads ${CMAKE_DL_LIBS})

# ---------------------------------------------------------------------------
# Examples
# ---------------------------------------------------------------------------
if(MIE_BUILD_EXAMPLES)
  add_executable(mie_run examples/run_model.cc)
  target_link_libraries(mie_run PRIVATE mie)
endif()
