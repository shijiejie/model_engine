cmake_minimum_required(VERSION 3.16)
project(mie CXX)

# C++11 on purpose: the Android NDK r16b ships clang 5.0, which has no C++17.
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# Prefer the compiler's pthread flag over the legacy `-lpthreads` probe.
# The Rockchip sysroot provides pthreads through libc, not a library named
# libpthreads, and CMake's fallback probe otherwise records a bad link flag.
if(CMAKE_CROSSCOMPILING AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
  add_library(mie_threads INTERFACE)
  set(MIE_THREADS_TARGET mie_threads)
else()
  set(THREADS_PREFER_PTHREAD_FLAG TRUE)
  find_package(Threads REQUIRED)
  set(MIE_THREADS_TARGET Threads::Threads)
endif()

option(MIE_ENABLE_RKNN "Build the Rockchip RKNN backend; ARM64 Linux only" OFF)
set(_mie_tflite_default ON)
if(MIE_ENABLE_RKNN)
  set(_mie_tflite_default OFF)
endif()
option(MIE_ENABLE_TFLITE_BACKENDS
       "Build the existing TFLite/Android-oriented backends" ${_mie_tflite_default})
set(_mie_gpu_default ON)
if(WIN32)
  # The bundled Windows distribution contains only the TFLite C API import
  # library. Keep the default linkable; callers can opt into GPU by supplying
  # a compatible delegate library explicitly.
  set(_mie_gpu_default OFF)
endif()
option(MIE_ENABLE_GPU "Build the TFLite GPU backend" ${_mie_gpu_default})
option(MIE_ENABLE_MTK "Build the MediaTek NeuroPilot backend" OFF)
option(MIE_ENABLE_MTK_DLA "Build the MediaTek DLA (Neuron Runtime) backend" OFF)
option(MIE_ENABLE_QNN_NATIVE "Build the Qualcomm QNN native C API backend" OFF)
option(MIE_BUILD_EXAMPLES "Build the examples" ON)

if(MIE_ENABLE_RKNN)
  if(ANDROID OR NOT CMAKE_SYSTEM_NAME STREQUAL "Linux" OR
     NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
    message(FATAL_ERROR
      "MIE_ENABLE_RKNN=ON requires non-Android ARM64 Linux "
      "(CMAKE_SYSTEM_NAME=Linux, CMAKE_SYSTEM_PROCESSOR=aarch64/arm64)")
  endif()
  if(MIE_ENABLE_TFLITE_BACKENDS)
    message(FATAL_ERROR
      "MIE_ENABLE_RKNN=ON is mutually exclusive with "
      "MIE_ENABLE_TFLITE_BACKENDS=ON; use a fresh build directory or "
      "pass -DMIE_ENABLE_TFLITE_BACKENDS=OFF")
  endif()
endif()

# ---------------------------------------------------------------------------
# Source selection
# ---------------------------------------------------------------------------
set(MIE_SOURCES
  src/engine.cc
  src/fingerprint.cc
  src/options.cc
)

if(MIE_ENABLE_TFLITE_BACKENDS)
  list(APPEND MIE_SOURCES
    src/delegate_plugin.cc
    src/tflite_interpreter.cc
    src/backend_cpu.cc
    src/backend_gpu.cc
    src/backend_qnn.cc
    src/backend_mtk.cc
  )
endif()

if(MIE_ENABLE_MTK_DLA)
  list(APPEND MIE_SOURCES src/backend_mtk_dla.cc)
endif()
if(MIE_ENABLE_QNN_NATIVE)
  list(APPEND MIE_SOURCES src/backend_qnn_native.cc)
endif()
if(MIE_ENABLE_RKNN)
  list(APPEND MIE_SOURCES src/backend_rknn.cc)
endif()

add_library(mie STATIC ${MIE_SOURCES})
target_include_directories(mie PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
)
set_target_properties(mie PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_definitions(mie PRIVATE
  MIE_ENABLE_RKNN=$<BOOL:${MIE_ENABLE_RKNN}>
  MIE_ENABLE_TFLITE_BACKENDS=$<BOOL:${MIE_ENABLE_TFLITE_BACKENDS}>
  MIE_ENABLE_MTK_DLA=$<BOOL:${MIE_ENABLE_MTK_DLA}>
  MIE_ENABLE_QNN_NATIVE=$<BOOL:${MIE_ENABLE_QNN_NATIVE}>
)

# ---------------------------------------------------------------------------
# TensorFlow Lite and the existing Android backends
# ---------------------------------------------------------------------------
if(MIE_ENABLE_TFLITE_BACKENDS)
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
    set(_mie_win_dir "${MIE_TFLITE_ROOT}/libs/windows_x86_64")
    if(EXISTS "${_mie_win_dir}/tensorflowlite_c.dll.if.lib")
      set(MIE_TFLITE_LIBRARIES "${_mie_win_dir}/tensorflowlite_c.dll.if.lib")
    endif()
    if(NOT MIE_TFLITE_GPU_DELEGATE_LIB
       AND EXISTS "${_mie_win_dir}/tensorflowlite_gpu_delegate.dll.if.lib")
      set(MIE_TFLITE_GPU_DELEGATE_LIB
          "${_mie_win_dir}/tensorflowlite_gpu_delegate.dll.if.lib")
    endif()
  endif()

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

  if(MIE_ENABLE_GPU)
    target_compile_definitions(mie PRIVATE MIE_ENABLE_GPU=1)
    if(MIE_TFLITE_GPU_DELEGATE_LIB)
      target_link_libraries(mie PUBLIC ${MIE_TFLITE_GPU_DELEGATE_LIB})
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
    if(NOT MIE_MTK_INCLUDE_DIR)
      message(FATAL_ERROR
        "MIE_ENABLE_MTK=ON needs -DMIE_MTK_INCLUDE_DIR=<dir holding "
        "NeuroPilotTFLiteShim.h>")
    endif()
    target_include_directories(mie SYSTEM PRIVATE ${MIE_MTK_INCLUDE_DIR})
    # NeuroPilotTFLiteShim.h calls __android_log_print, so the Android log lib
    # is required whenever the MTK backend is built. The CPU/GPU/QNN backends do
    # not use it, so this is scoped to MTK rather than linked on every Android
    # target.
    target_link_libraries(mie PUBLIC log)
    if(DEFINED ANDROID_API AND ANDROID_API LESS 27)
      message(FATAL_ERROR
        "MIE_ENABLE_MTK=ON requires -DANDROID_API=27 or newer")
    endif()
  else()
    target_compile_definitions(mie PRIVATE MIE_ENABLE_MTK=0)
  endif()
else()
  target_compile_definitions(mie PRIVATE
    MIE_ENABLE_GPU=0
    MIE_ENABLE_MTK=0
  )
endif()

if(MIE_ENABLE_MTK_DLA)
  set(MIE_MTK_SDK_INCLUDE_DIR "" CACHE STRING
      "Directory holding neuron/api/RuntimeV2.h")
  if(NOT MIE_MTK_SDK_INCLUDE_DIR)
    message(FATAL_ERROR
      "MIE_ENABLE_MTK_DLA=ON needs -DMIE_MTK_SDK_INCLUDE_DIR=<SDK include dir>")
  endif()
  target_include_directories(mie SYSTEM PRIVATE ${MIE_MTK_SDK_INCLUDE_DIR})
endif()

if(MIE_ENABLE_QNN_NATIVE)
  set(MIE_QNN_SDK_INCLUDE_DIR "" CACHE STRING
      "Directory holding QnnInterface.h")
  if(NOT MIE_QNN_SDK_INCLUDE_DIR)
    message(FATAL_ERROR
      "MIE_ENABLE_QNN_NATIVE=ON needs -DMIE_QNN_SDK_INCLUDE_DIR=<QNN include dir>")
  endif()
  target_include_directories(mie SYSTEM PRIVATE ${MIE_QNN_SDK_INCLUDE_DIR})
endif()

# ---------------------------------------------------------------------------
# Rockchip RKNN, ARM Linux only
# ---------------------------------------------------------------------------
if(MIE_ENABLE_RKNN)
  set(MIE_RKNN_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/rknn" CACHE PATH
      "RKNN SDK root containing include/ and lib/Linux/aarch64/")
  set(MIE_RKNN_INCLUDE_DIR "${MIE_RKNN_ROOT}/include" CACHE PATH
      "Directory containing rknn_api.h")
  set(MIE_RKNN_LIBRARY
      "${MIE_RKNN_ROOT}/lib/Linux/aarch64/librknnrt.so" CACHE FILEPATH
      "AArch64 Linux RKNN runtime library")
  if(NOT EXISTS "${MIE_RKNN_INCLUDE_DIR}/rknn_api.h")
    message(FATAL_ERROR "RKNN header not found: ${MIE_RKNN_INCLUDE_DIR}/rknn_api.h")
  endif()
  if(NOT EXISTS "${MIE_RKNN_LIBRARY}")
    message(FATAL_ERROR "RKNN runtime not found: ${MIE_RKNN_LIBRARY}")
  endif()
  target_include_directories(mie SYSTEM PRIVATE ${MIE_RKNN_INCLUDE_DIR})
  target_link_libraries(mie PUBLIC ${MIE_RKNN_LIBRARY})
  install(FILES "${MIE_RKNN_LIBRARY}" DESTINATION lib)
endif()

target_link_libraries(mie PUBLIC ${MIE_THREADS_TARGET} ${CMAKE_DL_LIBS})

# ---------------------------------------------------------------------------
# Examples
# ---------------------------------------------------------------------------
if(MIE_BUILD_EXAMPLES)
  add_executable(mie_run examples/run_model.cc)
  target_link_libraries(mie_run PRIVATE mie)
  # The prebuilt TFLite .so references API-26 versioned libc symbols
  # (strtod_l@LIBC_O, __gnu_strerror_r@LIBC, ...) that do not resolve when the
  # executable is linked at android-21. Let them stay undefined at link time,
  # exactly as ndk-build does; the device's newer libc provides them at runtime.
  # This is a GNU ld flag and only applies to the Android prebuilt .so — MSVC's
  # link.exe would reject it, so scope it to Android.
  if(ANDROID)
    target_link_options(mie_run PRIVATE "-Wl,--allow-shlib-undefined")
  endif()
  if(MIE_ENABLE_RKNN)
    set_target_properties(mie_run PROPERTIES
      BUILD_RPATH "$ORIGIN/../third_party/rknn/lib/Linux/aarch64"
      INSTALL_RPATH "$ORIGIN/../lib")
  endif()

  # Benchmark driver, arguments parsed with the single-header cmdline library
  # (third_party/cmdline, MIT license — see the file header).
  add_executable(mie_bench examples/bench_model.cc)
  target_link_libraries(mie_bench PRIVATE mie)
  target_include_directories(mie_bench PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/cmdline)
  if(ANDROID)
    target_link_options(mie_bench PRIVATE "-Wl,--allow-shlib-undefined")
  endif()
  if(MIE_ENABLE_RKNN)
    set_target_properties(mie_bench PROPERTIES
      BUILD_RPATH "$ORIGIN/../third_party/rknn/lib/Linux/aarch64"
      INSTALL_RPATH "$ORIGIN/../lib")
  endif()
endif()

include(GNUInstallDirs)
install(TARGETS mie
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
if(MIE_BUILD_EXAMPLES)
  install(TARGETS mie_run mie_bench RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
