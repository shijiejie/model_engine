# ndk-build entry point.
#
# ndk-build looks for jni/Android.mk and jni/Application.mk under
# NDK_PROJECT_PATH, so from the project root a plain
#
#   <ndk>\ndk-build.cmd
#
# is all you need — no NDK_PROJECT_PATH / APP_BUILD_SCRIPT flags. (Passing
# NDK_APPLICATION_MK does NOT work on r16b: build/core/build-local.mk
# re-assigns it with $(wildcard $(NDK_PROJECT_PATH)/jni/Application.mk), which
# clobbers the command-line value and silently falls back to the default app.)
#
# scripts\build_android.bat wraps this and packages the result.
#
# Overridable, e.g. `ndk-build MIE_ENABLE_MTK=1 MIE_MTK_INCLUDE_DIR=...`:
#   MIE_ENABLE_GPU=0|1        GPU backend              (default 1)
#   MIE_ENABLE_MTK=0|1        MediaTek backend         (default 0)
#   MIE_ENABLE_MTK_DLA=0|1    MediaTek DLA backend     (default 0)
#   MIE_ENABLE_QNN_NATIVE=0|1 Qualcomm QNN native backend (default 0)
#   MIE_BUILD_EXAMPLE=0|1     build the mie_run driver (default 1)
#   MIE_MTK_INCLUDE_DIR=<dir> directory holding NeuroPilotTFLiteShim.h,
#                             required when MIE_ENABLE_MTK=1
#   MIE_MTK_SDK_INCLUDE_DIR=<dir>
#                             directory holding neuron/api/RuntimeAPI.h,
#                             required when MIE_ENABLE_MTK_DLA=1
#   MIE_QNN_SDK_INCLUDE_DIR=<dir>
#                             directory holding QnnInterface.h,
#                             required when MIE_ENABLE_QNN_NATIVE=1
#   MIE_TFLITE_ROOT=<dir>     overrides third_party/tflite-dist
#
# MIE_ENABLE_MTK=1 also needs APP_PLATFORM >= android-27, because
# NeuroPilotTFLiteShim.h depends on <android/NeuralNetworks.h>.
# MIE_ENABLE_MTK_DLA=1 has no platform requirement: the Neuron Runtime headers
# are plain C.

LOCAL_PATH := $(call my-dir)
MIE_ROOT := $(abspath $(LOCAL_PATH)/..)

# The sources live at the project root, so point LOCAL_PATH there and every
# LOCAL_SRC_FILES below is project-root relative.
LOCAL_PATH := $(MIE_ROOT)

MIE_ENABLE_GPU      ?= 1
MIE_ENABLE_MTK      ?= 0
MIE_ENABLE_MTK_DLA  ?= 0
MIE_ENABLE_QNN_NATIVE ?= 0
MIE_BUILD_EXAMPLE   ?= 1
MIE_TFLITE_ROOT     ?= $(MIE_ROOT)/third_party/tflite-dist
MIE_MTK_INCLUDE_DIR ?=
MIE_MTK_SDK_INCLUDE_DIR ?=
MIE_QNN_SDK_INCLUDE_DIR ?=

MIE_SRC_FILES := \
    src/engine.cc \
    src/fingerprint.cc \
    src/options.cc \
    src/delegate_plugin.cc \
    src/tflite_interpreter.cc \
    src/backend_cpu.cc \
    src/backend_gpu.cc \
    src/backend_qnn.cc \
    src/backend_qnn_native.cc \
    src/backend_mtk.cc \
    src/backend_mtk_dla.cc

ifeq ($(wildcard $(MIE_TFLITE_ROOT)/libs/android/$(TARGET_ARCH_ABI)/libtensorflowlite_c.so),)
$(error TFLite C API lib is missing for $(TARGET_ARCH_ABI) under $(MIE_TFLITE_ROOT)/libs/android/)
endif

# ---------------------------------------------------------------------------
# Prebuilt dependencies. The C API .so exports the full TFLite C API, including
# the XNNPACK symbols backend_cpu.cc uses; the AAR's libtensorflowlite_jni.so
# does not reliably re-export them. LOCAL_MODULE must match the .so's SONAME,
# otherwise ndk-build copies the file under a name that differs from the
# DT_NEEDED it records and the device loader fails to find it.
# ---------------------------------------------------------------------------
include $(CLEAR_VARS)
LOCAL_MODULE := tensorflowlite_c
LOCAL_SRC_FILES := $(MIE_TFLITE_ROOT)/libs/android/$(TARGET_ARCH_ABI)/libtensorflowlite_c.so
LOCAL_EXPORT_C_INCLUDES := $(MIE_TFLITE_ROOT)/include
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := tensorflowlite_gpu_delegate
LOCAL_SRC_FILES := $(MIE_TFLITE_ROOT)/libs/android/$(TARGET_ARCH_ABI)/libtensorflowlite_gpu_delegate.so
include $(PREBUILT_SHARED_LIBRARY)

# ---------------------------------------------------------------------------
# libmie — the framework, static so an app links it in.
# ---------------------------------------------------------------------------
include $(CLEAR_VARS)
LOCAL_MODULE := mie
LOCAL_SRC_FILES := $(MIE_SRC_FILES)

LOCAL_C_INCLUDES := $(MIE_ROOT)/include $(MIE_ROOT)
LOCAL_CPPFLAGS := -std=c++11 -Wall -Wextra
LOCAL_CFLAGS := -DMIE_ENABLE_GPU=$(MIE_ENABLE_GPU) -DMIE_ENABLE_MTK=$(MIE_ENABLE_MTK)
LOCAL_CFLAGS += -DMIE_ENABLE_MTK_DLA=$(MIE_ENABLE_MTK_DLA)
LOCAL_CFLAGS += -DMIE_ENABLE_QNN_NATIVE=$(MIE_ENABLE_QNN_NATIVE)

ifeq ($(MIE_ENABLE_MTK),1)
ifeq ($(MIE_MTK_INCLUDE_DIR),)
$(error MIE_ENABLE_MTK=1 requires MIE_MTK_INCLUDE_DIR=<dir holding NeuroPilotTFLiteShim.h>)
endif
LOCAL_C_INCLUDES += $(MIE_MTK_INCLUDE_DIR)
endif

ifeq ($(MIE_ENABLE_MTK_DLA),1)
ifeq ($(MIE_MTK_SDK_INCLUDE_DIR),)
$(error MIE_ENABLE_MTK_DLA=1 requires MIE_MTK_SDK_INCLUDE_DIR=<dir holding neuron/api/RuntimeV2.h>)
endif
LOCAL_C_INCLUDES += $(MIE_MTK_SDK_INCLUDE_DIR)
endif

ifeq ($(MIE_ENABLE_QNN_NATIVE),1)
ifeq ($(MIE_QNN_SDK_INCLUDE_DIR),)
$(error MIE_ENABLE_QNN_NATIVE=1 requires MIE_QNN_SDK_INCLUDE_DIR=<dir holding QnnInterface.h>)
endif
LOCAL_C_INCLUDES += $(MIE_QNN_SDK_INCLUDE_DIR)
endif

# Recorded for consumers; a static library links nothing by itself.
LOCAL_SHARED_LIBRARIES := tensorflowlite_c tensorflowlite_gpu_delegate
LOCAL_EXPORT_C_INCLUDES := $(MIE_ROOT)/include $(MIE_TFLITE_ROOT)/include
include $(BUILD_STATIC_LIBRARY)

# ---------------------------------------------------------------------------
# mie_run — the standalone driver used for device testing.
#
# ndk-build installs the prebuilt TFLite .so next to it in libs/<abi>/, so the
# whole deployable set is that one directory:
#   adb push libs/arm64-v8a/. /data/local/tmp/mie/
# ---------------------------------------------------------------------------
ifeq ($(MIE_BUILD_EXAMPLE),1)
include $(CLEAR_VARS)
LOCAL_MODULE := mie_run
LOCAL_SRC_FILES := examples/run_model.cc
LOCAL_C_INCLUDES := $(MIE_ROOT)/include
LOCAL_CPPFLAGS := -std=c++11 -Wall -Wextra
LOCAL_CFLAGS := -DMIE_ENABLE_GPU=$(MIE_ENABLE_GPU) -DMIE_ENABLE_MTK=$(MIE_ENABLE_MTK)
LOCAL_CFLAGS += -DMIE_ENABLE_MTK_DLA=$(MIE_ENABLE_MTK_DLA)
LOCAL_CFLAGS += -DMIE_ENABLE_QNN_NATIVE=$(MIE_ENABLE_QNN_NATIVE)
LOCAL_STATIC_LIBRARIES := mie
LOCAL_SHARED_LIBRARIES := tensorflowlite_c tensorflowlite_gpu_delegate
LOCAL_LDLIBS := -llog -ldl -lEGL -lGLESv3

# ndk-build adds the libc++ archives itself but never passes -stdlib, so a bare
# clang++ falls back to its own default and appends a spurious -lstdc++. That
# showed up as libstdc++.so in the executable's DT_NEEDED — the deprecated system
# STL riding along next to a statically linked libc++. Say which STL we mean so
# the driver stops guessing.
#
# -static-libstdc++ makes the driver pull `libc++.a`, which is a linker SCRIPT
# (INPUT(-lc++_static -lc++abi -landroid_support)), not an archive. Its -l
# references are only resolvable through -L, and ndk-build hands the STL archives
# over as explicit paths, so that directory has to be added by hand.
LOCAL_LDFLAGS := -stdlib=libc++ -static-libstdc++ \
                 -L$(NDK_ROOT)/sources/cxx-stl/llvm-libc++/libs/$(TARGET_ARCH_ABI)

# TFLite 2.18's .so references API 26+ symbols (e.g. strtod_l@LIBC_O) that the
# r14b sysroot stub libc (<= android-25) does not declare. The device's newer
# bionic resolves them at run time; this flag keeps the static link from failing
# on those unnamed/versioned symbols.
LOCAL_LDFLAGS += -Wl,--allow-shlib-undefined
include $(BUILD_EXECUTABLE)
endif
