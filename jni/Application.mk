# ndk-build application configuration.
#
# This file must live in jni/ — on r16b, build/core/build-local.mk resolves
# Application.mk with
#   NDK_APPLICATION_MK := $(strip $(wildcard $(NDK_PROJECT_PATH)/jni/Application.mk))
# so a file anywhere else is silently ignored (and passing NDK_APPLICATION_MK on
# the command line is clobbered by that same assignment).
#
# APP_PLATFORM=android-21 is enough for CPU/GPU/QNN. The MediaTek backend needs
# android-27, since NeuroPilotTFLiteShim.h depends on <android/NeuralNetworks.h>:
#
#   <ndk>\ndk-build.cmd APP_PLATFORM=android-27 MIE_ENABLE_MTK=1 ^
#       MIE_MTK_INCLUDE_DIR=<dir>
#
# APP_STL is c++_static on purpose: the framework links its C++ runtime
# statically, so no libc++_shared.so has to be shipped. If you link libmie.a
# into an app, build that app with APP_STL := c++_static too — Config passes
# std::string/std::vector across the API boundary, so mixing a static and a
# shared libc++ would break it.
#
# -std=c++11 is set per-module in Android.mk, so APP_CPPFLAGS would only
# duplicate it on the command line.

APP_ABI := arm64-v8a
APP_PLATFORM := android-21
APP_STL := c++_static
APP_OPTIM := release
