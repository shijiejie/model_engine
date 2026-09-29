@echo off
rem Example SDK paths used by scripts\build_android.bat when building the
rem Android backends. This file is committed as a template -- copy your real
rem paths in here, or pass /sdk <dir> / /qnnsdk <dir> to build_android.bat to
rem have it rewrite this file with the detected paths.
rem
rem Every variable is optional and guarded with `if not defined`, so an explicit
rem environment variable (or command-line flag) always takes precedence. Each is
rem only needed when the matching backend is enabled -- see README "Build".
rem
rem Android NDK (needed for every Android build):
if not defined ANDROID_NDK_ROOT set "ANDROID_NDK_ROOT=D:\path\to\android-ndk-r16b"
rem MediaTek NeuroPilot SDK, per-chip include dir (only for /dla; must contain
rem neuron/api/RuntimeV2.h):
if not defined MIE_MTK_SDK_INCLUDE_DIR set "MIE_MTK_SDK_INCLUDE_DIR=D:\path\to\neuropilot-sdk\neuron_sdk\<chip>\include"
rem Qualcomm QNN (QAIRT) SDK include dir (only for /qnnnative; must contain
rem QnnInterface.h):
if not defined MIE_QNN_SDK_INCLUDE_DIR set "MIE_QNN_SDK_INCLUDE_DIR=D:\path\to\qairt\<version>\include\QNN"