# Cross toolchain for the Rockchip ARM64 Linux RKNN backend.
#
# The default location matches the development host. Override
# MIE_ROCKCHIP_TOOLCHAIN_ROOT when the SDK is installed elsewhere:
#   cmake -S . -B build-rknn \
#     -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64-rockchip1240-linux-gnu.cmake \
#     -DMIE_ROCKCHIP_TOOLCHAIN_ROOT=/opt/aarch64-rockchip1240-linux-gnu \
#     -DMIE_ENABLE_RKNN=ON -DMIE_ENABLE_TFLITE_BACKENDS=OFF

set(MIE_ROCKCHIP_TOOLCHAIN_ROOT
    "/opt/aarch64-rockchip1240-linux-gnu"
    CACHE PATH "Rockchip aarch64 GNU/Linux toolchain root")
set(MIE_ROCKCHIP_TRIPLE "aarch64-rockchip1240-linux-gnu" CACHE STRING
    "Rockchip cross compiler target triple")

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)
set(CMAKE_SYSROOT
    "${MIE_ROCKCHIP_TOOLCHAIN_ROOT}/${MIE_ROCKCHIP_TRIPLE}/sysroot")

set(_mie_cross_bin "${MIE_ROCKCHIP_TOOLCHAIN_ROOT}/bin")
set(CMAKE_C_COMPILER "${_mie_cross_bin}/${MIE_ROCKCHIP_TRIPLE}-gcc")
set(CMAKE_CXX_COMPILER "${_mie_cross_bin}/${MIE_ROCKCHIP_TRIPLE}-g++")
set(CMAKE_ASM_COMPILER "${_mie_cross_bin}/${MIE_ROCKCHIP_TRIPLE}-as")
set(CMAKE_AR "${_mie_cross_bin}/${MIE_ROCKCHIP_TRIPLE}-ar")
set(CMAKE_RANLIB "${_mie_cross_bin}/${MIE_ROCKCHIP_TRIPLE}-ranlib")
set(CMAKE_STRIP "${_mie_cross_bin}/${MIE_ROCKCHIP_TRIPLE}-strip")

set(CMAKE_FIND_ROOT_PATH "${CMAKE_SYSROOT}")
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# Avoid trying to execute target binaries during configure checks.
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
