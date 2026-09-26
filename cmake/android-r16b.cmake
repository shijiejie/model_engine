# CMake toolchain for the Android NDK r16b (clang 5.0 -> C++11 only).
#
# r16b predates the modern NDK layout, so its own android.toolchain.cmake is not
# usable with current CMake and the pieces have to be wired by hand:
#   * unified headers   -> <ndk>/sysroot/usr/include[/<triple>]
#   * platform libs     -> <ndk>/platforms/android-<api>/arch-<arch>/usr/lib
#   * libc++            -> <ndk>/sources/cxx-stl/llvm-libc++/...
#   * linker            -> the GNU 4.9 ld; this NDK ships no `ld` next to clang,
#                          so clang's default bare `ld` lookup fails. Select it
#                          explicitly with -fuse-ld=<path>.
#
#   cmake -B build -G Ninja \
#     -DCMAKE_TOOLCHAIN_FILE=cmake/android-r16b.cmake \
#     -DNDK_ROOT=/path/to/android-ndk-r16b \
#     -DANDROID_ABI=arm64-v8a -DANDROID_API=21

# Linux (not Android) as the system name: CMake's built-in Android support
# assumes a modern NDK and would fight the flags below. The compiler still
# targets *-linux-android, so __ANDROID__ is defined and the output is a normal
# Android ELF.
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)
set(ANDROID TRUE)

if(NOT NDK_ROOT)
  # The environment is the only channel that reaches CMake's try_compile
  # scratch project, where this file is re-included without the -D cache.
  if(DEFINED ENV{ANDROID_NDK_ROOT})
    set(NDK_ROOT "$ENV{ANDROID_NDK_ROOT}")
  elseif(DEFINED ENV{NDK_ROOT})
    set(NDK_ROOT "$ENV{NDK_ROOT}")
  else()
    message(FATAL_ERROR
      "Set ANDROID_NDK_ROOT in the environment, or pass "
      "-DNDK_ROOT=<path to android-ndk-r16b>")
  endif()
endif()
set(NDK_ROOT "${NDK_ROOT}" CACHE PATH "Android NDK root")

if(NOT ANDROID_ABI)
  set(ANDROID_ABI "arm64-v8a")
endif()
if(NOT ANDROID_API)
  set(ANDROID_API "21")
endif()

if(ANDROID_ABI STREQUAL "arm64-v8a")
  set(_triple "aarch64-linux-android")
  set(_arch "arm64")
elseif(ANDROID_ABI STREQUAL "armeabi-v7a")
  set(_triple "arm-linux-androideabi")
  set(_arch "arm")
elseif(ANDROID_ABI STREQUAL "x86_64")
  set(_triple "x86_64-linux-android")
  set(_arch "x86_64")
elseif(ANDROID_ABI STREQUAL "x86")
  set(_triple "i686-linux-android")
  set(_arch "x86")
else()
  message(FATAL_ERROR "Unsupported ANDROID_ABI: ${ANDROID_ABI}")
endif()

if(CMAKE_HOST_WIN32)
  set(_exe ".exe")
else()
  set(_exe "")
endif()

set(_host     "${NDK_ROOT}/toolchains/llvm/prebuilt/windows-x86_64")
set(_llvm_bin "${_host}/bin")
set(_gcc_root "${NDK_ROOT}/toolchains/${_triple}-4.9/prebuilt/windows-x86_64")
set(_gcc_bin  "${_gcc_root}/bin")

set(_sysroot "${NDK_ROOT}/sysroot")
set(_platlib "${NDK_ROOT}/platforms/android-${ANDROID_API}/arch-${_arch}/usr/lib")
set(_cxxinc  "${NDK_ROOT}/sources/cxx-stl/llvm-libc++/include")
set(_cxxlib  "${NDK_ROOT}/sources/cxx-stl/llvm-libc++/libs/${ANDROID_ABI}")
set(_libgcc  "${_gcc_root}/lib/gcc/${_triple}/4.9.x")

set(_gnu_ld "${_gcc_bin}/${_triple}-ld${_exe}")
file(TO_CMAKE_PATH "${_gnu_ld}" _gnu_ld)

if(NOT EXISTS "${_llvm_bin}/clang++${_exe}")
  message(FATAL_ERROR "No clang++ under ${_llvm_bin}")
endif()
if(NOT EXISTS "${_gnu_ld}")
  message(FATAL_ERROR "No GNU ld at ${_gnu_ld}")
endif()

set(CMAKE_C_COMPILER   "${_llvm_bin}/clang${_exe}")
set(CMAKE_CXX_COMPILER "${_llvm_bin}/clang++${_exe}")
set(CMAKE_AR           "${_gcc_bin}/${_triple}-ar${_exe}")
set(CMAKE_RANLIB       "${_gcc_bin}/${_triple}-ranlib${_exe}")
set(CMAKE_STRIP        "${_gcc_bin}/${_triple}-strip${_exe}")

# -B (not just -L) so the driver finds crtbegin_dynamic.o / crtend_android.o,
# which r16b keeps in the per-platform lib dir rather than in the sysroot.
set(_common "--target=${_triple}${ANDROID_API} --sysroot=${_sysroot} -I${_sysroot}/usr/include/${_triple} -B${_platlib}")

set(CMAKE_C_FLAGS_INIT   "${_common}")
set(CMAKE_CXX_FLAGS_INIT "${_common} -stdlib=libc++ -I${_cxxinc}")

# Whether to link the C++ runtime statically.
#
# Default ON. Loading libc++_shared.so puts its symbols in the global scope,
# where they interpose on any vendor .so that carries its own C++ runtime. That
# breaks exception handling inside such a library and shows up as a bare
# "libc++abi: terminating / Aborted" instead of a usable error. Statically
# linking libc++ keeps the process's C++ runtime private to our binary.
# Turn OFF when linking into an Android app that already ships libc++_shared.
option(MIE_STATIC_LIBCXX "Link libc++ statically" ON)

if(MIE_STATIC_LIBCXX)
  set(_stdlib_link "-stdlib=libc++ -static-libstdc++")
  set(_stdlib_libs "-lm -lc -ldl")
else()
  set(_stdlib_link "-stdlib=libc++")
  set(_stdlib_libs "-lc++_shared -lm -lc -ldl")
endif()

set(_link "-fuse-ld=${_gnu_ld} -L${_platlib} -L${_cxxlib} -L${_libgcc} ${_stdlib_link}")
# Android 5+ refuses to exec a non-PIE binary ("only position independent
# executables are supported"), and CMake will not add -pie on its own because
# it believes it is targeting plain Linux.
set(CMAKE_EXE_LINKER_FLAGS_INIT    "${_link} -pie")
set(CMAKE_SHARED_LINKER_FLAGS_INIT "${_link}")

# libgcc comes from the 4.9 toolchain that clang 5 relies on for builtins.
set(CMAKE_CXX_STANDARD_LIBRARIES "${_stdlib_libs}")
set(CMAKE_C_STANDARD_LIBRARIES   "-lm -lc -ldl")

set(CMAKE_FIND_ROOT_PATH "${_sysroot}")
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
