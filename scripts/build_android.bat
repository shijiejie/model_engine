@echo off
rem ===========================================================================
rem  mie - build with ndk-build and package the result
rem
rem    scripts\build_android.bat [options]
rem
rem  Options:
rem    /mtk            also build the MediaTek NeuroPilot backend
rem                    (implies /api27; needs third_party\mtk\include)
rem    /dla            also build the MediaTek DLA (Neuron Runtime V2) backend
rem                    (needs neuron\api\RuntimeV2.h - see /sdk below)
rem    /sdk <dir>      the NeuroPilot SDK's per-chip include dir for /dla
rem                    (e.g. <sdk>\neuron_sdk\<chip>\include). Overrides
rem                    MIE_MTK_SDK_INCLUDE_DIR and is remembered in
rem                    scripts\local_paths.bat for the next run.
rem    /qnnnative      also build the Qualcomm QNN native C API backend
rem                    (needs QnnInterface.h - see /qnnsdk below)
rem    /qnnsdk <dir>   the QNN SDK include dir for /qnnnative (the one holding
rem                    QnnInterface.h and System\QnnSystemContext.h). Overrides
rem                    MIE_QNN_SDK_INCLUDE_DIR and is remembered in
rem                    scripts\local_paths.bat for the next run.
rem    /serial <id>    adb device serial for /push (required when more than
rem                    one device is attached)
rem    /api27          build against API level 27 instead of 21
rem    /abi <abi>      target ABI (default arm64-v8a)
rem    /debug          build with APP_OPTIM=debug
rem    /zip            also produce dist\mie-<abi>.zip
rem    /push           adb push the runtime set to /data/local/tmp/mie
rem    /no-clean       skip the pre-build clean (see the note below)
rem    /nopause        never pause
rem    /help
rem
rem  NOTE on /no-clean: ndk-build keeps objects in obj\local\<abi>\ with neither
rem  the platform nor the flags in the path, so changing APP_PLATFORM or
rem  MIE_ENABLE_MTK does NOT invalidate them - a stale object tree silently
rem  produces a binary built for the previous configuration. Clean is therefore
rem  the default.
rem ===========================================================================

setlocal EnableDelayedExpansion
set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%.." || exit /b 1

set "ABI=arm64-v8a"
set "WANT_MTK=0"
set "WANT_DLA=0"
set "WANT_API27=0"
set "WANT_ZIP=0"
set "WANT_PUSH=0"
set "DO_CLEAN=1"
set "WANT_PAUSE=1"
set "OPTIM=release"
set "HAD_ARGS=0"
set "WANT_SDK_DIR="
set "WANT_QNN_NATIVE=0"
set "WANT_QNN_SDK_DIR="
set "ADB_SERIAL="

:parse
if "%~1"=="" goto :parsed
set "HAD_ARGS=1"
if /i "%~1"=="/mtk"      ( set "WANT_MTK=1"   & set "WANT_API27=1" & shift & goto :parse )
if /i "%~1"=="/dla"      ( set "WANT_DLA=1"   & shift & goto :parse )
if /i "%~1"=="/sdk"      ( set "WANT_SDK_DIR=%~2" & shift & shift & goto :parse )
if /i "%~1"=="/qnnnative" ( set "WANT_QNN_NATIVE=1" & shift & goto :parse )
if /i "%~1"=="/qnnsdk"    ( set "WANT_QNN_SDK_DIR=%~2" & shift & shift & goto :parse )
if /i "%~1"=="/serial"   ( set "ADB_SERIAL=%~2"   & shift & shift & goto :parse )
if /i "%~1"=="/api27"    ( set "WANT_API27=1" & shift & goto :parse )
if /i "%~1"=="/debug"    ( set "OPTIM=debug"  & shift & goto :parse )
if /i "%~1"=="/zip"      ( set "WANT_ZIP=1"   & shift & goto :parse )
if /i "%~1"=="/push"     ( set "WANT_PUSH=1"  & shift & goto :parse )
if /i "%~1"=="/no-clean" ( set "DO_CLEAN=0"   & shift & goto :parse )
if /i "%~1"=="/nopause"  ( set "WANT_PAUSE=0" & shift & goto :parse )
if /i "%~1"=="/help"     goto :usage
if /i "%~1"=="/?"        goto :usage
if /i "%~1"=="/abi"      ( set "ABI=%~2" & shift & shift & goto :parse )
echo [mie] unknown option: %~1
goto :usage

:parsed
rem Local machine paths (NDK root, the DLA SDK include dir), written by this
rem script's /sdk handling. Every assignment inside is guarded with
rem `if not defined`, so an explicit environment variable or /sdk flag always
rem wins; the file is a pure fallback and safe to delete.
if exist "%SCRIPT_DIR%local_paths.bat" call "%SCRIPT_DIR%local_paths.bat"

rem Double-clicked windows have no arguments; keep them open to show the result.
if "%HAD_ARGS%"=="0" set "WANT_PAUSE=1"

rem --- locate the NDK -------------------------------------------------------
set "NDK=%ANDROID_NDK_ROOT%"
if not defined NDK set "NDK=%ANDROID_NDK%"
if not defined NDK (
  echo [mie] ERROR: set ANDROID_NDK_ROOT to your NDK first, e.g.
  echo            set ANDROID_NDK_ROOT=D:\path\to\android-ndk-r16b
  goto :fail
)
if not exist "%NDK%\ndk-build.cmd" (
  echo [mie] ERROR: no ndk-build.cmd under "%NDK%"
  goto :fail
)

rem --- dependencies ----------------------------------------------------------
if not exist "third_party\tflite-dist\libs\android\%ABI%\libtensorflowlite_c.so" (
  echo [mie] ERROR: TFLite C API lib missing for %ABI% under
  echo        third_party\tflite-dist\libs\android\ - place the prebuilt
  echo        libtensorflowlite_c.so and libtensorflowlite_gpu_delegate.so there.
  goto :fail
)

if "%WANT_MTK%"=="1" if not exist "third_party\mtk\include\NeuroPilotTFLiteShim.h" (
  echo [mie] ERROR: /mtk needs third_party\mtk\include\NeuroPilotTFLiteShim.h
  echo            That header is confidential MediaTek material and is NOT
  echo            fetched automatically - copy it from your licensed SDK.
  goto :fail
)

rem The DLA backend needs the SDK's neuron/api headers (it includes
rem RuntimeV2.h). Priority: /sdk flag, then MIE_MTK_SDK_INCLUDE_DIR from the
rem environment (or scripts\local_paths.bat), then a project-local copy.
set "MTK_SDK_INC="
if defined WANT_SDK_DIR set "MTK_SDK_INC=%WANT_SDK_DIR%"
if not defined MTK_SDK_INC set "MTK_SDK_INC=%MIE_MTK_SDK_INCLUDE_DIR%"
if not defined MTK_SDK_INC if exist "third_party\mtk\sdk_include\neuron\api\RuntimeV2.h" set "MTK_SDK_INC=%CD%\third_party\mtk\sdk_include"
if "%WANT_DLA%"=="1" (
  if not defined MTK_SDK_INC (
    echo [mie] ERROR: /dla needs the NeuroPilot SDK's neuron/api headers.
    echo            Pass /sdk ^<dir^> or set MIE_MTK_SDK_INCLUDE_DIR to the
    echo            SDK's per-chip include dir, e.g.
    echo            ^<sdk^>\neuron_sdk\^<chip^>\include, or copy them to
    echo            third_party\mtk\sdk_include\ ^(they are confidential MediaTek
    echo            material and are NOT fetched automatically^).
    goto :fail
  )
  if not exist "%MTK_SDK_INC%\neuron\api\RuntimeV2.h" (
    echo [mie] ERROR: no neuron\api\RuntimeV2.h under "%MTK_SDK_INC%"
    goto :fail
  )
)

rem The QNN native backend needs the QNN SDK's C API headers (it includes
rem QnnInterface.h and System\QnnSystemContext.h). Priority: /qnnsdk flag, then
rem MIE_QNN_SDK_INCLUDE_DIR from the environment (or scripts\local_paths.bat).
set "QNN_SDK_INC="
if defined WANT_QNN_SDK_DIR set "QNN_SDK_INC=%WANT_QNN_SDK_DIR%"
if not defined QNN_SDK_INC set "QNN_SDK_INC=%MIE_QNN_SDK_INCLUDE_DIR%"
if "%WANT_QNN_NATIVE%"=="1" (
  if not defined QNN_SDK_INC (
    echo [mie] ERROR: /qnnnative needs the QNN SDK's C API headers.
    echo            Pass /qnnsdk ^<dir^> or set MIE_QNN_SDK_INCLUDE_DIR to the
    echo            QNN SDK include dir holding QnnInterface.h, e.g.
    echo            ^<sdk^>\include\QNN. They are confidential Qualcomm material
    echo            and are NOT fetched automatically.
    goto :fail
  )
  if not exist "%QNN_SDK_INC%\QnnInterface.h" (
    echo [mie] ERROR: no QnnInterface.h under "%QNN_SDK_INC%"
    goto :fail
  )
)

set "PLATFORM=android-21"
if "%WANT_API27%"=="1" set "PLATFORM=android-27"
set "BACKEND=cpu, gpu, qnn"
if "%WANT_MTK%"=="1" set "BACKEND=%BACKEND%, mtk"
if "%WANT_DLA%"=="1" set "BACKEND=%BACKEND%, dla"
if "%WANT_QNN_NATIVE%"=="1" set "BACKEND=%BACKEND%, qnnnative"

echo.
echo [mie] NDK      : %NDK%
echo [mie] ABI      : %ABI%
echo [mie] platform : %PLATFORM%
echo [mie] backend  : %BACKEND%
echo [mie] optim    : %OPTIM%
echo.

rem --- build -----------------------------------------------------------------
if "%DO_CLEAN%"=="1" (
  echo [mie] cleaning previous objects...
  call "%NDK%\ndk-build.cmd" clean >nul 2>&1
)

set "BUILD_ARGS=APP_ABI=%ABI% APP_PLATFORM=%PLATFORM% APP_OPTIM=%OPTIM% MIE_ENABLE_MTK=%WANT_MTK% MIE_ENABLE_MTK_DLA=%WANT_DLA% MIE_ENABLE_QNN_NATIVE=%WANT_QNN_NATIVE%"
if "%WANT_MTK%"=="1" set "BUILD_ARGS=%BUILD_ARGS% MIE_MTK_INCLUDE_DIR=%CD%\third_party\mtk\include"
if "%WANT_DLA%"=="1" set "BUILD_ARGS=%BUILD_ARGS% MIE_MTK_SDK_INCLUDE_DIR=%MTK_SDK_INC%"
if "%WANT_QNN_NATIVE%"=="1" set "BUILD_ARGS=%BUILD_ARGS% MIE_QNN_SDK_INCLUDE_DIR=%QNN_SDK_INC%"

echo [mie] building...
call "%NDK%\ndk-build.cmd" %BUILD_ARGS%
if errorlevel 1 (
  echo.
  echo [mie] BUILD FAILED
  goto :fail
)

if not exist "libs\%ABI%\mie_run" (
  echo [mie] ERROR: ndk-build reported success but libs\%ABI%\mie_run is missing
  goto :fail
)

rem --- package ---------------------------------------------------------------
set "DIST=dist\mie-%ABI%"
if exist "%DIST%" rd /s /q "%DIST%"
mkdir "%DIST%\include\mie" 2>nul
mkdir "%DIST%\lib"         2>nul
mkdir "%DIST%\bin"         2>nul

copy /y "include\mie\engine.h"        "%DIST%\include\mie\" >nul
copy /y "obj\local\%ABI%\libmie.a"    "%DIST%\lib\"          >nul
copy /y "libs\%ABI%\*.so"             "%DIST%\bin\"          >nul
copy /y "libs\%ABI%\mie_run"          "%DIST%\bin\"          >nul

set "TFLITE_VER=unknown"
if exist "third_party\tflite-dist\VERSION" set /p TFLITE_VER=<"third_party\tflite-dist\VERSION"

rem %DATE%/%TIME% are locale- and codepage-dependent - a Chinese locale writes the
rem weekday as mojibake into the file. Get an unambiguous ISO stamp instead.
set "STAMP=unknown"
for /f "usebackq delims=" %%i in (`powershell -NoProfile -Command "Get-Date -Format yyyy-MM-ddTHH:mm:ss"`) do set "STAMP=%%i"

rem Written line by line rather than as one redirected block: parentheses and
rem conditional text inside a ( ) block need escaping and are easy to get wrong.
set "BUILDTXT=%DIST%\BUILD.txt"
echo mie - %ABI% build> "%BUILDTXT%"
echo.>> "%BUILDTXT%"
echo built     : %STAMP%>> "%BUILDTXT%"
echo abi       : %ABI%>> "%BUILDTXT%"
echo platform  : %PLATFORM%>> "%BUILDTXT%"
echo backend   : %BACKEND%>> "%BUILDTXT%"
echo optim     : %OPTIM%>> "%BUILDTXT%"
echo tflite    : %TFLITE_VER%>> "%BUILDTXT%"
echo ndk       : %NDK%>> "%BUILDTXT%"
echo.>> "%BUILDTXT%"
echo layout:>> "%BUILDTXT%"
echo   include/  public headers, add -Iinclude>> "%BUILDTXT%"
echo   lib/      libmie.a, link it or list the .a directly>> "%BUILDTXT%"
echo   bin/      mie_run plus the TFLite runtime .so it needs at runtime>> "%BUILDTXT%"
echo.>> "%BUILDTXT%"
echo Runtime note: mie_run needs libtensorflowlite_c.so and>> "%BUILDTXT%"
echo libtensorflowlite_gpu_delegate.so on LD_LIBRARY_PATH. The Android linker does>> "%BUILDTXT%"
echo NOT search the executable's own directory, so keep bin/ together and run:>> "%BUILDTXT%"
echo   cd bin ^&^& LD_LIBRARY_PATH=. [ADSP_LIBRARY_PATH=^<skels^>] ./mie_run model.tflite qnn>> "%BUILDTXT%"
echo.>> "%BUILDTXT%"
echo Integration note: libmie.a is built with APP_STL=c++_static. Link it>> "%BUILDTXT%"
echo into an app built with APP_STL := c++_static as well - Config passes>> "%BUILDTXT%"
echo std::string and std::vector across the API boundary, so mixing a>> "%BUILDTXT%"
echo static and a shared libc++ would break it.>> "%BUILDTXT%"

echo.
echo [mie] packaged: %DIST%
echo        include\  lib\  bin\

if "%WANT_ZIP%"=="1" (
  if exist "dist\mie-%ABI%.zip" del /q "dist\mie-%ABI%.zip"
  powershell -NoProfile -Command "Compress-Archive -Path '%DIST%\*' -DestinationPath 'dist\mie-%ABI%.zip'"
  if errorlevel 1 ( echo [mie] ERROR: zip failed & goto :fail )
  echo [mie] zip     : dist\mie-%ABI%.zip
)

rem Remember the machine-local paths for the next run. Written only when /sdk
rem or /qnnsdk was given this run; every assignment is guarded, so the file
rem never overrides an explicit environment variable or flag. The values
rem persisted are the RESOLVED ones (%NDK% / %MTK_SDK_INC% / %QNN_SDK_INC%),
rem which the dependency checks above have already validated.
if defined WANT_SDK_DIR goto :save_paths
if defined WANT_QNN_SDK_DIR goto :save_paths
goto :save_paths_done

:save_paths
echo [mie] saving local paths to scripts\local_paths.bat
> "%SCRIPT_DIR%local_paths.bat" echo @echo off
>> "%SCRIPT_DIR%local_paths.bat" echo rem Written by build_android.bat. Machine-local, safe to delete.
>> "%SCRIPT_DIR%local_paths.bat" echo if not defined ANDROID_NDK_ROOT set "ANDROID_NDK_ROOT=%NDK%"
if defined MTK_SDK_INC >> "%SCRIPT_DIR%local_paths.bat" echo if not defined MIE_MTK_SDK_INCLUDE_DIR set "MIE_MTK_SDK_INCLUDE_DIR=%MTK_SDK_INC%"
if defined QNN_SDK_INC >> "%SCRIPT_DIR%local_paths.bat" echo if not defined MIE_QNN_SDK_INCLUDE_DIR set "MIE_QNN_SDK_INCLUDE_DIR=%QNN_SDK_INC%"

:save_paths_done

if "%WANT_PUSH%"=="1" (
  echo.
  set "ADB=adb"
  if defined ADB_SERIAL set "ADB=adb -s !ADB_SERIAL!"
  if not defined ADB_SERIAL (
    set /a NDEV=0
    for /f "skip=1 tokens=2" %%s in ('adb devices 2^>nul') do (
      if /i "%%s"=="device" set /a NDEV+=1
    )
    if !NDEV! GTR 1 (
      echo [mie] ERROR: !NDEV! adb devices attached - pass /serial ^<id^>
      goto :fail
    )
  )
  echo [mie] pushing runtime set to /data/local/tmp/mie ...
  !ADB! shell mkdir -p /data/local/tmp/mie 2>nul
  !ADB! push "%DIST%\bin\." /data/local/tmp/mie/ >nul
  if errorlevel 1 ( echo [mie] ERROR: adb push failed - is a device connected? & goto :fail )
  !ADB! shell chmod +x /data/local/tmp/mie/mie_run
  echo [mie] pushed to /data/local/tmp/mie
)

echo.
echo [mie] OK
popd
if "%WANT_PAUSE%"=="1" pause
exit /b 0

:fail
popd
echo.
if "%WANT_PAUSE%"=="1" pause
exit /b 1

:usage
echo.
echo mie - build with ndk-build and package the result
echo.
echo   scripts\build_android.bat [options]
echo.
echo   /mtk            build the MediaTek NeuroPilot backend, implies /api27
echo   /dla            build the MediaTek DLA (Neuron Runtime V2) backend
echo   /sdk ^<dir^>      SDK per-chip include dir for /dla, remembered for
echo                   next runs (scripts\local_paths.bat)
echo   /qnnnative      build the Qualcomm QNN native C API backend
echo   /qnnsdk ^<dir^>   QNN SDK include dir for /qnnnative (holding
echo                   QnnInterface.h), remembered for next runs
echo   /serial ^<id^>    adb device serial for /push
echo   /api27          target API level 27 instead of 21
echo   /abi ^<abi^>      target ABI, default arm64-v8a
echo   /debug          APP_OPTIM=debug
echo   /zip            also write dist\mie-^<abi^>.zip
echo   /push           adb push the runtime set to /data/local/tmp/mie
echo   /no-clean       skip the pre-build clean
echo   /nopause        never pause
echo   /help           this text
echo.
echo Requires ANDROID_NDK_ROOT (or scripts\local_paths.bat remembering it).
echo.
popd
if "%WANT_PAUSE%"=="1" pause
exit /b 1
