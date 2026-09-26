#!/system/bin/sh
# Compile .tflite models into .dla files ON THE DEVICE with the NeuroPilot SDK's
# ncc-tflite, for the mie kMtkDla backend.
#
# The SDK ships ncc-tflite per chip (neuron_sdk/<chip>/bin). The host build is a
# Linux ELF that cannot run here, but the on-device build works from a rooted or
# normal shell because it only needs the system libs. Deploy once, from e.g.
# neuron_sdk/<chip>:
#
#   adb push bin/. /data/local/tmp/np/bin/
#   adb push lib/. /data/local/tmp/np/lib/
#   adb shell chmod +x /data/local/tmp/np/bin/*
#
# Usage:
#   dla_compile.sh <model.tflite> [more.tflite ...]
#
# Environment:
#   NP_DIR   toolchain directory            (default /data/local/tmp/np)
#   ARCH     target architecture            (default mdla3.0; run
#                                           `ncc-tflite --arch=?` for the list,
#                                           the tested parts take mdla3.0)
#   EXTRA    extra ncc-tflite options, e.g. EXTRA=--use-sw-dilated-conv
#
# The .dla is written next to the model as <model>.<arch>.dla, with the compiler
# log beside it as .log. Exit status is non-zero if any model failed.
#
# Two failure modes worth recognising:
#   "MDLA: unsupported operation"      an op the NPU cannot run at all (a TFLite
#                                      custom op such as the SSD NMS tail) —
#                                      cut it out of the model or use kMtk/kGpu
#   "DilationRate unsupported ..."     the atrous receptive field is too large
#                                      for the hardware (DeepLabV3 ASPP at
#                                      rate 18); --use-sw-dilated-conv helps
#                                      smaller rates only

set -u

NP_DIR=${NP_DIR:-/data/local/tmp/np}
ARCH=${ARCH:-mdla3.0}
EXTRA=${EXTRA:-}
NCC="$NP_DIR/bin/ncc-tflite"

export LD_LIBRARY_PATH="$NP_DIR/lib"

if [ $# -eq 0 ]; then
  sed -n '2,40p' "$0" | sed 's/^# \{0,1\}//'
  exit 2
fi

if [ ! -x "$NCC" ]; then
  echo "no ncc-tflite at $NCC (deploy the SDK's per-chip bin/ and lib/ first)" >&2
  exit 2
fi

failed=0
for model in "$@"; do
  if [ ! -f "$model" ]; then
    echo "[skip] no such file: $model"
    failed=1
    continue
  fi
  case "$model" in
    *.tflite) ;;
    *)
      echo "[skip] ncc-tflite only accepts a .tflite extension: $model"
      echo "       it checks the extension, not the content — copy it first,"
      echo "       e.g. cp '$model' '${model%.*}.tflite'"
      failed=1
      continue
      ;;
  esac

  out="${model%.*}.$ARCH.dla"
  log="$out.log"
  rm -f "$out"

  # EXTRA is deliberately word-split: it exists to carry several options.
  # shellcheck disable=SC2086
  "$NCC" --arch="$ARCH" $EXTRA -d "$out" "$model" > "$log" 2>&1
  if [ -f "$out" ]; then
    echo "[ok]   $(basename "$model") -> $(basename "$out") ($(stat -c %s "$out") bytes)"
  else
    echo "[fail] $(basename "$model")"
    grep -E 'unsupported|too large|cannot|not supported|[Ee]rror|fail' "$log" \
      | head -5 | sed 's/^/       /'
    failed=1
  fi
done

exit $failed
