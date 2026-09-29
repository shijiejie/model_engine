"""Strip the TFLite_Detection_PostProcess (CUSTOM) tail from a .tflite in place.

tflite2onnx and AI Hub both reject TFLite custom ops, and the post-process op is
always the LAST node of the main subgraph, so we cut it: repoint the subgraph
outputs to the pre-NMS tensors that fed it and drop the node. This is pure byte
patching — the `tflite` package is read-only, so vector positions are located by
probing vtable slots.

Usage:  python tools/aihub/strip_nms.py detect.tflite detect_backbone.tflite
"""
import sys

import tflite
import flatbuffers
import flatbuffers.encode as encode
from flatbuffers.packer import uoffset, int32

B = lambda tab, pos: encode.Get(uoffset, tab.Bytes, pos)  # uint32 at pos
I = lambda tab, pos: encode.Get(int32, tab.Bytes, pos)  # int32 at pos


def vec_len_prefix(tab, vlen):
    """Yield length-prefix positions of vectors holding `vlen` elements."""
    for slot in range(16):
        o = tab.Offset(slot)
        if not o:
            continue
        field = tab.Pos + o              # absolute position of the field
        prefix = field + B(tab, field)   # ... of the vector's length prefix
        if prefix + 4 <= len(tab.Bytes) and I(tab, prefix) == vlen:
            yield prefix


def strip(src, dst):
    with open(src, "rb") as f:
        buf = bytearray(f.read())        # mutable buffer for in-place patching
    im = tflite.Model.GetRootAs(buf, 0)
    sg = im.Subgraphs(0)
    tab = sg._tab

    n = sg.OperatorsLength()
    last = sg.Operators(n - 1)
    if im.OperatorCodes(last.OpcodeIndex()).BuiltinCode() != tflite.BuiltinOperator.CUSTOM:
        raise SystemExit(f"{src}: last op is not CUSTOM")
    in_ids = [last.Inputs(i) for i in range(last.InputsLength())]
    n_out = last.OutputsLength()
    names = [sg.Tensors(i).Name().decode() for i in range(sg.TensorsLength())]
    print("CUSTOM inputs:", [(i, names[i]) for i in in_ids])

    # Dedupe by prefix position (vtable slots can alias the same vector).
    ops_prefix = next(iter(dict.fromkeys(vec_len_prefix(tab, n))))
    out_prefix = next(iter(dict.fromkeys(vec_len_prefix(tab, n_out))))

    encode.Write(int32, buf, ops_prefix, n - 1)           # drop the CUSTOM node
    new_out = in_ids[:2]                                  # keep box encodings + scores
    encode.Write(int32, buf, out_prefix, len(new_out))    # outputs: N -> 2
    for i, tid in enumerate(new_out):
        encode.Write(int32, buf, out_prefix + 4 + 4 * i, tid)

    with open(dst, "wb") as f:
        f.write(buf)
    print(f"saved {dst}, {len(buf)} bytes; new outputs:",
          [(i, names[i]) for i in new_out])


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: strip_nms.py <in.tflite> <out.tflite>")
    strip(sys.argv[1], sys.argv[2])
