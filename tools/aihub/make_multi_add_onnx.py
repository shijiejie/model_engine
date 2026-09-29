"""Rebuild the multi_add toy (4 fp32 [1,8,8,3] inputs -> 2 outputs) as ONNX.

Matches testdata/multi_add.bin behavior: with all inputs filled to 1.0, both
outputs are all-3.0 (checksum 5cba39c5), the cross-backend reference.
Topology: t = a+b ; x = t+c ; y = t+d  (shared intermediate, 2 ADD nodes).

Usage:  python tools/aihub/make_multi_add_onnx.py   (writes ./multi_add.onnx)
"""
import onnx
from onnx import helper, TensorProto

shape = [1, 8, 8, 3]
inputs = [helper.make_tensor_value_info(n, TensorProto.FLOAT, shape)
          for n in ("a", "b", "c", "d")]
outputs = [helper.make_tensor_value_info(n, TensorProto.FLOAT, shape)
           for n in ("x", "y")]

nodes = [
    helper.make_node("Add", ["a", "b"], ["t"]),
    helper.make_node("Add", ["t", "c"], ["x"]),
    helper.make_node("Add", ["t", "d"], ["y"]),
]
graph = helper.make_graph(nodes, "multi_add", inputs, outputs)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
model.ir_version = 8
onnx.checker.check_model(model)
onnx.save(model, "multi_add.onnx")
print("saved multi_add.onnx")
