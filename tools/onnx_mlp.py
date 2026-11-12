"""Write a small multilayer perceptron as an ONNX file, with numpy only.

ONNX stores a network as a graph: named inputs and outputs, the weights
("initializers") and a list of operations that connect them. The file itself
is a protocol buffer, a simple binary format of numbered fields. The networks
here are a chain of linear layers with tanh between them, which needs only two
ONNX operations:

    Gemm   y = x W^T + b   (a torch.nn.Linear layer; transB=1 because PyTorch
                            stores W as (outputs, inputs))
    Tanh   y = tanh(x)

Writing the few fields they use by hand is shorter than depending on the onnx
package, and it does not change when PyTorch's own exporter does. The field
numbers below are those of onnx/onnx.proto.
"""

from __future__ import annotations

import numpy as np

OPSET = 17  # ONNX operator set; Unity's Sentis reads 7 to 25
IR_VERSION = 8  # the file format version that goes with opset 17
FLOAT = 1  # TensorProto.DataType
ATTRIBUTE_INT = 2  # AttributeProto.AttributeType

Layers = list[tuple[np.ndarray, np.ndarray]]  # per layer: W (outputs, inputs) and b (outputs,)


# ---------------------------------------------------------------------------
# Protocol buffer encoding: every field is a key (field number and wire type)
# followed by a value, either a variable-length integer or a length and bytes.
# ---------------------------------------------------------------------------

def _varint(n: int) -> bytes:
    if n < 0:
        n += 1 << 64  # int64 fields keep negative numbers in two's complement
    out = bytearray()
    while True:
        low, n = n & 0x7F, n >> 7
        out.append(low | (0x80 if n else 0))
        if not n:
            return bytes(out)


def _int(field: int, value: int) -> bytes:
    return _varint(field << 3 | 0) + _varint(value)


def _bytes(field: int, data: bytes) -> bytes:
    return _varint(field << 3 | 2) + _varint(len(data)) + data


def _text(field: int, text: str) -> bytes:
    return _bytes(field, text.encode("utf-8"))


# ---------------------------------------------------------------------------
# The ONNX messages used
# ---------------------------------------------------------------------------

def _tensor(name: str, array: np.ndarray) -> bytes:
    """TensorProto: dims 1, data_type 2, name 8, raw_data 9 (little-endian float32)."""
    a = np.ascontiguousarray(array, dtype="<f4")
    return b"".join([*(_int(1, int(d)) for d in a.shape), _int(2, FLOAT), _text(8, name),
                     _bytes(9, a.tobytes())])


def _value_info(name: str, dims: list[int | str]) -> bytes:
    """ValueInfoProto of a float tensor. A dimension given as text is named, so any size goes."""
    shape = b"".join(_bytes(1, _text(2, d) if isinstance(d, str) else _int(1, int(d))) for d in dims)
    tensor_type = _int(1, FLOAT) + _bytes(2, shape)  # elem_type 1, shape 2
    return _text(1, name) + _bytes(2, _bytes(1, tensor_type))  # name 1, type 2 (tensor_type 1)


def _node(op: str, inputs: list[str], outputs: list[str], name: str, **ints: int) -> bytes:
    """NodeProto: input 1, output 2, name 3, op_type 4, attribute 5 (name 1, i 3, type 20)."""
    attributes = [_bytes(5, _text(1, k) + _int(3, v) + _int(20, ATTRIBUTE_INT)) for k, v in ints.items()]
    return b"".join([*(_text(1, i) for i in inputs), *(_text(2, o) for o in outputs),
                     _text(3, name), _text(4, op), *attributes])


def check_layers(layers: Layers) -> None:
    for k, (w, b) in enumerate(layers):
        if w.ndim != 2 or b.shape != (w.shape[0],):
            raise ValueError(f"layer {k}: weight {w.shape} and bias {b.shape} do not fit")
        if k and w.shape[1] != layers[k - 1][0].shape[0]:
            raise ValueError(f"layer {k} takes {w.shape[1]} inputs but layer {k - 1} gives "
                             f"{layers[k - 1][0].shape[0]}")


def mlp_model(layers: Layers, input_name: str = "obs", output_name: str = "logits",
              metadata: dict[str, str] | None = None, producer: str = "arenamind") -> bytes:
    """The bytes of an .onnx file: linear layers with tanh between them, any batch size."""
    check_layers(layers)
    nodes, weights = [], []
    x = input_name
    for k, (w, b) in enumerate(layers):
        last = k == len(layers) - 1
        w_name, b_name = f"layer{k}.weight", f"layer{k}.bias"
        weights += [_tensor(w_name, w), _tensor(b_name, b)]
        y = output_name if last else f"layer{k}.linear"
        nodes.append(_node("Gemm", [x, w_name, b_name], [y], f"layer{k}.gemm", transB=1))
        if not last:
            x = f"layer{k}.tanh"
            nodes.append(_node("Tanh", [y], [x], x))
    n_in, n_out = layers[0][0].shape[1], layers[-1][0].shape[0]
    # GraphProto: node 1, name 2, initializer 5, input 11, output 12
    graph = b"".join([*(_bytes(1, n) for n in nodes), _text(2, "policy"),
                      *(_bytes(5, t) for t in weights),
                      _bytes(11, _value_info(input_name, ["batch", n_in])),
                      _bytes(12, _value_info(output_name, ["batch", n_out]))])
    # ModelProto: ir_version 1, producer_name 2, graph 7, opset_import 8 (version 2),
    # metadata_props 14 (key 1, value 2)
    meta = [_bytes(14, _text(1, k) + _text(2, v)) for k, v in (metadata or {}).items()]
    return b"".join([_int(1, IR_VERSION), _text(2, producer), _bytes(7, graph),
                     _bytes(8, _int(2, OPSET)), *meta])


def forward(layers: Layers, x: np.ndarray) -> np.ndarray:
    """What the .onnx file computes, in numpy (float64), for checking it."""
    h = np.asarray(x, dtype=np.float64)
    for k, (w, b) in enumerate(layers):
        h = h @ np.asarray(w, dtype=np.float64).T + np.asarray(b, dtype=np.float64)
        if k < len(layers) - 1:
            h = np.tanh(h)
    return h
