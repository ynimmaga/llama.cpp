"""KERNEL_SPEC for GGML_OP_LEAKY_RELU, consumed by aot_compile.py:

    python aot_compile.py leaky_relu_spec.py <root>/leaky_relu

An op the ggml-openvino backend has no built-in translator for: adding it takes only this spec and
leaky_relu_kernel.py, with no C++ changes in ggml-triton-ext or llama.cpp.
"""
from leaky_relu_kernel import leaky_relu_kernel

BLOCKS = [256, 1024, 4096, 16384]
PTR = {"f32": "*fp32", "f16": "*fp16"}
DTYPES = ["f32", "f16"]

SWEEP = [
    {"dtypes": {"input.0": dt, "output.0": dt}, "block": {"BLOCK_M": block}}
    for dt in DTYPES
    for block in BLOCKS
]

KERNEL_SPEC = {
    "ggml_op": "GGML_OP_LEAKY_RELU",
    "kernel_fn": leaky_relu_kernel,
    "num_inputs": 1,
    "num_outputs": 1,
    # ggml_leaky_relu stores negative_slope as a float at op_params[0].
    "attrs": [{"name": "negative_slope", "op_params_offset": 0, "type": "f32"}],
    "outputs": [{"shape_like": {"tensor": "input", "index": 0}}],
    # One flat grid over the whole tensor, ceil-divided by BLOCK_M.
    "grid": [{
        "extent_from": [
            {"tensor": "input", "index": 0, "axis": -4},
            {"tensor": "input", "index": 0, "axis": -3},
            {"tensor": "input", "index": 0, "axis": -2},
            {"tensor": "input", "index": 0, "axis": -1},
        ],
        "block_param": "BLOCK_M",
    }],
    "sweep": SWEEP,
    "signature_template": lambda cfg: {
        "x_ptr": PTR[cfg["dtypes"]["input.0"]],
        "y_ptr": PTR[cfg["dtypes"]["output.0"]],
        "D0": "i32", "D1": "i32", "D2": "i32", "D3": "i32",
        "slope": "fp32",
        "BLOCK": "constexpr",
    },
    "constexprs_template": lambda cfg: {"BLOCK": cfg["block"]["BLOCK_M"]},
    "args": [
        {"kind": "input_ptr", "index": 0},
        {"kind": "output_ptr", "index": 0},
        {"kind": "shape", "tensor": "input", "index": 0, "axis": -4},
        {"kind": "shape", "tensor": "input", "index": 0, "axis": -3},
        {"kind": "shape", "tensor": "input", "index": 0, "axis": -2},
        {"kind": "shape", "tensor": "input", "index": 0, "axis": -1},
        {"kind": "attr", "name": "negative_slope"},
        {"kind": "scratch"},
        {"kind": "scratch"},
    ],
}
