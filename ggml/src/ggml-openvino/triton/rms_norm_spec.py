"""KERNEL_SPEC for GGML_OP_RMS_NORM, consumed by aot_compile.py:

    CPATH=... LIBRARY_PATH=... python aot_compile.py rms_norm_spec.py <out_dir>

Reference example for writing a new op's spec: each field here maps directly onto manifest.json
(see registry.h / README), except `kernel_fn`/`sweep`/`signature_template`/`constexprs_template`,
which only exist to drive compilation and don't appear in the emitted manifest.
"""
from rms_norm_kernel import rms_norm_kernel

BLOCKS = [32 << i for i in range(9)]  # 32 .. 8192
PTR = {"f32": "*fp32", "f16": "*fp16"}
DTYPE_PAIRS = [("f32", "f32"), ("f16", "f16"), ("f16", "f32"), ("f32", "f16")]

# One compiled variant per (dtype pair, block size): BLOCK is a Triton compile-time constant, so
# the whole row must fit in one tile, and the GPU plugin may hand this op an f16 input with an f32
# output (or vice versa) when it runs the model in mixed precision.
SWEEP = [
    {"dtypes": {"input.0": x_dt, "output.0": y_dt}, "block": {"BLOCK_M": block}}
    for x_dt, y_dt in DTYPE_PAIRS
    for block in BLOCKS
]

KERNEL_SPEC = {
    "ggml_op": "GGML_OP_RMS_NORM",
    "kernel_fn": rms_norm_kernel,
    "num_inputs": 1,
    "num_outputs": 1,
    # eps lives at op_params[0] (a float) for every ggml_rms_norm node -- see ggml.c.
    "attrs": [{"name": "eps", "op_params_offset": 0, "type": "f32"}],
    "outputs": [{"shape_like": {"tensor": "input", "index": 0}}],
    # One program per row: grid = D0*D1*D2 (the input's outer dims, flattened), no block_param
    # (divisor 1, one program per element of the extent). Axes are always
    # negative (-1 = last/contiguous dim; -4,-3,-2 = the three dims before it): a rank < 4 input
    # (ggml tensors are conceptually rank <= 4) pads the missing leading axes with size 1 / stride 0
    # automatically, so this manifest works unmodified for any input of rank 1-4.
    "grid": [{
        "extent_from": [
            {"tensor": "input", "index": 0, "axis": -4},
            {"tensor": "input", "index": 0, "axis": -3},
            {"tensor": "input", "index": 0, "axis": -2},
        ],
    }],
    # BLOCK_M is the column tile: one tile must span the whole row, so a variant is usable only if
    # BLOCK_M >= the row length (input axis -1); the smallest such variant is chosen.
    "block_covers": [{"block_param": "BLOCK_M", "tensor": "input", "index": 0, "axis": -1}],
    "sweep": SWEEP,
    "signature_template": lambda cfg: {
        "x_ptr": PTR[cfg["dtypes"]["input.0"]],
        "y_ptr": PTR[cfg["dtypes"]["output.0"]],
        "n_cols": "i32",
        "eps": "fp32",
        "D1": "i32",
        "D2": "i32",
        "sx0": "i32",
        "sx1": "i32",
        "sx2": "i32",
        "sy0": "i32",
        "sy1": "i32",
        "sy2": "i32",
        "BLOCK": "constexpr",
    },
    "constexprs_template": lambda cfg: {"BLOCK": cfg["block"]["BLOCK_M"]},
    # Must match rms_norm_kernel's positional parameter order exactly (BLOCK is constexpr, not a
    # runtime argument). Triton's own XPU launcher always appends two scratch pointers after the
    # user args, hence the trailing pair here.
    "args": [
        {"kind": "input_ptr", "index": 0},
        {"kind": "output_ptr", "index": 0},
        {"kind": "shape", "tensor": "input", "index": 0, "axis": -1},
        {"kind": "attr", "name": "eps"},
        {"kind": "shape", "tensor": "input", "index": 0, "axis": -3},
        {"kind": "shape", "tensor": "input", "index": 0, "axis": -2},
        {"kind": "stride", "tensor": "input", "index": 0, "axis": -4},
        {"kind": "stride", "tensor": "input", "index": 0, "axis": -3},
        {"kind": "stride", "tensor": "input", "index": 0, "axis": -2},
        {"kind": "stride", "tensor": "output", "index": 0, "axis": -4},
        {"kind": "stride", "tensor": "output", "index": 0, "axis": -3},
        {"kind": "stride", "tensor": "output", "index": 0, "axis": -2},
        {"kind": "scratch"},
        {"kind": "scratch"},
    ],
}
