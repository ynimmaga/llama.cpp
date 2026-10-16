"""KERNEL_SPEC for GGML_OP_SCALE, consumed by aot_compile.py:

    CPATH=... LIBRARY_PATH=... python aot_compile.py scale_spec.py <out_dir>

Second, independent op added purely through this spec file plus scale_bias_kernel.py -- zero
changes to ggml-triton-ext's or llama.cpp's C++, proving a user really can plug in support for an
arbitrary GGML op. Genuinely different shape from rms_norm_spec.py: elementwise over the whole
(flattened) tensor rather than one-program-per-row, and two attrs instead of one.
"""
from scale_bias_kernel import scale_bias_kernel

BLOCKS = [256, 1024, 4096, 16384]
PTR = {"f32": "*fp32", "f16": "*fp16"}
DTYPES = ["f32", "f16"]

SWEEP = [
    {"dtypes": {"input.0": dt, "output.0": dt}, "block": {"BLOCK_M": block}}
    for dt in DTYPES
    for block in BLOCKS
]

KERNEL_SPEC = {
    "ggml_op": "GGML_OP_SCALE",
    "kernel_fn": scale_bias_kernel,
    "num_inputs": 1,
    "num_outputs": 1,
    # ggml_scale_impl packs { s, b } into op_params as two back-to-back floats -- see ggml.c.
    "attrs": [
        {"name": "scale", "op_params_offset": 0, "type": "f32"},
        {"name": "bias", "op_params_offset": 1, "type": "f32"},
    ],
    "outputs": [{"shape_like": {"tensor": "input", "index": 0}}],
    # One flat grid over the whole tensor (all 4 axes multiplied, padding to 1 for rank < 4),
    # ceil-divided by BLOCK_M -- unlike RMS_NORM's one-axis-per-row grid.
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
        "scale": "fp32",
        "bias": "fp32",
        "BLOCK": "constexpr",
    },
    "constexprs_template": lambda cfg: {"BLOCK": cfg["block"]["BLOCK_M"]},
    # Must match scale_bias_kernel's positional parameter order (BLOCK is constexpr, not a runtime
    # argument); Triton's XPU launcher always appends two scratch pointers after the user args.
    "args": [
        {"kind": "input_ptr", "index": 0},
        {"kind": "output_ptr", "index": 0},
        {"kind": "shape", "tensor": "input", "index": 0, "axis": -4},
        {"kind": "shape", "tensor": "input", "index": 0, "axis": -3},
        {"kind": "shape", "tensor": "input", "index": 0, "axis": -2},
        {"kind": "shape", "tensor": "input", "index": 0, "axis": -1},
        {"kind": "attr", "name": "scale"},
        {"kind": "attr", "name": "bias"},
        {"kind": "scratch"},
        {"kind": "scratch"},
    ],
}
