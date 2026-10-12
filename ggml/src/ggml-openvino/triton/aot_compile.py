"""Ahead-of-time compile the Triton RMS norm kernel into SPIR-V variants for the ggml-openvino bridge.

    CPATH=... LIBRARY_PATH=... python aot_compile.py <out_dir>

BLOCK is a compile-time constant (the row is reduced in one tile), so one variant is emitted per
power-of-two row width and per (input, output) element type pair: the GPU plugin may hand the op an
f16 input with an f32 output when it runs the model in f16. manifest.json records what Triton's own XPU launcher
needs to run each variant (triton/backends/intel/driver.c): kernel name, build flags, work-group
shape (num_warps x threads_per_warp), shared local memory, and scratch sizes.
"""
import json
import os
import sys

import triton

from rms_norm_kernel import rms_norm_kernel

BLOCKS = [32 << i for i in range(9)]  # 32 .. 8192
PTR = {"f32": "*fp32", "f16": "*fp16"}
DTYPE_PAIRS = [("f32", "f32"), ("f16", "f16"), ("f16", "f32"), ("f32", "f16")]


def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    target = triton.runtime.driver.active.get_current_target()
    manifest = {"kernel": "rms_norm", "arch": target.arch.get("arch"), "variants": []}
    for x_dt, y_dt in DTYPE_PAIRS:
        for block in BLOCKS:
            src = triton.compiler.ASTSource(
                fn=rms_norm_kernel,
                signature={"x_ptr": PTR[x_dt], "y_ptr": PTR[y_dt], "n_cols": "i32", "eps": "fp32", "D1": "i32", "D2": "i32",
                           "sx0": "i32", "sx1": "i32", "sx2": "i32", "sy0": "i32", "sy1": "i32", "sy2": "i32",
                           "BLOCK": "constexpr"},
                constexprs={"BLOCK": block})
            k = triton.compile(src, target=target)
            md = k.metadata
            fname = f"rms_norm_{x_dt}_{y_dt}_{block}.spv"
            with open(os.path.join(out_dir, fname), "wb") as f:
                f.write(k.asm["spv"])
            manifest["variants"].append({
                "x_dtype": x_dt, "y_dtype": y_dt, "block": block, "file": fname, "name": md.name,
                "build_flags": md.build_flags, "num_warps": md.num_warps,
                "threads_per_warp": md.threads_per_warp, "shared": md.shared,
                "global_scratch_size": md.global_scratch_size, "profile_scratch_size": md.profile_scratch_size,
                # user args in order (constexpr BLOCK is not an argument), then Triton's two scratch pointers
                "args": ["ptr", "ptr", "i32", "f32"] + ["i32"] * 8,
            })
            print(f"{fname}: warps={md.num_warps} tpw={md.threads_per_warp} shared={md.shared} "
                  f"scratch={md.global_scratch_size}/{md.profile_scratch_size} flags='{md.build_flags}'")
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "triton_kernels")
