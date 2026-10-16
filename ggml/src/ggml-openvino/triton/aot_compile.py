"""Generic AOT compiler: builds any Triton kernel's KERNEL_SPEC into a ggml-triton-ext manifest v2
package (manifest.json + .spv files), ready to drop under TRITON_EXT_KERNEL_ROOT/<name>/.

    CPATH=... LIBRARY_PATH=... python aot_compile.py <spec_module.py> <out_dir>

<spec_module.py> is any importable Python file exposing a KERNEL_SPEC dict -- see rms_norm_spec.py
for a worked example and for what each field means. This script has no per-op knowledge: adding
support for a new GGML op is purely a matter of writing a new spec module, no changes here.
"""
import importlib.util
import json
import os
import sys


def load_spec(path):
    spec = importlib.util.spec_from_file_location("kernel_spec", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.KERNEL_SPEC


def compile_variants(spec, out_dir):
    import triton

    target = triton.runtime.driver.active.get_current_target()
    variants = []
    for cfg in spec["sweep"]:
        signature = spec["signature_template"](cfg)
        constexprs = spec["constexprs_template"](cfg)
        src = triton.compiler.ASTSource(fn=spec["kernel_fn"], signature=signature, constexprs=constexprs)
        k = triton.compile(src, target=target)
        md = k.metadata

        tag = "_".join(str(v) for v in list(cfg["dtypes"].values()) + list(cfg["block"].values()))
        fname = f"{spec['ggml_op'].lower()}_{tag}.spv"
        with open(os.path.join(out_dir, fname), "wb") as f:
            f.write(k.asm["spv"])

        args = list(spec["args"])
        if md.shared and not any(a.get("kind") == "shared_mem" for a in args):
            args.append({"kind": "shared_mem"})  # Triton passes local memory as a trailing kernel arg

        variants.append({
            "dtypes": cfg["dtypes"],
            "block": cfg["block"],
            "name": md.name,
            "build_flags": md.build_flags,
            "file": fname,
            "num_warps": md.num_warps,
            "threads_per_warp": md.threads_per_warp,
            "shared": md.shared,
            "global_scratch_size": md.global_scratch_size,
            "profile_scratch_size": md.profile_scratch_size,
            "args": args,
        })
        print(f"{fname}: warps={md.num_warps} tpw={md.threads_per_warp} shared={md.shared} "
              f"scratch={md.global_scratch_size}/{md.profile_scratch_size} flags='{md.build_flags}'")
    return variants


def main(spec_path, out_dir):
    spec = load_spec(spec_path)
    os.makedirs(out_dir, exist_ok=True)
    manifest = {
        "ggml_op": spec["ggml_op"],
        "num_inputs": spec["num_inputs"],
        "num_outputs": spec["num_outputs"],
        "attrs": spec.get("attrs", []),
        "outputs": spec["outputs"],
        "grid": spec["grid"],
        "block_covers": spec.get("block_covers", []),
        "variants": compile_variants(spec, out_dir),
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "triton_kernels")
