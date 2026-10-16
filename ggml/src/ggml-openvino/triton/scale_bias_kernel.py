"""Triton kernel matching ggml's GGML_OP_SCALE: y = x * scale + bias, elementwise.

Unlike rms_norm_kernel.py (one program per row, whole row in one tile), this kernel treats the
(contiguous -- ggml_scale requires ggml_is_padded_1d) tensor as flat and tiles over total element
count, a genuinely different grid/shape pattern -- proving the generic mechanism isn't just "one
row per program".
"""
import triton
import triton.language as tl


@triton.jit
def scale_bias_kernel(x_ptr, y_ptr, D0, D1, D2, D3, scale, bias, BLOCK: tl.constexpr):
    n_elements = D0 * D1 * D2 * D3
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = x * scale + bias
    tl.store(y_ptr + offs, y.to(y_ptr.dtype.element_ty), mask=mask)
