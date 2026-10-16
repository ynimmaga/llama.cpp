"""Triton RMS norm matching ggml's GGML_OP_RMS_NORM: y = x / sqrt(mean(x^2, last dim) + eps).

Tensors are viewed as 4D [D0, D1, D2, n_cols] with element strides for the three outer dims (the
last dim is contiguous), so a view the GPU plugin keeps in place is read without a copy.
One program per row; BLOCK covers the whole row, so the reduction is a single tl.sum.
"""
import triton
import triton.language as tl


@triton.jit
def rms_norm_kernel(x_ptr, y_ptr, n_cols, eps, D1, D2, sx0, sx1, sx2, sy0, sy1, sy2, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    i2 = row % D2
    i1 = (row // D2) % D1
    i0 = row // (D2 * D1)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.load(x_ptr + i0 * sx0 + i1 * sx1 + i2 * sx2 + offs, mask=mask, other=0.0).to(tl.float32)
    mean_sq = tl.sum(x * x, axis=0) / n_cols
    y = x * (1.0 / tl.sqrt(mean_sq + eps))
    tl.store(y_ptr + i0 * sy0 + i1 * sy1 + i2 * sy2 + offs, y.to(y_ptr.dtype.element_ty), mask=mask)
