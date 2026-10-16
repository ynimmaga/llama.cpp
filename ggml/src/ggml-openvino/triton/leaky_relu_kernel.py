"""Triton kernel for GGML_OP_LEAKY_RELU: y = x if x > 0 else x * negative_slope, elementwise."""
import triton
import triton.language as tl


@triton.jit
def leaky_relu_kernel(x_ptr, y_ptr, D0, D1, D2, D3, slope, BLOCK: tl.constexpr):
    n_elements = D0 * D1 * D2 * D3
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = tl.where(x > 0.0, x, x * slope)
    tl.store(y_ptr + offs, y.to(y_ptr.dtype.element_ty), mask=mask)
