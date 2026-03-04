import torch
import os

import triton
import triton.language as tl


def _get_triton_backend() -> str | None:
    try:
        return triton.runtime.driver.active.get_current_target().backend
    except RuntimeError:
        return None


def is_hip():
    return _get_triton_backend() == "hip"


def is_cuda():
    return _get_triton_backend() == "cuda"


def supports_host_descriptor():
    return is_cuda() and torch.cuda.get_device_capability()[0] >= 9


def is_blackwell():
    return is_cuda() and torch.cuda.get_device_capability()[0] == 10


def is_hopper():
    return is_cuda() and torch.cuda.get_device_capability()[0] == 9


@triton.jit
def _maybe_make_tensor_desc(desc_or_ptr, shape, strides, block_shape):
    if isinstance(desc_or_ptr, tl.tensor_descriptor):
        return desc_or_ptr
    return tl.make_tensor_descriptor(desc_or_ptr, shape, strides, block_shape)


@triton.jit
def _maybe_write_tensor_descriptor(
    desc_or_ptr,
    row_start,
    value,
    row_mask,
    HEAD_DIM: tl.constexpr,
):
    if isinstance(desc_or_ptr, tl.tensor_descriptor):
        desc_or_ptr.store([row_start, 0], value)
    else:
        row_offsets = row_start + tl.arange(0, row_mask.shape[0])
        col_offsets = tl.arange(0, HEAD_DIM)
        out_ptrs = desc_or_ptr + row_offsets[:, None] * HEAD_DIM + col_offsets[None, :]
        tl.store(out_ptrs, value, mask=row_mask[:, None])
