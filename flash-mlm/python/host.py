import torch

import triton

from python.mlm_kernel import _mlm_main_kernel


def flash_attn_mlm(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    context_len: int,
    scale: float,
    is_mla: bool = False,
    context_batch_size: int | None = None,
    cu_seqlens_q: torch.Tensor | None = None,
    cu_seqlens_kv: torch.Tensor | None = None,
    block_m: int = 64,
    block_n: int = 64,
) -> torch.Tensor:
    """Basic host launcher for _mlm_main_kernel.

    Expected layouts:
      q, k, v: [B, H, N, D]
      k_cache, v_cache: [B, H, C, D]
    """

    if not (
        q.is_cuda and k.is_cuda and v.is_cuda and k_cache.is_cuda and v_cache.is_cuda
    ):
        raise ValueError("All tensors must be CUDA tensors")

    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q/k/v must be rank-4 tensors [B, H, N, D]")

    if k_cache.ndim != 4 or v_cache.ndim != 4:
        raise ValueError("k_cache/v_cache must be rank-4 tensors [B, H, C, D]")

    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q, k, v must have the same shape")

    b, h, n, d = q.shape
    bc, hc, c, dc = k_cache.shape
    if (bc, hc, dc) != (b, h, d):
        raise ValueError("k_cache shape must be [B, H, C, D] matching q's B/H/D")
    if v_cache.shape != k_cache.shape:
        raise ValueError("v_cache shape must match k_cache shape")

    if context_len < 0 or context_len > c:
        raise ValueError("context_len must be in [0, k_cache.shape[2]]")

    if context_batch_size is None:
        context_batch_size = bc
    if context_batch_size < 1:
        raise ValueError("context_batch_size must be >= 1")

    if d not in (16, 32, 64, 128, 256):
        raise ValueError("HEAD_DIM must be one of {16, 32, 64, 128, 256}")

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    k_cache = k_cache.contiguous()
    v_cache = v_cache.contiguous()

    out = torch.empty_like(q)

    # Current kernel signature includes these args; they are not yet consumed in-kernel.
    if cu_seqlens_q is None:
        cu_seqlens_q = torch.zeros(1, device=q.device, dtype=torch.int32)
    if cu_seqlens_kv is None:
        cu_seqlens_kv = torch.zeros(1, device=q.device, dtype=torch.int32)

    grid = (triton.cdiv(n, block_n), b * h, 1)

    _mlm_main_kernel[grid](
        q,
        k,
        v,
        out,
        k_cache,
        v_cache,
        b,
        context_batch_size,
        h,
        context_len,
        scale,
        n,
        cu_seqlens_q,
        cu_seqlens_kv,
        is_mla=is_mla,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        HEAD_DIM=d,
    )

    return out
