import torch

import triton

from python.host_utils import (
    PackMetadata,
    pack_for_kernel,
    pad_packed_main_tensors_for_mlm_compressed,
)
from python.mlm_kernel import _mlm_compressed_kernel, _mlm_main_kernel


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


def flash_attn_mlm_compressed(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    num_heads: int,
    q_meta: PackMetadata,
    total_context_len: int,
    cu_seqlens_kv: torch.Tensor,
    scale: float,
    is_mla: bool = False,
    context_batch_size: int | None = None,
    block_m: int = 64,
    block_n: int = 64,
) -> torch.Tensor:
    """Host launcher for packed-sequence `_mlm_compressed_kernel`.

    Expected layouts:
      q, k, v: padded [B, H, N, D]
      k_cache, v_cache:
        - MLA: [total_context_len, D]
        - non-MLA: [H * total_context_len, D]

    Args:
        q: Query tensor with padded layout [B, H, N, D].
        k: Key tensor with padded layout [B, H, N, D].
        v: Value tensor with padded layout [B, H, N, D].
        k_cache: Packed cache K tensor.
            - MLA: [total_context_len, D]
            - non-MLA: [H * total_context_len, D]
        v_cache: Packed cache V tensor with same shape as k_cache.
        num_heads: Number of attention heads (H).
        q_meta: Precomputed packing metadata for q/k/v.
        total_context_len: Total packed context token count across context batches.
        cu_seqlens_kv: Cumulative context lengths [context_batch_size+1].
        scale: Attention scaling factor applied to QK logits.
        is_mla: Whether to run in MLA latent mode (shared K/V latent layout).
        context_batch_size: Number of context batches represented by cu_seqlens_kv.
            If None, inferred from cu_seqlens_kv.
        block_m: K/V tile size for the Triton kernel.
        block_n: Q tile size for the Triton kernel.

    Returns:
        Packed output tensor with shape [H * total_q_len, D].
    """

    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q/k/v must be rank-4 padded tensors [B, H, N, D]")
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q, k, v must have the same shape")

    b, h, n, d = q.shape
    if h != num_heads:
        raise ValueError("q/k/v head dimension must match num_heads")

    if not (
        q.is_cuda
        and k.is_cuda
        and v.is_cuda
        and k_cache.is_cuda
        and v_cache.is_cuda
        and cu_seqlens_kv.is_cuda
    ):
        raise ValueError("All tensors must be CUDA tensors")

    if q_meta.B != b or q_meta.N != n:
        raise ValueError("q_meta shape metadata does not match q/k/v padded shape")

    q = pack_for_kernel(q, q_meta, flatten_for_kernel=True)
    if is_mla:
        k = pack_for_kernel(k[:, :1, :, :], q_meta, flatten_for_kernel=True)
        v = k
    else:
        k = pack_for_kernel(k, q_meta, flatten_for_kernel=True)
        v = pack_for_kernel(v, q_meta, flatten_for_kernel=True)

    total_q_len_unpadded = q_meta.total_tokens

    if q_meta.batch_ids_block_n != block_n:
        raise ValueError("q_meta was built with different block_n")
    cu_seqlens_q = q_meta.cu_seqlens

    q, k, v, total_q_len = pad_packed_main_tensors_for_mlm_compressed(
        q,
        k,
        v,
        num_heads=num_heads,
        is_mla=is_mla,
        total_q_len_unpadded=total_q_len_unpadded,
        cu_seqlens_q=cu_seqlens_q,
        block_m=block_m,
        block_n=block_n,
    )

    batch_ids_q = q_meta.batch_ids_q
    q_tile_starts_q = q_meta.q_tile_starts_q

    if not (batch_ids_q.is_cuda and cu_seqlens_q.is_cuda):
        raise ValueError("batch_ids_q and cu_seqlens_q must be CUDA tensors")

    if k_cache.ndim != 2 or v_cache.ndim != 2:
        raise ValueError("k_cache/v_cache must be rank-2 packed tensors")
    if v_cache.shape != k_cache.shape:
        raise ValueError("v_cache shape must match k_cache shape")

    if num_heads < 1:
        raise ValueError("num_heads must be >= 1")
    if total_q_len < 0 or total_context_len < 0:
        raise ValueError("total_q_len and total_context_len must be >= 0")

    y_dim_q, d = q.shape
    if y_dim_q != num_heads * total_q_len:
        raise ValueError("q first dim must equal num_heads * total_q_len")

    expected_main_rows = total_q_len if is_mla else num_heads * total_q_len
    if k.shape[0] != expected_main_rows or v.shape[0] != expected_main_rows:
        raise ValueError(
            "k/v first dim mismatch: expected "
            f"{expected_main_rows} rows for is_mla={is_mla}"
        )

    if k_cache.shape[1] != d:
        raise ValueError("k_cache/v_cache second dim must match q/k/v head_dim")

    expected_context_rows = (
        total_context_len if is_mla else num_heads * total_context_len
    )
    if k_cache.shape[0] != expected_context_rows:
        raise ValueError(
            "k_cache/v_cache first dim mismatch: expected "
            f"{expected_context_rows} rows for is_mla={is_mla}"
        )

    if d not in (16, 32, 64, 128, 256):
        raise ValueError("HEAD_DIM must be one of {16, 32, 64, 128, 256}")

    if cu_seqlens_q.ndim != 1 or cu_seqlens_kv.ndim != 1:
        raise ValueError("cu_seqlens_q and cu_seqlens_kv must be rank-1")
    if cu_seqlens_q.numel() < 2 or cu_seqlens_kv.numel() < 2:
        raise ValueError("cu_seqlens_q and cu_seqlens_kv must have at least 2 elements")

    if context_batch_size is None:
        context_batch_size = int(cu_seqlens_kv.numel() - 1)
    if context_batch_size < 1:
        raise ValueError("context_batch_size must be >= 1")
    if context_batch_size + 1 > cu_seqlens_kv.numel():
        raise ValueError("context_batch_size is incompatible with cu_seqlens_kv length")

    num_q_tiles = int(q_tile_starts_q.numel())
    if batch_ids_q.ndim != 1 or batch_ids_q.numel() != num_q_tiles:
        raise ValueError(
            "batch_ids_q must be rank-1 with exactly one entry per query tile"
        )
    if q_tile_starts_q.ndim != 1:
        raise ValueError("q_tile_starts_q must be rank-1")

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    k_cache = k_cache.contiguous()
    v_cache = v_cache.contiguous()
    batch_ids_q = batch_ids_q.contiguous()
    q_tile_starts_q = q_tile_starts_q.contiguous()
    cu_seqlens_q = cu_seqlens_q.contiguous()
    cu_seqlens_kv = cu_seqlens_kv.contiguous()

    out = torch.empty_like(q)
    grid = (num_q_tiles, num_heads, 1)

    _mlm_compressed_kernel[grid](
        q,
        k,
        v,
        out,
        k_cache,
        v_cache,
        context_batch_size,
        num_heads,
        total_context_len,
        scale,
        total_q_len,
        batch_ids_q,
        q_tile_starts_q,
        cu_seqlens_q,
        cu_seqlens_kv,
        is_mla=is_mla,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        HEAD_DIM=d,
    )

    if total_q_len != total_q_len_unpadded:
        out = (
            out.view(num_heads, total_q_len, d)[:, :total_q_len_unpadded, :]
            .reshape(num_heads * total_q_len_unpadded, d)
            .contiguous()
        )

    return out
