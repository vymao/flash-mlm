import torch

import triton
from triton.tools.tensor_descriptor import TensorDescriptor

from flash_mlm.host_utils import (
    PackMetadata,
    make_contiguous,
    pack_and_pad_main_tensors_for_mlm_compressed,
    require_cuda_tensors,
    validate_cu_seqlens_rank1_min2,
    validate_head_dim_supported,
    validate_packed_cache_shapes,
    validate_qkv_same_shape_rank4,
)
from flash_mlm.kernel_utils import supports_host_descriptor
from flash_mlm.mlm_kernel import (
    _MLM_BLOCK_M_OPTIONS,
    _mlm_compressed_kernel,
    _mlm_compressed_kernel_auto_block_m,
    _mlm_main_kernel,
    _mlm_main_kernel_auto_tile,
)
from flash_mlm.cache import InferenceCache


def _make_host_desc(x: torch.Tensor, rows: int, head_dim: int) -> TensorDescriptor:
    return TensorDescriptor(
        x,
        shape=[rows, head_dim],
        strides=[head_dim, 1],
        block_shape=[1, 1],
    )


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
    block_n: int = 32,
    auto_tune_tiles: bool = False,
) -> torch.Tensor:
    """Host launcher for dense (non-packed) MLM attention kernel.

    Expected layouts:
      - q, k, v: [B, H, N, D]
      - k_cache, v_cache: [B_cache, H, C, D], where B_cache is either
        ``context_batch_size`` or 1 (shared cache across query batch)

    Args:
        q: Query tensor with shape [B, H, N, D].
        k: Key tensor with shape [B, H, N, D].
        v: Value tensor with shape [B, H, N, D].
        k_cache: Cache K tensor with shape [B_cache, H, C, D].
        v_cache: Cache V tensor with shape [B_cache, H, C, D].
        context_len: Active cache length C_active used from k_cache/v_cache.
            Must satisfy ``0 <= context_len <= C``.
        scale: Softmax scaling factor applied to QK logits.
        is_mla: Whether to run in MLA latent mode.
        context_batch_size: Number of context sequences represented by
            cache tensors. If None, inferred from ``k_cache.shape[0]``.
        cu_seqlens_q: Unused in the dense path. Kept for API compatibility.
        cu_seqlens_kv: Unused in the dense path. Kept for API compatibility.
        block_m: K/V tile size when ``auto_tune_tiles=False``.
        block_n: Q tile size for kernel launch and grid construction.
        auto_tune_tiles: If True, use autotuned BLOCK_M (BLOCK_N fixed by
            ``block_n``); otherwise use provided ``block_m``/``block_n``.

    Returns:
        Output tensor with shape [B, H, N, D] on the same device/dtype as ``q``.
    """

    require_cuda_tensors(q, k, v, k_cache, v_cache)

    if k_cache.ndim != 4 or v_cache.ndim != 4:
        raise ValueError("k_cache/v_cache must be rank-4 tensors [B, H, C, D]")

    b, h, n, d = validate_qkv_same_shape_rank4(
        q,
        k,
        v,
        error_prefix="q/k/v",
    )
    bc, hc, c, dc = k_cache.shape
    if (hc, dc) != (h, d):
        raise ValueError("k_cache shape must have H/D matching q")
    if v_cache.shape != k_cache.shape:
        raise ValueError("v_cache shape must match k_cache shape")

    if context_batch_size is None:
        context_batch_size = bc
    if context_batch_size < 1:
        raise ValueError("context_batch_size must be >= 1")
    if bc not in (1, context_batch_size):
        raise ValueError("k_cache batch dim must be 1 or match context_batch_size")

    if context_len < 0 or context_len > c:
        raise ValueError("context_len must be in [0, k_cache.shape[2]]")

    # Kept for API compatibility with compressed path / future varlen main path.
    _ = (cu_seqlens_q, cu_seqlens_kv)

    validate_head_dim_supported(d)

    q, k, v, k_cache, v_cache = make_contiguous(q, k, v, k_cache, v_cache)

    out = torch.empty_like(q)

    if supports_host_descriptor():
        y_dim = b * h * n
        y_dim_context = (
            context_batch_size * context_len
            if is_mla
            else context_batch_size * h * context_len
        )
        desc_q = _make_host_desc(q, y_dim, d)
        desc_k = _make_host_desc(k, y_dim, d)
        desc_v = _make_host_desc(v, y_dim, d)
        desc_o = _make_host_desc(out, y_dim, d)
        desc_k_cache = _make_host_desc(k_cache, y_dim_context, d)
        desc_v_cache = _make_host_desc(v_cache, y_dim_context, d)
    else:
        desc_q = q
        desc_k = k
        desc_v = v
        desc_o = out
        desc_k_cache = k_cache
        desc_v_cache = v_cache

    grid = (triton.cdiv(n, block_n), b * h, 1)

    if auto_tune_tiles:
        _mlm_main_kernel_auto_tile[grid](
            desc_q,
            desc_k,
            desc_v,
            desc_o,
            desc_k_cache,
            desc_v_cache,
            b,
            context_batch_size,
            h,
            context_len,
            scale,
            n,
            is_mla=is_mla,
            BLOCK_N=block_n,
            HEAD_DIM=d,
        )
    else:
        _mlm_main_kernel[grid](
            desc_q,
            desc_k,
            desc_v,
            desc_o,
            desc_k_cache,
            desc_v_cache,
            b,
            context_batch_size,
            h,
            context_len,
            scale,
            n,
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
    num_heads: int,
    q_meta: PackMetadata,
    scale: float,
    inference_cache: InferenceCache | None = None,
    layer_id: int | str | None = None,
    is_mla: bool = False,
    context_batch_size: int | None = None,
    block_m: int = 64,
    block_n: int = 32,
    auto_tune_tiles: bool = False,
    prefill: bool = False,
    causal_query_seq_attn: bool = False,
) -> torch.Tensor:
    """Host launcher for packed-sequence `_mlm_compressed_kernel`.

    Expected layouts:
      q, k, v: padded [B, H, N, D]
      cached k_cache, v_cache:
        - MLA: [total_context_len, D]
        - non-MLA: [H * total_context_len, D]

    Args:
        q: Query tensor with padded layout [B, H, N, D].
        k: Key tensor with padded layout [B, H, N, D].
        v: Value tensor with padded layout [B, H, N, D].
        num_heads: Number of attention heads (H).
        q_meta: Precomputed packing metadata for q/k/v.
        scale: Attention scaling factor applied to QK logits.
        is_mla: Whether to run in MLA latent mode (shared K/V latent layout).
        context_batch_size: Number of context batches represented by cached
            ``cu_seqlens_kv``. If None, inferred from cached values.
        block_m: K/V tile size for the Triton kernel.
        block_n: Q tile size for the Triton kernel.
        inference_cache: Optional layer-scoped inference cache containing
            prefetched packed context KV and reusable packing workspaces. If
            None, this call runs with empty context only.
        layer_id: Optional layer identifier used to fetch/store prefetched KV
            context when ``inference_cache`` is provided.
        prefill: If True, store this call's packed main K/V as the next cached
            context for ``layer_id`` after the kernel launch completes.
        causal_query_seq_attn: If True, each query subsequence can attend
            to packed main tokens from all previous query subsequences in the
            same batch (in addition to cached context).

    Returns:
        Packed output tensor with shape [H * total_q_len, D].
    """

    b, h, n, d = validate_qkv_same_shape_rank4(
        q,
        k,
        v,
        error_prefix="q/k/v",
    )
    if h != num_heads:
        raise ValueError("q/k/v head dimension must match num_heads")

    if prefill and inference_cache is None:
        raise ValueError("prefill=True requires inference_cache")

    if (inference_cache is None) != (layer_id is None):
        raise ValueError(
            "inference_cache and layer_id must be provided together or both omitted"
        )

    if q_meta.B != b or q_meta.N != n:
        raise ValueError("q_meta shape metadata does not match q/k/v padded shape")

    total_q_len_unpadded = q_meta.total_tokens

    if q_meta.batch_ids_block_n != block_n:
        raise ValueError("q_meta was built with different block_n")
    cu_seqlens_q = q_meta.cu_seqlens

    use_layer_cache = inference_cache is not None and layer_id is not None
    entry = None
    if use_layer_cache:
        try:
            entry = inference_cache.get_kv_cache(
                layer_id,
                is_mla=is_mla,
                num_heads=num_heads,
                head_dim=d,
                dtype=q.dtype,
                device=q.device,
            )
        except ValueError:
            # Missing or incompatible cache entry falls back to empty context.
            entry = None

    if entry is None:
        # First prefill call can run without pre-existing context cache.
        total_context_len = 0
        context_batch_size = b if context_batch_size is None else context_batch_size
        cu_seqlens_kv = torch.zeros(
            context_batch_size + 1,
            device=q.device,
            dtype=torch.int32,
        )
        kv_rows = total_context_len if is_mla else num_heads * total_context_len
        k_cache = torch.empty((kv_rows, d), device=q.device, dtype=q.dtype)
        v_cache = torch.empty_like(k_cache)
    else:
        k_cache = entry.k_cache
        v_cache = entry.v_cache
        total_context_len = entry.total_context_len
        cu_seqlens_kv = entry.cu_seqlens_kv
        if context_batch_size is None:
            context_batch_size = entry.context_batch_size

    require_cuda_tensors(q, k, v, k_cache, v_cache, cu_seqlens_kv)

    pad_block_m = max(_MLM_BLOCK_M_OPTIONS) if auto_tune_tiles else block_m
    packing_cache = inference_cache.packing if inference_cache is not None else None
    if prefill and use_layer_cache:
        layer_key = str(layer_id)
        k_buffer_name = f"layer:{layer_key}:k_cache"
        v_buffer_name = f"layer:{layer_key}:v_cache"
    else:
        k_buffer_name = "k"
        v_buffer_name = "v"

    q, k, v, total_q_len = pack_and_pad_main_tensors_for_mlm_compressed(
        q,
        k,
        v,
        q_meta=q_meta,
        num_heads=num_heads,
        is_mla=is_mla,
        block_m=pad_block_m,
        block_n=block_n,
        packing_cache=packing_cache,
        q_buffer_name="q",
        k_buffer_name=k_buffer_name,
        v_buffer_name=v_buffer_name,
    )

    if prefill and use_layer_cache:
        # Store padded packed main K/V for descriptor-safe tail loads in future calls.
        # cu_seqlens remains logical (unpadded) sequence lengths.
        inference_cache.prefill_kv_cache(
            layer_id,
            k_cache=k,
            v_cache=v,
            total_context_len=int(total_q_len),
            cu_seqlens_kv=cu_seqlens_q,
            context_batch_size=int(b),
            is_mla=is_mla,
            num_heads=num_heads,
            head_dim=d,
        )

    batch_ids_q = q_meta.batch_ids_q
    q_tile_starts_q = q_meta.q_tile_starts_q

    if not (batch_ids_q.is_cuda and cu_seqlens_q.is_cuda):
        raise ValueError("batch_ids_q and cu_seqlens_q must be CUDA tensors")

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

    validate_packed_cache_shapes(
        k_cache,
        v_cache,
        num_heads=num_heads,
        total_context_len=int(total_context_len),
        head_dim=d,
        is_mla=is_mla,
    )
    validate_head_dim_supported(d)
    validate_cu_seqlens_rank1_min2(cu_seqlens_q, cu_seqlens_kv)

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

    (
        q,
        k,
        v,
        k_cache,
        v_cache,
        batch_ids_q,
        q_tile_starts_q,
        cu_seqlens_q,
        cu_seqlens_kv,
    ) = make_contiguous(
        q,
        k,
        v,
        k_cache,
        v_cache,
        batch_ids_q,
        q_tile_starts_q,
        cu_seqlens_q,
        cu_seqlens_kv,
    )

    out = torch.empty_like(q)
    grid = (num_q_tiles, num_heads, 1)

    if supports_host_descriptor():
        y_dim_q = num_heads * total_q_len
        y_dim_kv_main = total_q_len if is_mla else num_heads * total_q_len
        y_dim_context = total_context_len if is_mla else num_heads * total_context_len
        desc_q = _make_host_desc(q, y_dim_q, d)
        desc_k = _make_host_desc(k, y_dim_kv_main, d)
        desc_v = _make_host_desc(v, y_dim_kv_main, d)
        desc_o = _make_host_desc(out, y_dim_q, d)
        desc_k_cache = _make_host_desc(k_cache, y_dim_context, d)
        desc_v_cache = _make_host_desc(v_cache, y_dim_context, d)
    else:
        desc_q = q
        desc_k = k
        desc_v = v
        desc_o = out
        desc_k_cache = k_cache
        desc_v_cache = v_cache

    if auto_tune_tiles:
        _mlm_compressed_kernel_auto_block_m[grid](
            desc_q,
            desc_k,
            desc_v,
            desc_o,
            desc_k_cache,
            desc_v_cache,
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
            causal_query_seq_attn=causal_query_seq_attn,
            BLOCK_N=block_n,
            HEAD_DIM=d,
        )
    else:
        _mlm_compressed_kernel[grid](
            desc_q,
            desc_k,
            desc_v,
            desc_o,
            desc_k_cache,
            desc_v_cache,
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
            causal_query_seq_attn=causal_query_seq_attn,
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
