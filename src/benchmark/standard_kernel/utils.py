import math

import torch
import torch.nn.functional as F
import triton

from flash_mlm.host import flash_attn_mlm, flash_attn_mlm_compressed
from flash_mlm.host.cache import InferenceCache
from flash_mlm.host.host_utils import (
    build_pack_metadata,
    pack_for_kernel,
    unpack_from_kernel,
)


def _resolve_padded_len(active_len: int, max_len: int | None, name: str) -> int:
    if max_len is None:
        return active_len
    if max_len < active_len:
        raise ValueError(f"--max-len ({max_len}) must be >= {name} ({active_len})")
    return max_len


def _make_varlen_lengths(
    batch_size: int,
    max_len: int,
    min_ratio: float,
    max_ratio: float,
    *,
    device: torch.device,
) -> torch.Tensor:
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if max_len < 1:
        raise ValueError("max_len must be >= 1")
    if batch_size == 1:
        return torch.tensor([max_len], device=device, dtype=torch.int32)

    ratios = torch.linspace(min_ratio, max_ratio, steps=batch_size, device=device)
    return torch.clamp((ratios * max_len).to(torch.int32), min=1, max=max_len)


def _build_cu_seqlens(lengths: torch.Tensor) -> torch.Tensor:
    cu = torch.zeros(lengths.numel() + 1, device=lengths.device, dtype=torch.int32)
    cu[1:] = lengths.cumsum(dim=0)
    return cu


def _estimate_tflops(
    lengths_q: torch.Tensor,
    lengths_kv: torch.Tensor,
    num_heads: int,
    head_dim: int,
    latency_ms: float,
) -> float:
    q = lengths_q.to(torch.float64)
    if lengths_kv.numel() == 1:
        kv_for_q = lengths_kv.expand_as(lengths_q)
    elif lengths_kv.numel() == lengths_q.numel():
        kv_for_q = lengths_kv
    else:
        raise ValueError("lengths_kv must have 1 or batch_size elements")
    kv_total = (kv_for_q + lengths_q).to(torch.float64)
    token_pairs = torch.sum(q * kv_total).item()
    total_flops = 4.0 * num_heads * head_dim * token_pairs
    return total_flops * 1e-12 / (latency_ms * 1e-3)


def _measure_peak_cuda_memory_mb(fn, *, iters: int = 3) -> tuple[float, float]:
    if iters < 1:
        raise ValueError("iters must be >= 1")

    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    baseline_alloc = torch.cuda.memory_allocated()
    baseline_reserved = torch.cuda.memory_reserved()

    for _ in range(iters):
        _ = fn()

    torch.cuda.synchronize()
    peak_alloc = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    mb = float(1024 * 1024)
    return (
        max(0.0, float(peak_alloc - baseline_alloc) / mb),
        float(peak_reserved) / mb,
    )


def _bench_noncompressed(
    *,
    payload: dict,
    is_mla: bool,
    num_heads: int,
    head_dim: int,
    context_len_active: int,
    context_batch_size: int,
    block_m: int,
    block_n: int,
    auto_tune_tiles: bool,
    check_correctness: bool,
):
    v_tensor = payload["k"] if is_mla else payload["v"]

    inference_cache = InferenceCache()
    if context_len_active > 0:
        if is_mla:
            k_cache_flat = payload["k_cache_dense"][:, 0, :context_len_active, :]
            k_cache_flat = k_cache_flat.reshape(
                context_batch_size * context_len_active, head_dim
            )
            v_cache_flat = None
        else:
            k_cache_flat = payload["k_cache_dense"][:, :, :context_len_active, :]
            v_cache_flat = payload["v_cache_dense"][:, :, :context_len_active, :]
            k_cache_flat = k_cache_flat.reshape(
                context_batch_size * num_heads * context_len_active, head_dim
            )
            v_cache_flat = v_cache_flat.reshape(
                context_batch_size * num_heads * context_len_active, head_dim
            )

        cu_seqlens_kv = torch.arange(
            context_batch_size + 1, device=payload["q"].device, dtype=torch.int32
        ) * int(context_len_active)
        inference_cache.prefill_kv_cache(
            layer_id=0,
            k_cache=k_cache_flat,
            v_cache=v_cache_flat,
            total_context_len=int(context_batch_size * context_len_active),
            cu_seqlens_kv=cu_seqlens_kv,
            context_batch_size=context_batch_size,
            is_mla=is_mla,
            num_heads=num_heads,
            head_dim=head_dim,
        )

    def _flash_fn():
        return flash_attn_mlm(
            payload["q"],
            payload["k"],
            v_tensor,
            scale=payload["scale"],
            inference_cache=inference_cache,
            layer_id=0,
            is_mla=is_mla,
            context_batch_size=context_batch_size,
            block_m=block_m,
            block_n=block_n,
            auto_tune_tiles=auto_tune_tiles,
        )

    def _torch_fn():
        return _reference_dense_attention_with_cache(
            payload["q"],
            payload["k"],
            v_tensor,
            payload["k_cache_dense"],
            payload["v_cache_dense"],
            payload["lengths_q_dense"],
            payload["lengths_kv_dense"],
            payload["scale"],
            is_mla,
            context_batch_size,
        )

    def _sdpa_fn():
        return _sdpa_dense_attention_with_cache(
            payload["q"],
            payload["k"],
            v_tensor,
            payload["k_cache_dense"],
            payload["v_cache_dense"],
            payload["lengths_q_dense"],
            payload["lengths_kv_dense"],
            payload["scale"],
            is_mla,
            context_batch_size,
        )

    if check_correctness:
        torch.testing.assert_close(_flash_fn(), _torch_fn(), atol=2e-2, rtol=0)

    flash_ms = triton.testing.do_bench(_flash_fn)
    baseline_ms = triton.testing.do_bench(_torch_fn)
    sdpa_ms = triton.testing.do_bench(_sdpa_fn)
    flash_tflops = _estimate_tflops(
        payload["lengths_q_dense"],
        payload["lengths_kv_dense"],
        num_heads,
        head_dim,
        flash_ms,
    )
    baseline_tflops = _estimate_tflops(
        payload["lengths_q_dense"],
        payload["lengths_kv_dense"],
        num_heads,
        head_dim,
        baseline_ms,
    )
    sdpa_tflops = _estimate_tflops(
        payload["lengths_q_dense"],
        payload["lengths_kv_dense"],
        num_heads,
        head_dim,
        sdpa_ms,
    )

    flash_peak_alloc_mb, flash_peak_reserved_mb = _measure_peak_cuda_memory_mb(
        _flash_fn
    )
    baseline_peak_alloc_mb, baseline_peak_reserved_mb = _measure_peak_cuda_memory_mb(
        _torch_fn
    )
    sdpa_peak_alloc_mb, sdpa_peak_reserved_mb = _measure_peak_cuda_memory_mb(_sdpa_fn)
    return (
        flash_ms,
        baseline_ms,
        sdpa_ms,
        flash_tflops,
        baseline_tflops,
        sdpa_tflops,
        flash_peak_alloc_mb,
        flash_peak_reserved_mb,
        baseline_peak_alloc_mb,
        baseline_peak_reserved_mb,
        sdpa_peak_alloc_mb,
        sdpa_peak_reserved_mb,
    )


def _bench_compressed(
    *,
    payload: dict,
    is_mla: bool,
    num_heads: int,
    head_dim: int,
    context_batch_size: int,
    block_m: int,
    block_n: int,
    auto_tune_tiles: bool,
    check_correctness: bool,
):
    if is_mla:
        k_cache = payload["k_cache_compact_mla"]
        v_cache = None
    else:
        k_cache = payload["k_cache_compact"]
        v_cache = payload["v_cache_compact"]

    inference_cache = InferenceCache()
    inference_cache.prefill_kv_cache(
        layer_id=0,
        k_cache=k_cache,
        v_cache=v_cache,
        total_context_len=payload["total_context_len"],
        cu_seqlens_kv=payload["cu_seqlens_kv"],
        context_batch_size=context_batch_size,
        is_mla=is_mla,
        num_heads=num_heads,
        head_dim=head_dim,
    )

    def _flash_fn():
        return flash_attn_mlm_compressed(
            payload["q"],
            payload["k"],
            payload["v"],
            num_heads=num_heads,
            q_meta=payload["q_meta"],
            scale=payload["scale"],
            inference_cache=inference_cache,
            layer_id=0,
            is_mla=is_mla,
            context_batch_size=context_batch_size,
            block_m=block_m,
            block_n=block_n,
            auto_tune_tiles=auto_tune_tiles,
        )

    def _torch_fn():
        return _reference_varlen_attention_with_cache(
            payload["q"],
            payload["k"],
            payload["v"],
            payload["lengths_q_varlen"],
            k_cache,
            v_cache,
            payload["cu_seqlens_kv"],
            payload["scale"],
            is_mla,
            context_batch_size,
        )

    def _sdpa_fn():
        return _sdpa_varlen_attention_with_cache(
            payload["q"],
            payload["k"],
            payload["v"],
            payload["lengths_q_varlen"],
            k_cache,
            v_cache,
            payload["cu_seqlens_kv"],
            payload["scale"],
            is_mla,
            context_batch_size,
        )

    if check_correctness:
        packed_out = _flash_fn()
        out = unpack_from_kernel(packed_out, payload["q_meta"], H=num_heads)
        ref = _torch_fn()
        torch.testing.assert_close(out, ref, atol=2e-2, rtol=0)

    flash_ms = triton.testing.do_bench(_flash_fn)
    baseline_ms = triton.testing.do_bench(_torch_fn)
    sdpa_ms = triton.testing.do_bench(_sdpa_fn)
    flash_tflops = _estimate_tflops(
        payload["lengths_q_varlen"],
        payload["lengths_kv_varlen"],
        num_heads,
        head_dim,
        flash_ms,
    )
    baseline_tflops = _estimate_tflops(
        payload["lengths_q_varlen"],
        payload["lengths_kv_varlen"],
        num_heads,
        head_dim,
        baseline_ms,
    )
    sdpa_tflops = _estimate_tflops(
        payload["lengths_q_varlen"],
        payload["lengths_kv_varlen"],
        num_heads,
        head_dim,
        sdpa_ms,
    )

    flash_peak_alloc_mb, flash_peak_reserved_mb = _measure_peak_cuda_memory_mb(
        _flash_fn
    )
    baseline_peak_alloc_mb, baseline_peak_reserved_mb = _measure_peak_cuda_memory_mb(
        _torch_fn
    )
    sdpa_peak_alloc_mb, sdpa_peak_reserved_mb = _measure_peak_cuda_memory_mb(_sdpa_fn)
    return (
        flash_ms,
        baseline_ms,
        sdpa_ms,
        flash_tflops,
        baseline_tflops,
        sdpa_tflops,
        flash_peak_alloc_mb,
        flash_peak_reserved_mb,
        baseline_peak_alloc_mb,
        baseline_peak_reserved_mb,
        sdpa_peak_alloc_mb,
        sdpa_peak_reserved_mb,
    )


def _reference_varlen_attention_with_cache(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lengths_q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    scale: float,
    is_mla: bool,
    context_batch_size: int,
) -> torch.Tensor:
    bsz, n_heads, _, _ = q.shape
    head_dim = q.shape[-1]
    total_context_len = int(cu_seqlens_kv[-1].item())
    out = torch.zeros_like(q)
    for b in range(bsz):
        lq = int(lengths_q[b].item())
        if lq == 0:
            continue
        ctx_batch = 0 if context_batch_size == 1 else b
        c_start = int(cu_seqlens_kv[ctx_batch].item())
        c_end = int(cu_seqlens_kv[ctx_batch + 1].item())
        for h in range(n_heads):
            q_bh = q[b, h, :lq, :]
            if is_mla:
                k_main = k[b, h, :lq, :]
                v_main = k_main
                k_ctx = k_cache[c_start:c_end, :]
                v_ctx = k_ctx
            else:
                k_main = k[b, h, :lq, :]
                v_main = v[b, h, :lq, :]
                row_offset = h * total_context_len
                k_ctx = k_cache[row_offset + c_start : row_offset + c_end, :]
                v_ctx = v_cache[row_offset + c_start : row_offset + c_end, :]

            k_all = torch.cat([k_ctx, k_main], dim=0)
            v_all = torch.cat([v_ctx, v_main], dim=0)
            logits = torch.matmul(q_bh, k_all.transpose(0, 1)) * scale
            probs = torch.softmax(logits.float(), dim=-1).to(q.dtype)
            out[b, h, :lq, :] = torch.matmul(probs, v_all)
    return out


def _sdpa_varlen_attention_with_cache(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lengths_q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    scale: float,
    is_mla: bool,
    context_batch_size: int,
) -> torch.Tensor:
    bsz, n_heads, _, _ = q.shape
    head_dim = q.shape[-1]
    total_context_len = int(cu_seqlens_kv[-1].item())
    out = torch.zeros_like(q)
    for b in range(bsz):
        lq = int(lengths_q[b].item())
        if lq == 0:
            continue
        ctx_batch = 0 if context_batch_size == 1 else b
        c_start = int(cu_seqlens_kv[ctx_batch].item())
        c_end = int(cu_seqlens_kv[ctx_batch + 1].item())

        q_bh = q[b, :, :lq, :]
        if is_mla:
            k_main = k[b, :, :lq, :]
            v_main = k_main
            k_ctx = k_cache[c_start:c_end, :].unsqueeze(0).expand(n_heads, -1, -1)
            v_ctx = k_ctx
        else:
            k_main = k[b, :, :lq, :]
            v_main = v[b, :, :lq, :]
            k_ctx = k_cache.view(n_heads, total_context_len, head_dim)[
                :, c_start:c_end, :
            ]
            v_ctx = v_cache.view(n_heads, total_context_len, head_dim)[
                :, c_start:c_end, :
            ]

        k_all = torch.cat([k_ctx, k_main], dim=1)
        v_all = torch.cat([v_ctx, v_main], dim=1)
        out[b, :, :lq, :] = F.scaled_dot_product_attention(
            q_bh,
            k_all,
            v_all,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
            scale=scale,
        )
    return out


def _sdpa_dense_attention_with_cache(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    lengths_q: torch.Tensor,
    lengths_kv: torch.Tensor,
    scale: float,
    is_mla: bool,
    context_batch_size: int,
) -> torch.Tensor:
    bsz, n_heads, _, _ = q.shape
    if lengths_q.numel() != bsz:
        raise ValueError("lengths_q must have batch_size elements")
    if lengths_kv.numel() not in (1, bsz):
        raise ValueError("lengths_kv must have 1 or batch_size elements")

    out = torch.zeros_like(q)
    for b in range(bsz):
        lq = int(lengths_q[b].item())
        if lq == 0:
            continue
        ctx_batch = 0 if context_batch_size == 1 else b
        lkv = (
            int(lengths_kv[0].item())
            if lengths_kv.numel() == 1
            else int(lengths_kv[ctx_batch].item())
        )

        q_bh = q[b, :, :lq, :]
        if is_mla:
            k_main = k[b, :, :lq, :]
            v_main = k_main
            k_ctx = k_cache[ctx_batch, 0, :lkv, :].unsqueeze(0).expand(n_heads, -1, -1)
            v_ctx = k_ctx
        else:
            k_main = k[b, :, :lq, :]
            v_main = v[b, :, :lq, :]
            k_ctx = k_cache[ctx_batch, :, :lkv, :]
            v_ctx = v_cache[ctx_batch, :, :lkv, :]

        k_all = torch.cat([k_ctx, k_main], dim=1)
        v_all = torch.cat([v_ctx, v_main], dim=1)
        out[b, :, :lq, :] = F.scaled_dot_product_attention(
            q_bh,
            k_all,
            v_all,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
            scale=scale,
        )
    return out


def _reference_dense_attention_with_cache(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    lengths_q: torch.Tensor,
    lengths_kv: torch.Tensor,
    scale: float,
    is_mla: bool,
    context_batch_size: int,
) -> torch.Tensor:
    bsz, n_heads, _, _ = q.shape
    if lengths_q.numel() != bsz:
        raise ValueError("lengths_q must have batch_size elements")
    if lengths_kv.numel() not in (1, bsz):
        raise ValueError("lengths_kv must have 1 or batch_size elements")

    out = torch.zeros_like(q)
    for b in range(bsz):
        lq = int(lengths_q[b].item())
        if lq == 0:
            continue
        ctx_batch = 0 if context_batch_size == 1 else b
        lkv = (
            int(lengths_kv[0].item())
            if lengths_kv.numel() == 1
            else int(lengths_kv[ctx_batch].item())
        )
        for h in range(n_heads):
            q_bh = q[b, h, :lq, :]
            if is_mla:
                k_main = k[b, h, :lq, :]
                v_main = k_main
                k_ctx = k_cache[ctx_batch, 0, :lkv, :]
                v_ctx = k_ctx
            else:
                k_main = k[b, h, :lq, :]
                v_main = v[b, h, :lq, :]
                k_ctx = k_cache[ctx_batch, h, :lkv, :]
                v_ctx = v_cache[ctx_batch, h, :lkv, :]

            k_all = torch.cat([k_ctx, k_main], dim=0)
            v_all = torch.cat([v_ctx, v_main], dim=0)
            logits = torch.matmul(q_bh, k_all.transpose(0, 1)) * scale
            probs = torch.softmax(logits.float(), dim=-1).to(q.dtype)
            out[b, h, :lq, :] = torch.matmul(probs, v_all)
    return out


def _build_shared_inputs(
    *,
    batch_size: int,
    num_heads: int,
    query_seq_len_active: int,
    query_seq_len_padded: int,
    context_len_active: int,
    context_subseq_len_active: int,
    context_subseq_len_padded: int,
    context_batch_size: int,
    num_context_seqs_per_query: int,
    head_dim: int,
    has_cache: bool,
    block_n: int,
    q_ratio_min: float,
    q_ratio_max: float,
    kv_ratio_min: float,
    kv_ratio_max: float,
    device: torch.device,
    dtype: torch.dtype,
):
    q = torch.randn(
        (batch_size, num_heads, query_seq_len_padded, head_dim),
        device=device,
        dtype=dtype,
    )
    k = torch.randn(
        (batch_size, num_heads, query_seq_len_padded, head_dim),
        device=device,
        dtype=dtype,
    )
    v = torch.randn(
        (batch_size, num_heads, query_seq_len_padded, head_dim),
        device=device,
        dtype=dtype,
    )

    lengths_q_dense = torch.full(
        (batch_size,), query_seq_len_active, device=device, dtype=torch.int32
    )
    lengths_q_varlen = _make_varlen_lengths(
        batch_size,
        query_seq_len_active,
        q_ratio_min,
        q_ratio_max,
        device=device,
    )

    if has_cache:
        context_len_padded = context_subseq_len_padded * num_context_seqs_per_query
        lengths_kv_subseq = _make_varlen_lengths(
            context_batch_size * num_context_seqs_per_query,
            context_subseq_len_active,
            kv_ratio_min,
            kv_ratio_max,
            device=device,
        ).view(context_batch_size, num_context_seqs_per_query)
        lengths_kv_varlen = lengths_kv_subseq.sum(dim=1).to(torch.int32)

        context_k_subseq = torch.randn(
            (
                context_batch_size,
                num_heads,
                num_context_seqs_per_query,
                context_subseq_len_padded,
                head_dim,
            ),
            device=device,
            dtype=dtype,
        )
        context_k = context_k_subseq.reshape(
            context_batch_size, num_heads, context_len_padded, head_dim
        )
        context_v_subseq = torch.randn_like(context_k_subseq)
        context_v = context_v_subseq.reshape(
            context_batch_size, num_heads, context_len_padded, head_dim
        )

        non_mla_k_chunks: list[torch.Tensor] = []
        non_mla_v_chunks: list[torch.Tensor] = []
        mla_k_chunks: list[torch.Tensor] = []
        mla_v_chunks: list[torch.Tensor] = []

        for c in range(context_batch_size):
            subseq_meta = build_pack_metadata(
                lengths_kv_subseq[c], context_subseq_len_padded, block_n=block_n
            )

            k_item = context_k_subseq[c].permute(1, 0, 2, 3).contiguous()
            v_item = context_v_subseq[c].permute(1, 0, 2, 3).contiguous()

            packed_k_item = pack_for_kernel(k_item, subseq_meta)
            packed_v_item = pack_for_kernel(v_item, subseq_meta)

            packed_k_item_mla = pack_for_kernel(k_item[:, :1, :, :], subseq_meta)
            packed_v_item_mla = pack_for_kernel(v_item[:, :1, :, :], subseq_meta)

            for h in range(num_heads):
                non_mla_k_chunks.append(packed_k_item[h])
                non_mla_v_chunks.append(packed_v_item[h])
            mla_k_chunks.append(packed_k_item_mla[0])
            mla_v_chunks.append(packed_v_item_mla[0])

        total_context_len = int(lengths_kv_varlen.sum().item())
        if total_context_len > 0:
            k_cache_compact = torch.cat(non_mla_k_chunks, dim=0)
            v_cache_compact = torch.cat(non_mla_v_chunks, dim=0)
            k_cache_compact_mla = torch.cat(mla_k_chunks, dim=0)
            v_cache_compact_mla = torch.cat(mla_v_chunks, dim=0)
        else:
            k_cache_compact_mla = torch.empty((0, head_dim), device=device, dtype=dtype)
            v_cache_compact_mla = torch.empty_like(k_cache_compact_mla)
            k_cache_compact = torch.empty((0, head_dim), device=device, dtype=dtype)
            v_cache_compact = torch.empty_like(k_cache_compact)
    else:
        lengths_kv_varlen = torch.zeros(
            context_batch_size, device=device, dtype=torch.int32
        )
        context_k = torch.empty(
            (context_batch_size, num_heads, 0, head_dim), device=device, dtype=dtype
        )
        context_v = torch.empty_like(context_k)
        k_cache_compact_mla = torch.empty((0, head_dim), device=device, dtype=dtype)
        v_cache_compact_mla = torch.empty_like(k_cache_compact_mla)
        k_cache_compact = torch.empty((0, head_dim), device=device, dtype=dtype)
        v_cache_compact = torch.empty_like(k_cache_compact)

    if has_cache:
        if context_batch_size == 1:
            k_cache_dense = context_k
            v_cache_dense = context_v
            lengths_kv_dense = torch.tensor(
                [context_len_active], device=device, dtype=torch.int32
            )
        else:
            k_cache_dense = context_k
            v_cache_dense = context_v
            lengths_kv_dense = torch.full(
                (batch_size,), context_len_active, device=device, dtype=torch.int32
            )
    else:
        k_cache_dense = torch.empty(
            (context_batch_size, num_heads, 0, head_dim), device=device, dtype=dtype
        )
        v_cache_dense = torch.empty_like(k_cache_dense)
        lengths_kv_dense = torch.zeros(
            1 if context_batch_size == 1 else batch_size,
            device=device,
            dtype=torch.int32,
        )

    cu_seqlens_kv = _build_cu_seqlens(lengths_kv_varlen)
    total_context_len = int(cu_seqlens_kv[-1].item())

    q_meta = build_pack_metadata(
        lengths_q_varlen, query_seq_len_padded, block_n=block_n
    )

    return {
        "q": q,
        "k": k,
        "v": v,
        "scale": 1.0 / math.sqrt(head_dim),
        "lengths_q_dense": lengths_q_dense,
        "lengths_q_varlen": lengths_q_varlen,
        "lengths_kv_dense": lengths_kv_dense,
        "lengths_kv_varlen": lengths_kv_varlen,
        "k_cache_dense": k_cache_dense,
        "v_cache_dense": v_cache_dense,
        "k_cache_compact": k_cache_compact,
        "v_cache_compact": v_cache_compact,
        "k_cache_compact_mla": k_cache_compact_mla,
        "v_cache_compact_mla": v_cache_compact_mla,
        "cu_seqlens_kv": cu_seqlens_kv,
        "total_context_len": total_context_len,
        "q_meta": q_meta,
    }
