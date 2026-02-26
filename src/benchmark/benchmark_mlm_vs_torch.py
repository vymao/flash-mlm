import argparse
import math
import sys
from pathlib import Path

import torch
import triton

_THIS_DIR = Path(__file__).resolve().parent
_PARENT = _THIS_DIR.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

from flash_mlm.host import flash_attn_mlm, flash_attn_mlm_compressed  # noqa: E402
from flash_mlm.host_utils import build_pack_metadata, unpack_from_kernel  # noqa: E402
from utils.plot_utils import write_benchmark_bar_plots  # noqa: E402


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


def _sample_concatenated_context_lengths(
    *,
    context_batch_size: int,
    context_subseq_len_active: int,
    num_context_seqs_per_query: int,
    kv_ratio_min: float,
    kv_ratio_max: float,
    device: torch.device,
) -> torch.Tensor:
    if context_subseq_len_active < 1:
        return torch.zeros(context_batch_size, device=device, dtype=torch.int32)
    if num_context_seqs_per_query == 1:
        return _make_varlen_lengths(
            context_batch_size,
            context_subseq_len_active,
            kv_ratio_min,
            kv_ratio_max,
            device=device,
        )

    subseq = _make_varlen_lengths(
        context_batch_size * num_context_seqs_per_query,
        context_subseq_len_active,
        kv_ratio_min,
        kv_ratio_max,
        device=device,
    ).view(context_batch_size, num_context_seqs_per_query)
    totals = subseq.sum(dim=1)
    max_total_context_len = context_subseq_len_active * num_context_seqs_per_query
    return torch.clamp(totals, min=1, max=max_total_context_len).to(torch.int32)


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


def _reference_dense_attention_with_cache(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    context_len: int,
    scale: float,
    is_mla: bool,
    context_batch_size: int,
) -> torch.Tensor:
    bsz, n_heads, n_ctx, _ = q.shape
    out = torch.zeros_like(q)
    for b in range(bsz):
        ctx_batch = 0 if context_batch_size == 1 else b
        for h in range(n_heads):
            q_bh = q[b, h, :n_ctx, :]
            if is_mla:
                k_main = k[b, h, :n_ctx, :]
                v_main = k_main
                k_ctx = k_cache[ctx_batch, 0, :context_len, :]
                v_ctx = k_ctx
            else:
                k_main = k[b, h, :n_ctx, :]
                v_main = v[b, h, :n_ctx, :]
                k_ctx = k_cache[ctx_batch, h, :context_len, :]
                v_ctx = v_cache[ctx_batch, h, :context_len, :]

            k_all = torch.cat([k_ctx, k_main], dim=0)
            v_all = torch.cat([v_ctx, v_main], dim=0)
            logits = torch.matmul(q_bh, k_all.transpose(0, 1)) * scale
            probs = torch.softmax(logits.float(), dim=-1).to(q.dtype)
            out[b, h, :n_ctx, :] = torch.matmul(probs, v_all)
    return out


def _build_shared_inputs(
    *,
    batch_size: int,
    num_heads: int,
    query_seq_len_active: int,
    query_seq_len_padded: int,
    context_len_active: int,
    context_subseq_len_active: int,
    context_len_padded: int,
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
        lengths_kv_varlen = _sample_concatenated_context_lengths(
            context_batch_size=context_batch_size,
            context_subseq_len_active=context_subseq_len_active,
            num_context_seqs_per_query=num_context_seqs_per_query,
            kv_ratio_min=kv_ratio_min,
            kv_ratio_max=kv_ratio_max,
            device=device,
        )
        context_k = torch.randn(
            (context_batch_size, num_heads, context_len_padded, head_dim),
            device=device,
            dtype=dtype,
        )
        context_v = torch.randn_like(context_k)
    else:
        lengths_kv_varlen = torch.zeros(
            context_batch_size, device=device, dtype=torch.int32
        )
        context_k = torch.empty(
            (context_batch_size, num_heads, 0, head_dim), device=device, dtype=dtype
        )
        context_v = torch.empty_like(context_k)

    if has_cache:
        if context_batch_size == 1:
            k_cache_dense = context_k.expand(
                batch_size, num_heads, context_len_padded, head_dim
            ).contiguous()
            v_cache_dense = context_v.expand(
                batch_size, num_heads, context_len_padded, head_dim
            ).contiguous()
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
            (batch_size, num_heads, 0, head_dim), device=device, dtype=dtype
        )
        v_cache_dense = torch.empty_like(k_cache_dense)
        lengths_kv_dense = torch.zeros(
            1 if context_batch_size == 1 else batch_size,
            device=device,
            dtype=torch.int32,
        )

    cu_seqlens_kv = _build_cu_seqlens(lengths_kv_varlen)
    total_context_len = int(cu_seqlens_kv[-1].item())

    if total_context_len > 0:
        mla_k_chunks = []
        mla_v_chunks = []
        for c in range(context_batch_size):
            lc = int(lengths_kv_varlen[c].item())
            if lc > 0:
                mla_k_chunks.append(context_k[c, 0, :lc, :])
                mla_v_chunks.append(context_v[c, 0, :lc, :])
        k_cache_compact_mla = torch.cat(mla_k_chunks, dim=0)
        v_cache_compact_mla = torch.cat(mla_v_chunks, dim=0)

        non_mla_k_chunks = []
        non_mla_v_chunks = []
        for h in range(num_heads):
            for c in range(context_batch_size):
                lc = int(lengths_kv_varlen[c].item())
                if lc > 0:
                    non_mla_k_chunks.append(context_k[c, h, :lc, :])
                    non_mla_v_chunks.append(context_v[c, h, :lc, :])
        k_cache_compact = torch.cat(non_mla_k_chunks, dim=0)
        v_cache_compact = torch.cat(non_mla_v_chunks, dim=0)
    else:
        k_cache_compact_mla = torch.empty((0, head_dim), device=device, dtype=dtype)
        v_cache_compact_mla = torch.empty_like(k_cache_compact_mla)
        k_cache_compact = torch.empty((0, head_dim), device=device, dtype=dtype)
        v_cache_compact = torch.empty_like(k_cache_compact)

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

    def _flash_fn():
        return flash_attn_mlm(
            payload["q"],
            payload["k"],
            v_tensor,
            payload["k_cache_dense"],
            payload["v_cache_dense"],
            context_len=context_len_active,
            scale=payload["scale"],
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
            context_len_active,
            payload["scale"],
            is_mla,
            context_batch_size,
        )

    if check_correctness:
        torch.testing.assert_close(_flash_fn(), _torch_fn(), atol=2e-2, rtol=0)

    flash_ms = triton.testing.do_bench(_flash_fn)
    baseline_ms = triton.testing.do_bench(_torch_fn)
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
    return flash_ms, baseline_ms, flash_tflops, baseline_tflops


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
        v_cache = payload["v_cache_compact_mla"]
    else:
        k_cache = payload["k_cache_compact"]
        v_cache = payload["v_cache_compact"]

    def _flash_fn():
        return flash_attn_mlm_compressed(
            payload["q"],
            payload["k"],
            payload["v"],
            k_cache,
            v_cache,
            num_heads=num_heads,
            q_meta=payload["q_meta"],
            total_context_len=payload["total_context_len"],
            cu_seqlens_kv=payload["cu_seqlens_kv"],
            scale=payload["scale"],
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

    if check_correctness:
        packed_out = _flash_fn()
        out = unpack_from_kernel(packed_out, payload["q_meta"], H=num_heads)
        ref = _torch_fn()
        torch.testing.assert_close(out, ref, atol=2e-2, rtol=0)

    flash_ms = triton.testing.do_bench(_flash_fn)
    baseline_ms = triton.testing.do_bench(_torch_fn)
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
    return flash_ms, baseline_ms, flash_tflops, baseline_tflops


def _bench_scenarios_for_context(
    *,
    context_subseq_len: int,
    batch_size: int,
    num_heads: int,
    query_seq_len_active: int,
    query_seq_len_padded: int,
    context_subseq_len_padded: int,
    context_batch_size: int,
    head_dim: int,
    latent_head_dim: int,
    num_context_seqs_per_query: int,
    block_m: int,
    block_n: int,
    auto_tune_tiles: bool,
    q_ratio_min: float,
    q_ratio_max: float,
    kv_ratio_min: float,
    kv_ratio_max: float,
    check_correctness: bool,
    device: torch.device,
) -> list[dict]:
    context_len = context_subseq_len * num_context_seqs_per_query
    context_len_padded = context_subseq_len_padded * num_context_seqs_per_query
    rows = []
    sections = [
        ("Non-MLA + Non-Cache", False, False),
        ("Non-MLA + Cache", False, True),
        ("MLA + Non-Cache", True, False),
        ("MLA + Cache", True, True),
    ]

    for section, is_mla, has_cache in sections:
        section_head_dim = latent_head_dim if is_mla else head_dim
        section_context_len = context_len if has_cache else 0
        payload = _build_shared_inputs(
            batch_size=batch_size,
            num_heads=num_heads,
            query_seq_len_active=query_seq_len_active,
            query_seq_len_padded=query_seq_len_padded,
            context_len_active=context_len,
            context_subseq_len_active=context_subseq_len,
            context_len_padded=context_len_padded,
            context_batch_size=context_batch_size,
            num_context_seqs_per_query=num_context_seqs_per_query,
            head_dim=section_head_dim,
            has_cache=has_cache,
            block_n=block_n,
            q_ratio_min=q_ratio_min,
            q_ratio_max=q_ratio_max,
            kv_ratio_min=kv_ratio_min,
            kv_ratio_max=kv_ratio_max,
            device=device,
            dtype=torch.float16,
        )

        noncompressed_ms, _, noncompressed_tflops, _ = _bench_noncompressed(
            payload=payload,
            is_mla=is_mla,
            num_heads=num_heads,
            context_len_active=section_context_len,
            context_batch_size=context_batch_size,
            head_dim=section_head_dim,
            block_m=block_m,
            block_n=block_n,
            auto_tune_tiles=auto_tune_tiles,
            check_correctness=check_correctness,
        )
        compressed_ms, standard_ms, compressed_tflops, standard_tflops = (
            _bench_compressed(
                payload=payload,
                is_mla=is_mla,
                num_heads=num_heads,
                head_dim=section_head_dim,
                context_batch_size=context_batch_size,
                block_m=block_m,
                block_n=block_n,
                auto_tune_tiles=auto_tune_tiles,
                check_correctness=check_correctness,
            )
        )

        rows.append(
            {
                "context_len": context_len,
                "section": section,
                "standard_ms": standard_ms,
                "noncompressed_ms": noncompressed_ms,
                "compressed_ms": compressed_ms,
                "standard_tflops": standard_tflops,
                "noncompressed_tflops": noncompressed_tflops,
                "compressed_tflops": compressed_tflops,
            }
        )
    return rows


def _print_results(rows: list[dict], *, context_len: int):
    print("\n=== attention benchmark by section ===")
    print(f"context_len={context_len}")
    print(
        "{:<22} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12}".format(
            "SECTION",
            "STD_MS",
            "NONC_MS",
            "COMP_MS",
            "STD_TF",
            "NONC_TF",
            "COMP_TF",
        )
    )
    for row in rows:
        print(
            "{:<22} {:>12.4f} {:>12.4f} {:>12.4f} {:>12.2f} {:>12.2f} {:>12.2f}".format(
                row["section"],
                row["standard_ms"],
                row["noncompressed_ms"],
                row["compressed_ms"],
                row["standard_tflops"],
                row["noncompressed_tflops"],
                row["compressed_tflops"],
            )
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark four sections (non-MLA/MLA x non-cache/cache) and report "
            "three bars per section: standard attention, non-compressed kernel, "
            "compressed kernel."
        )
    )
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--num-heads", type=int, default=20)
    parser.add_argument(
        "--query-len",
        type=int,
        default=750,
        help="Active (true) max query length before optional --max-len padding.",
    )
    parser.add_argument(
        "--context-len",
        type=int,
        default=750,
        help=(
            "Active (true) max context length per context subsequence before "
            "optional --max-len padding."
        ),
    )
    parser.add_argument(
        "--context-batch-size",
        type=int,
        default=1,
        help="Number of context sequences: 1 (shared) or batch-size (per sample).",
    )
    parser.add_argument(
        "--num-context-seqs-per-query",
        type=int,
        default=7,
        help="How many context sequences to concatenate for each query batch item.",
    )
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument(
        "--latent-head-dim",
        type=int,
        default=128,
        help="Head dimension used for MLA sections (can differ from --head-dim).",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=1024,
        help=(
            "Optional padded max sequence length per subsequence used for query "
            "tensors and context-cache tensors. Must be >= query-len and context-len."
        ),
    )
    parser.add_argument("--block-m", type=int, default=64)
    parser.add_argument("--block-n", type=int, default=32)
    parser.add_argument("--auto-tune-tiles", action="store_true")
    parser.add_argument(
        "--q-ratio-min",
        type=float,
        default=0.8,
        help="Compressed-only: min ratio for sampling true query lengths.",
    )
    parser.add_argument(
        "--q-ratio-max",
        type=float,
        default=1.0,
        help="Compressed-only: max ratio for sampling true query lengths.",
    )
    parser.add_argument(
        "--kv-ratio-min",
        type=float,
        default=0.8,
        help="Compressed-only: min ratio for sampling true context lengths.",
    )
    parser.add_argument(
        "--kv-ratio-max",
        type=float,
        default=1.0,
        help="Compressed-only: max ratio for sampling true context lengths.",
    )
    parser.add_argument("--plots-dir", type=str, default="plots")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--check-correctness", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    if args.context_batch_size not in (1, args.batch_size):
        raise ValueError("context-batch-size must be 1 or equal to batch-size")
    if args.num_context_seqs_per_query < 1:
        raise ValueError("num-context-seqs-per-query must be >= 1")
    if args.context_len < 0:
        raise ValueError("context-len must be >= 0")
    if args.head_dim < 1 or args.latent_head_dim < 1:
        raise ValueError("head dims must be >= 1")

    device = triton.runtime.driver.active.get_active_torch_device()
    plots_dir = Path(args.plots_dir)
    query_seq_len_padded = _resolve_padded_len(
        args.query_len, args.max_len, "--query-len"
    )
    context_len_padded = _resolve_padded_len(
        args.context_len, args.max_len, "--context-len"
    )

    torch.manual_seed(args.seed)

    rows = _bench_scenarios_for_context(
        context_subseq_len=args.context_len,
        batch_size=args.batch_size,
        num_heads=args.num_heads,
        query_seq_len_active=args.query_len,
        query_seq_len_padded=query_seq_len_padded,
        context_subseq_len_padded=context_len_padded,
        context_batch_size=args.context_batch_size,
        head_dim=args.head_dim,
        latent_head_dim=args.latent_head_dim,
        num_context_seqs_per_query=args.num_context_seqs_per_query,
        block_m=args.block_m,
        block_n=args.block_n,
        auto_tune_tiles=args.auto_tune_tiles,
        q_ratio_min=args.q_ratio_min,
        q_ratio_max=args.q_ratio_max,
        kv_ratio_min=args.kv_ratio_min,
        kv_ratio_max=args.kv_ratio_max,
        check_correctness=args.check_correctness,
        device=device,
    )
    _print_results(rows, context_len=args.context_len)
    write_benchmark_bar_plots(rows, context_len=args.context_len, plots_dir=plots_dir)

    print(f"\nSaved plots to: {plots_dir.resolve()}")


if __name__ == "__main__":
    main()
