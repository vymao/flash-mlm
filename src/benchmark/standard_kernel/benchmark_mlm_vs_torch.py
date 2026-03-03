import argparse
import sys
from pathlib import Path

import torch
import triton

_THIS_DIR = Path(__file__).resolve().parent
_PARENT = _THIS_DIR.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

from flash_mlm.host import flash_attn_mlm, flash_attn_mlm_compressed  # noqa: E402
from flash_mlm.cache import InferenceCache  # noqa: E402
from flash_mlm.host_utils import unpack_from_kernel  # noqa: E402
from benchmark.standard_kernel.utils import (  # noqa: E402
    _build_shared_inputs,
    _estimate_tflops,
    _reference_dense_attention_with_cache,
    _reference_varlen_attention_with_cache,
    _resolve_padded_len,
)
from benchmark.standard_kernel.plot_utils import (
    write_benchmark_bar_plots,
)  # noqa: E402


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
            v_cache_flat = k_cache_flat
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
    rows = []
    sections = [
        ("Non-MLA + Non-Cache", False, False, 1),
        ("Non-MLA + Cache (Single)", False, True, 1),
        ("Non-MLA + Cache (Full)", False, True, batch_size),
        ("MLA + Non-Cache", True, False, 1),
        ("MLA + Cache (Single)", True, True, 1),
        ("MLA + Cache (Full)", True, True, batch_size),
    ]

    for section, is_mla, has_cache, section_context_batch_size in sections:
        section_head_dim = latent_head_dim if is_mla else head_dim
        section_context_len = context_len if has_cache else 0
        payload = _build_shared_inputs(
            batch_size=batch_size,
            num_heads=num_heads,
            query_seq_len_active=query_seq_len_active,
            query_seq_len_padded=query_seq_len_padded,
            context_len_active=context_len,
            context_subseq_len_active=context_subseq_len,
            context_subseq_len_padded=context_subseq_len_padded,
            context_batch_size=section_context_batch_size,
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
            context_batch_size=section_context_batch_size,
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
                context_batch_size=section_context_batch_size,
                block_m=block_m,
                block_n=block_n,
                auto_tune_tiles=auto_tune_tiles,
                check_correctness=check_correctness,
            )
        )

        mean_context_subseq_len = (
            float(payload["lengths_kv_varlen"].to(torch.float32).mean().item())
            / float(num_context_seqs_per_query)
            if num_context_seqs_per_query > 0
            else 0.0
        )

        rows.append(
            {
                "context_len": context_len,
                "mean_context_subseq_len": mean_context_subseq_len,
                "query_batch_size": batch_size,
                "has_cache": has_cache,
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
            "Benchmark six sections (non-MLA/MLA x non-cache/single-cache/full-cache) "
            "and report "
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
    cache_rows = [row for row in rows if row["has_cache"]]
    plot_mean_context_subseq_len = (
        sum(row["mean_context_subseq_len"] for row in cache_rows) / len(cache_rows)
        if cache_rows
        else 0.0
    )
    _print_results(rows, context_len=args.context_len)
    write_benchmark_bar_plots(
        rows,
        mean_context_subseq_len=plot_mean_context_subseq_len,
        query_batch_size=args.batch_size,
        plots_dir=plots_dir,
    )

    print(f"\nSaved plots to: {plots_dir.resolve()}")


if __name__ == "__main__":
    main()
