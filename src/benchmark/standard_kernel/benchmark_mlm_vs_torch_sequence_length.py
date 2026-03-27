import argparse
import sys
from pathlib import Path

import torch
import triton

_THIS_DIR = Path(__file__).resolve().parent
_PARENT = _THIS_DIR.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

from benchmark.standard_kernel.plot_utils import (  # noqa: E402
    write_sequence_length_line_plots,
)
from benchmark.standard_kernel.utils import (  # noqa: E402
    _bench_compressed,
    _bench_noncompressed,
    _build_shared_inputs,
)

SECTION_SPECS = [
    ("Non-MLA + Non-Cache", False, False, 1),
    ("Non-MLA + Cache (Single)", False, True, 1),
    ("MLA + Non-Cache", True, False, 1),
    ("MLA + Cache (Single)", True, True, 1),
]


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def _sequence_lengths(min_seq_len: int, max_seq_len: int) -> list[int]:
    if min_seq_len < 1 or max_seq_len < 1:
        raise ValueError("sequence lengths must be >= 1")
    if min_seq_len > max_seq_len:
        raise ValueError("min-seq-len must be <= max-seq-len")
    if not _is_power_of_two(min_seq_len) or not _is_power_of_two(max_seq_len):
        raise ValueError("min-seq-len and max-seq-len must be powers of 2")

    lengths = []
    current = min_seq_len
    while current <= max_seq_len:
        lengths.append(current)
        current *= 2
    return lengths


def _bench_rows_for_sequence_length(
    *,
    sequence_length: int,
    batch_size: int,
    num_heads: int,
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
    rows = []
    for section, is_mla, has_cache, context_batch_size in SECTION_SPECS:
        section_head_dim = latent_head_dim if is_mla else head_dim
        section_context_len = (
            sequence_length * num_context_seqs_per_query if has_cache else 0
        )
        payload = _build_shared_inputs(
            batch_size=batch_size,
            num_heads=num_heads,
            query_seq_len_active=sequence_length,
            query_seq_len_padded=sequence_length,
            context_len_active=section_context_len,
            context_subseq_len_active=sequence_length,
            context_subseq_len_padded=sequence_length,
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

        (
            noncompressed_ms,
            _,
            _,
            noncompressed_tflops,
            _,
            _,
            noncompressed_peak_alloc_mb,
            noncompressed_peak_reserved_mb,
            _,
            _,
            _,
            _,
        ) = _bench_noncompressed(
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
        (
            compressed_ms,
            standard_ms,
            sdpa_ms,
            compressed_tflops,
            standard_tflops,
            sdpa_tflops,
            compressed_peak_alloc_mb,
            compressed_peak_reserved_mb,
            standard_peak_alloc_mb,
            standard_peak_reserved_mb,
            sdpa_peak_alloc_mb,
            sdpa_peak_reserved_mb,
        ) = _bench_compressed(
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

        rows.append(
            {
                "sequence_length": sequence_length,
                "section": section,
                "is_mla": is_mla,
                "has_cache": has_cache,
                "context_batch_size": context_batch_size,
                "standard_ms": standard_ms,
                "sdpa_ms": sdpa_ms,
                "noncompressed_ms": noncompressed_ms,
                "compressed_ms": compressed_ms,
                "standard_tflops": standard_tflops,
                "sdpa_tflops": sdpa_tflops,
                "noncompressed_tflops": noncompressed_tflops,
                "compressed_tflops": compressed_tflops,
                "standard_peak_alloc_mb": standard_peak_alloc_mb,
                "sdpa_peak_alloc_mb": sdpa_peak_alloc_mb,
                "noncompressed_peak_alloc_mb": noncompressed_peak_alloc_mb,
                "compressed_peak_alloc_mb": compressed_peak_alloc_mb,
                "standard_peak_reserved_mb": standard_peak_reserved_mb,
                "sdpa_peak_reserved_mb": sdpa_peak_reserved_mb,
                "noncompressed_peak_reserved_mb": noncompressed_peak_reserved_mb,
                "compressed_peak_reserved_mb": compressed_peak_reserved_mb,
            }
        )
    return rows


def _print_results(rows: list[dict], *, batch_size: int):
    sections = [section for section, _, _, _ in SECTION_SPECS]
    print(f"\n=== sequence length benchmark (batch_size={batch_size}) ===")
    for section in sections:
        section_rows = sorted(
            (row for row in rows if row["section"] == section),
            key=lambda row: row["sequence_length"],
        )
        if not section_rows:
            continue

        print(f"\n{section}")
        print(
            "{:<10} {:>10} {:>10} {:>10} {:>10}".format(
                "SEQ_LEN",
                "NAIVE_MS",
                "SDPA_MS",
                "NONC_MS",
                "COMP_MS",
            )
        )
        for row in section_rows:
            print(
                "{:<10d} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f}".format(
                    row["sequence_length"],
                    row["standard_ms"],
                    row["sdpa_ms"],
                    row["noncompressed_ms"],
                    row["compressed_ms"],
                )
            )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark naive attention, PyTorch SDPA, non-compressed kernel, and "
            "compressed kernel latency vs sequence length for four selected sections: "
            "non-MLA/MLA x non-cache/single-cache."
        )
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=20)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--latent-head-dim", type=int, default=128)
    parser.add_argument("--num-context-seqs-per-query", type=int, default=7)
    parser.add_argument("--min-seq-len", type=int, default=2)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--block-m", type=int, default=64)
    parser.add_argument("--block-n", type=int, default=32)
    parser.add_argument("--auto-tune-tiles", action="store_true")
    parser.add_argument("--q-ratio-min", type=float, default=1.0)
    parser.add_argument("--q-ratio-max", type=float, default=1.0)
    parser.add_argument("--kv-ratio-min", type=float, default=1.0)
    parser.add_argument("--kv-ratio-max", type=float, default=1.0)
    parser.add_argument("--plots-dir", type=str, default="plots")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--check-correctness", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    if args.batch_size < 1:
        raise ValueError("batch-size must be >= 1")
    if args.num_context_seqs_per_query < 1:
        raise ValueError("num-context-seqs-per-query must be >= 1")
    if args.head_dim < 1 or args.latent_head_dim < 1:
        raise ValueError("head dims must be >= 1")

    sequence_lengths = _sequence_lengths(args.min_seq_len, args.max_seq_len)
    device = triton.runtime.driver.active.get_active_torch_device()
    plots_dir = Path(args.plots_dir)

    torch.manual_seed(args.seed)

    rows: list[dict] = []
    for sequence_length in sequence_lengths:
        rows.extend(
            _bench_rows_for_sequence_length(
                sequence_length=sequence_length,
                batch_size=args.batch_size,
                num_heads=args.num_heads,
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
        )

    _print_results(rows, batch_size=args.batch_size)
    write_sequence_length_line_plots(
        rows,
        sections=[section for section, _, _, _ in SECTION_SPECS],
        batch_size=args.batch_size,
        plots_dir=plots_dir,
    )
    print(f"\nSaved plots to: {plots_dir.resolve()}")


if __name__ == "__main__":
    main()
