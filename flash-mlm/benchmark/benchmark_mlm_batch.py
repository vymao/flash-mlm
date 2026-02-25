import sys
import csv
import math
from pathlib import Path

import torch
import triton

_THIS_DIR = Path(__file__).resolve().parent
_PARENT = _THIS_DIR.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

from benchmark.common import (  # noqa: E402
    DEVICE,
    FIXED_CONTEXT_LEN,
    N_HEADS_DEFAULT,
    run_mlm_compressed_case,
)

PLOTS_DIR = Path("plots")
TILE_OPTIONS = [
    "64x32",
    "64x64",
    "64x128",
    "128x32",
    "128x64",
    "128x128",
]
TILE_NAMES = [
    "BLOCK_M=64,BLOCK_N=32",
    "BLOCK_M=64,BLOCK_N=64",
    "BLOCK_M=64,BLOCK_N=128",
    "BLOCK_M=128,BLOCK_N=32",
    "BLOCK_M=128,BLOCK_N=64",
    "BLOCK_M=128,BLOCK_N=128",
]
TILE_STYLES = [
    ("red", "-"),
    ("blue", "-"),
    ("green", "-"),
    ("orange", "-"),
    ("purple", "-"),
    ("brown", "-"),
]


def _to_float(value: str):
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return parsed


def print_best_latency_per_batch(save_dir: Path):
    latency_csvs = sorted(save_dir.glob("mlm-compressed-batch-scaling-latency-*.csv"))
    if not latency_csvs:
        print("No latency CSV files found to summarize best tile per batch.")
        return

    print("\n=== Best latency tile per batch ===")
    for csv_path in latency_csvs:
        print(f"{csv_path.stem}:")
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                continue

            metric_cols = [c for c in reader.fieldnames if c != "BATCH"]
            for row in reader:
                batch = row.get("BATCH", "?")
                best_col = None
                best_val = float("inf")
                for col in metric_cols:
                    value = _to_float(row.get(col, "nan"))
                    if math.isnan(value):
                        continue
                    if value < best_val:
                        best_val = value
                        best_col = col

                if best_col is None:
                    print(f"  BATCH={batch}: no valid value (OOM or missing)")
                    continue

                tile_name = best_col.replace(" (Latency (ms))", "")
                print(f"  BATCH={batch}: {tile_name} @ {best_val:.4f} ms")


batch_configs = []
latency_batch_configs = []
batch_x_vals = [2**i for i in range(9)]
for head_dim in [64, 128]:
    for is_mla in [False, True]:
        for has_cache in [False, True]:
            for context_batch_mode in ["shared", "match"]:
                batch_configs.append(
                    triton.testing.Benchmark(
                        x_names=["BATCH"],
                        x_vals=batch_x_vals,
                        line_arg="tile",
                        line_vals=TILE_OPTIONS,
                        line_names=TILE_NAMES,
                        styles=TILE_STYLES,
                        ylabel="TFLOPS",
                        plot_name=(
                            f"mlm-compressed-batch-scaling-fixedN{FIXED_CONTEXT_LEN}"
                            f"-heads{N_HEADS_DEFAULT}-d{head_dim}-mla={is_mla}-cache={has_cache}"
                            f"-ctxbs={context_batch_mode}"
                        ),
                        args={
                            "H": N_HEADS_DEFAULT,
                            "N_CTX": FIXED_CONTEXT_LEN,
                            "HEAD_DIM": head_dim,
                            "is_mla": is_mla,
                            "has_cache": has_cache,
                            "context_batch_mode": context_batch_mode,
                        },
                    )
                )
                latency_batch_configs.append(
                    triton.testing.Benchmark(
                        x_names=["BATCH"],
                        x_vals=batch_x_vals,
                        line_arg="tile",
                        line_vals=TILE_OPTIONS,
                        line_names=TILE_NAMES,
                        styles=TILE_STYLES,
                        ylabel="Latency (ms)",
                        plot_name=(
                            f"mlm-compressed-batch-scaling-latency-fixedN{FIXED_CONTEXT_LEN}"
                            f"-heads{N_HEADS_DEFAULT}-d{head_dim}-mla={is_mla}-cache={has_cache}"
                            f"-ctxbs={context_batch_mode}"
                        ),
                        args={
                            "H": N_HEADS_DEFAULT,
                            "N_CTX": FIXED_CONTEXT_LEN,
                            "HEAD_DIM": head_dim,
                            "is_mla": is_mla,
                            "has_cache": has_cache,
                            "context_batch_mode": context_batch_mode,
                        },
                    )
                )


@triton.testing.perf_report(batch_configs)
def bench_mlm_compressed_batch_scaling(
    BATCH,
    H,
    N_CTX,
    HEAD_DIM,
    is_mla,
    has_cache,
    context_batch_mode,
    tile,
    device=DEVICE,
):
    context_batch_size = 1 if context_batch_mode == "shared" else BATCH
    try:
        return run_mlm_compressed_case(
            batch_size=BATCH,
            num_heads=H,
            context_len=N_CTX,
            context_batch_size=context_batch_size,
            head_dim=HEAD_DIM,
            is_mla=is_mla,
            has_cache=has_cache,
            tile=tile,
            device=device,
        )
    except RuntimeError as err:
        if "out of memory" in str(err).lower():
            torch.cuda.empty_cache()
            return float("nan")
        raise


@triton.testing.perf_report(latency_batch_configs)
def bench_mlm_compressed_batch_scaling_latency(
    BATCH,
    H,
    N_CTX,
    HEAD_DIM,
    is_mla,
    has_cache,
    context_batch_mode,
    tile,
    device=DEVICE,
):
    context_batch_size = 1 if context_batch_mode == "shared" else BATCH
    try:
        return run_mlm_compressed_case(
            batch_size=BATCH,
            num_heads=H,
            context_len=N_CTX,
            context_batch_size=context_batch_size,
            head_dim=HEAD_DIM,
            is_mla=is_mla,
            has_cache=has_cache,
            tile=tile,
            return_latency_ms=True,
            device=device,
        )
    except RuntimeError as err:
        if "out of memory" in str(err).lower():
            torch.cuda.empty_cache()
            return float("nan")
        raise


if __name__ == "__main__":
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    bench_mlm_compressed_batch_scaling.run(save_path=str(PLOTS_DIR), print_data=True)
    bench_mlm_compressed_batch_scaling_latency.run(
        save_path=str(PLOTS_DIR), print_data=True
    )
    print_best_latency_per_batch(PLOTS_DIR)
