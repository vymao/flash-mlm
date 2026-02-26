from pathlib import Path

import matplotlib.pyplot as plt


def _plot_metric(
    rows: list[dict],
    *,
    standard_key: str,
    noncompressed_key: str,
    compressed_key: str,
    ylabel: str,
    title: str,
    out_path: Path,
):
    labels = [row["section"] for row in rows]
    x = list(range(len(rows)))
    standard_vals = [row[standard_key] for row in rows]
    noncompressed_vals = [row[noncompressed_key] for row in rows]
    compressed_vals = [row[compressed_key] for row in rows]
    width = 0.26

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(
        [i - width for i in x], standard_vals, width=width, label="Standard Attention"
    )
    ax.bar(
        [i for i in x], noncompressed_vals, width=width, label="Non-Compressed Kernel"
    )
    ax.bar(
        [i + width for i in x], compressed_vals, width=width, label="Compressed Kernel"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def write_benchmark_bar_plots(rows: list[dict], *, context_len: int, plots_dir: Path):
    _plot_metric(
        rows,
        standard_key="standard_ms",
        noncompressed_key="noncompressed_ms",
        compressed_key="compressed_ms",
        ylabel="Latency (ms)",
        title=f"attention latency by section (context_len={context_len})",
        out_path=plots_dir / f"mlm-threeway-latency-ctx{context_len}.png",
    )
    _plot_metric(
        rows,
        standard_key="standard_tflops",
        noncompressed_key="noncompressed_tflops",
        compressed_key="compressed_tflops",
        ylabel="Estimated TFLOPS",
        title=f"attention TFLOPS by section (context_len={context_len})",
        out_path=plots_dir / f"mlm-threeway-tflops-ctx{context_len}.png",
    )
