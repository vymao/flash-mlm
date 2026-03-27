from pathlib import Path

import matplotlib.pyplot as plt


def _plot_metric(
    rows: list[dict],
    *,
    standard_key: str,
    sdpa_key: str,
    noncompressed_key: str,
    compressed_key: str,
    ylabel: str,
    title: str,
    out_path: Path,
):
    labels = [row["section"] for row in rows]
    x = list(range(len(rows)))
    standard_vals = [row[standard_key] for row in rows]
    sdpa_vals = [row[sdpa_key] for row in rows]
    noncompressed_vals = [row[noncompressed_key] for row in rows]
    compressed_vals = [row[compressed_key] for row in rows]
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(
        [i - 1.5 * width for i in x],
        standard_vals,
        width=width,
        label="Naive Attention",
    )
    ax.bar([i - 0.5 * width for i in x], sdpa_vals, width=width, label="PyTorch SDPA")
    ax.bar(
        [i + 0.5 * width for i in x],
        noncompressed_vals,
        width=width,
        label="Non-Compressed Kernel",
    )
    ax.bar(
        [i + 1.5 * width for i in x],
        compressed_vals,
        width=width,
        label="Compressed Kernel",
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


def write_benchmark_bar_plots(
    rows: list[dict],
    *,
    mean_context_subseq_len: float,
    query_batch_size: int,
    plots_dir: Path,
):
    title_suffix = (
        "mean_true_context_subseq_len="
        f"{mean_context_subseq_len:.2f}, query_batch_size={query_batch_size}"
    )
    _plot_metric(
        rows,
        standard_key="standard_ms",
        sdpa_key="sdpa_ms",
        noncompressed_key="noncompressed_ms",
        compressed_key="compressed_ms",
        ylabel="Latency (ms)",
        title=f"Attention latency by section ({title_suffix})",
        out_path=plots_dir / "mlm-fourway-latency.png",
    )
    _plot_metric(
        rows,
        standard_key="standard_tflops",
        sdpa_key="sdpa_tflops",
        noncompressed_key="noncompressed_tflops",
        compressed_key="compressed_tflops",
        ylabel="Estimated TFLOPS",
        title=f"Attention TFLOPS by section ({title_suffix})",
        out_path=plots_dir / "mlm-fourway-tflops.png",
    )
    _plot_metric(
        rows,
        standard_key="standard_peak_alloc_mb",
        sdpa_key="sdpa_peak_alloc_mb",
        noncompressed_key="noncompressed_peak_alloc_mb",
        compressed_key="compressed_peak_alloc_mb",
        ylabel="Peak CUDA memory allocated delta (MB)",
        title=f"Attention peak allocated memory by section ({title_suffix})",
        out_path=plots_dir / "mlm-fourway-peak-alloc-mb.png",
    )
    _plot_metric(
        rows,
        standard_key="standard_peak_reserved_mb",
        sdpa_key="sdpa_peak_reserved_mb",
        noncompressed_key="noncompressed_peak_reserved_mb",
        compressed_key="compressed_peak_reserved_mb",
        ylabel="Peak CUDA memory reserved (MB)",
        title=f"Attention peak reserved memory by section ({title_suffix})",
        out_path=plots_dir / "mlm-fourway-peak-reserved-mb.png",
    )


def _section_plot_slug(section: str) -> str:
    return (
        section.lower()
        .replace(" + ", "-")
        .replace(" ", "-")
        .replace("(", "")
        .replace(")", "")
    )


def write_sequence_length_line_plots(
    rows: list[dict],
    *,
    sections: list[str],
    batch_size: int,
    plots_dir: Path,
):
    method_specs = [
        ("standard_ms", "Naive Attention", "o"),
        ("sdpa_ms", "PyTorch SDPA", "s"),
        ("noncompressed_ms", "Non-Compressed Kernel", "^"),
        ("compressed_ms", "Compressed Kernel", "D"),
    ]

    for section in sections:
        section_rows = sorted(
            (row for row in rows if row["section"] == section),
            key=lambda row: row["sequence_length"],
        )
        if not section_rows:
            continue

        seq_lengths = [row["sequence_length"] for row in section_rows]
        fig, ax = plt.subplots(figsize=(10, 6))
        for key, label, marker in method_specs:
            ax.plot(
                seq_lengths,
                [row[key] for row in section_rows],
                marker=marker,
                linewidth=2,
                label=label,
            )

        ax.set_xscale("log", base=2)
        ax.set_xticks(seq_lengths)
        ax.set_xticklabels([str(seq_len) for seq_len in seq_lengths])
        ax.set_xlabel("Sequence length")
        ax.set_ylabel("Latency (ms)")
        ax.set_title(f"{section} latency vs sequence length (batch_size={batch_size})")
        ax.grid(True, which="both", axis="y", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        out_path = plots_dir / f"mlm-sequence-length-{_section_plot_slug(section)}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
