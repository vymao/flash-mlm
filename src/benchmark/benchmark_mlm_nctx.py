import sys
from pathlib import Path

import triton

_THIS_DIR = Path(__file__).resolve().parent
_PARENT = _THIS_DIR.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

from benchmark.common import (
    N_HEADS_DEFAULT,
    DEVICE,
    run_mlm_compressed_case,
)  # noqa: E402
from flash_mlm.kernel_utils import is_hip  # noqa: E402

BATCH = 4
PLOTS_DIR = Path("plots")

configs = []
x_vals = [2**i for i in (7, 8, 9, 10)]
if not is_hip():
    x_vals.append(2**11)

for head_dim in [64, 128]:
    for is_mla in [False, True]:
        for has_cache in [False, True]:
            for context_batch_size in [1, BATCH]:
                configs.append(
                    triton.testing.Benchmark(
                        x_names=["N_CTX"],
                        x_vals=x_vals,
                        line_arg="tile",
                        line_vals=["32x32", "64x64"],
                        line_names=["BLOCK_M=32,BLOCK_N=32", "BLOCK_M=64,BLOCK_N=64"],
                        styles=[("red", "-"), ("blue", "-")],
                        ylabel="TFLOPS",
                        plot_name=(
                            f"mlm-compressed-batch{BATCH}-heads{N_HEADS_DEFAULT}-d{head_dim}"
                            f"-mla={is_mla}-cache={has_cache}-ctxbs={context_batch_size}"
                        ),
                        args={
                            "BATCH": BATCH,
                            "H": N_HEADS_DEFAULT,
                            "HEAD_DIM": head_dim,
                            "is_mla": is_mla,
                            "has_cache": has_cache,
                            "context_batch_size": context_batch_size,
                        },
                    )
                )


@triton.testing.perf_report(configs)
def bench_mlm_compressed_nctx(
    BATCH,
    H,
    N_CTX,
    HEAD_DIM,
    is_mla,
    has_cache,
    context_batch_size,
    tile,
    device=DEVICE,
):
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


if __name__ == "__main__":
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    bench_mlm_compressed_nctx.run(save_path=str(PLOTS_DIR), print_data=True)
