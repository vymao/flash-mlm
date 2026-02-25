# flash-mlm

Triton-based Flash MLM attention kernels for PyTorch.

## Requirements

- Linux
- Python `>=3.11,<3.12`
- NVIDIA or AMD GPU supported by your installed `torch` + `triton`

## Install

### From GitHub (latest main)

```bash
pip install "flash-mlm @ git+https://github.com/vymao/flash-mlm.git"
```

### From GitHub tag (recommended for reproducibility)

```bash
pip install "flash-mlm @ git+https://github.com/vymao/flash-mlm.git@v0.1.0"
```

## Quick usage (PyTorch)

```python
import torch
from flash_mlm import flash_attn_mlm_compressed, build_pack_metadata, unpack_from_kernel

device = "cuda"
B, H, N, D = 2, 8, 128, 64
scale = 1.0 / (D ** 0.5)

q = torch.randn(B, H, N, D, device=device, dtype=torch.float16)
k = torch.randn(B, H, N, D, device=device, dtype=torch.float16)
v = torch.randn(B, H, N, D, device=device, dtype=torch.float16)

lengths_q = torch.tensor([N, N - 16], device=device, dtype=torch.int32)
q_meta = build_pack_metadata(lengths_q, N=N, block_n=64)

cu_seqlens_kv = torch.zeros(B + 1, device=device, dtype=torch.int32)
total_context_len = 0
k_cache = torch.empty((H * total_context_len, D), device=device, dtype=torch.float16)
v_cache = torch.empty_like(k_cache)

packed_out = flash_attn_mlm_compressed(
    q,
    k,
    v,
    k_cache,
    v_cache,
    num_heads=H,
    q_meta=q_meta,
    total_context_len=total_context_len,
    cu_seqlens_kv=cu_seqlens_kv,
    scale=scale,
    is_mla=False,
)
out = unpack_from_kernel(packed_out, q_meta, H=H)
```

## Development

Install editable project dependencies with Poetry:

```bash
poetry install
```

Run tests:

```bash
poetry run pytest -q src/test/test_host_utils.py
poetry run pytest -q src/test/test.py::test_mlm_compressed_matches_reference
```

Run benchmarks:

```bash
python src/benchmark/benchmark_mlm_nctx.py
python src/benchmark/benchmark_mlm_batch.py
```

Plots/tables are written to `plots/`.
