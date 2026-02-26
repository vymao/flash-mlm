# flash-mlm

Triton-based attention kernels optimized for MLM-based models, in PyTorch.

## Highlights

- ✅ Packed variable-length query/key/value path
- ✅ Optional packed KV cache path
- ✅ MLA + non-MLA support (no RoPE)
- ✅ Benchmarks for context scaling and batch scaling

```text
[B,H,N,D] padded tensors
   -> pack metadata
   -> fused Triton MLM kernel
   -> packed output
   -> unpack to [B,H,N,D]
```

## On the Roadmap
- **Backpropagation support**: Generally this is not as critical because one can always just use `flash_attn_varlen_func` in the usual Flash-Attention implementation, though that requires more kernels.
- **Faster and more efficient tile scheduling**: Right now, we don't explicitly include a tile scheduler because we pack the sequences to omit non-attended tokens (ex. padding). This costs overhead in the host function.
- **Variable context batch sizes**: Currently, only a single context or a full batch (equal to the query batch size) is supported. Variable context batch sizes with index mapping would be a useful extension.
- **Compatibility with RoPE**: Currently, we don't support RoPE with MLA.
- **Paged Attention for KV cache**

Contributions are most welcome!

## Requirements

- Linux
- Python `>=3.11,<3.12`
- GPU runtime compatible with your installed `torch` + `triton`

## Install

```bash
# latest main
pip install "flash-mlm @ git+https://github.com/vymao/flash-mlm.git"

# pinned tag
pip install "flash-mlm @ git+https://github.com/vymao/flash-mlm.git@v0.1.0"
```

## Quick usage

```python
import torch
from flash_mlm import flash_attn_mlm_compressed, build_pack_metadata, unpack_from_kernel

B, H, N, D = 2, 8, 128, 64
q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
k = torch.randn_like(q)
v = torch.randn_like(q)

lengths_q = torch.tensor([N, N - 16], device="cuda", dtype=torch.int32)
q_meta = build_pack_metadata(lengths_q, N=N, block_n=64)

cu_seqlens_kv = torch.zeros(B + 1, device="cuda", dtype=torch.int32)
total_context_len = 0
k_cache = torch.empty((H * total_context_len, D), device="cuda", dtype=torch.float16)
v_cache = torch.empty_like(k_cache)

packed_out = flash_attn_mlm_compressed(
    q, k, v, k_cache, v_cache,
    num_heads=H,
    q_meta=q_meta,
    total_context_len=total_context_len,
    cu_seqlens_kv=cu_seqlens_kv,
    scale=1.0 / (D ** 0.5),
)
out = unpack_from_kernel(packed_out, q_meta, H=H)
```

## Benchmarking

```bash
python src/benchmark/benchmark_mlm_nctx.py
python src/benchmark/benchmark_mlm_batch.py
```

Outputs are written to `plots/`.

## Development

```bash
poetry install
poetry run pytest -q src/test/test_host_utils.py
poetry run pytest -q src/test/test.py::test_mlm_compressed_matches_reference
```

## Public API

- `flash_attn_mlm`
- `flash_attn_mlm_compressed`
- `PackMetadata`
- `build_pack_metadata`
- `unpack_from_kernel`
