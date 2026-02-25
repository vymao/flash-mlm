import math

import torch
import triton

from flash_mlm.host import flash_attn_mlm_compressed
from flash_mlm.host_utils import build_pack_metadata
from flash_mlm.kernel_utils import is_hip

DEVICE = triton.runtime.driver.active.get_active_torch_device()
N_HEADS_DEFAULT = 32
QUERY_SEQ_LEN = 600
FIXED_CONTEXT_LEN = 500 if is_hip() else 1000


def make_varlen_lengths(
    batch_size: int, max_len: int, min_ratio: float, max_ratio: float
):
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if max_len < 1:
        raise ValueError("max_len must be >= 1")

    if batch_size == 1:
        return torch.tensor([max_len], device=DEVICE, dtype=torch.int32)

    ratios = torch.linspace(min_ratio, max_ratio, steps=batch_size, device=DEVICE)
    lengths = torch.clamp((ratios * max_len).to(torch.int32), min=1, max=max_len)
    return lengths


def build_cu_seqlens(lengths: torch.Tensor):
    cu = torch.zeros(lengths.numel() + 1, device=lengths.device, dtype=torch.int32)
    cu[1:] = lengths.cumsum(dim=0)
    return cu


def estimate_tflops(
    lengths_q: torch.Tensor,
    lengths_kv: torch.Tensor,
    num_heads: int,
    head_dim: int,
    latency_ms: float,
):
    # For each batch/head: QK matmul + PV matmul ~= 4 * Lq * (Lq + Lkv) * D flops.
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


def run_mlm_compressed_case(
    *,
    batch_size: int,
    num_heads: int,
    context_len: int,
    context_batch_size: int,
    head_dim: int,
    is_mla: bool,
    has_cache: bool,
    tile: str,
    return_latency_ms: bool = False,
    device=DEVICE,
):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for benchmark")

    block_m, block_n = (int(x) for x in tile.split("x"))

    dtype = torch.float16
    if context_batch_size not in (1, batch_size):
        raise ValueError("context_batch_size must be 1 or equal to batch_size")

    q = torch.randn(
        (batch_size, num_heads, QUERY_SEQ_LEN, head_dim), device=device, dtype=dtype
    )
    k = torch.randn(
        (batch_size, num_heads, QUERY_SEQ_LEN, head_dim), device=device, dtype=dtype
    )
    v = torch.randn(
        (batch_size, num_heads, QUERY_SEQ_LEN, head_dim), device=device, dtype=dtype
    )

    lengths_q = make_varlen_lengths(
        batch_size, QUERY_SEQ_LEN, min_ratio=0.65, max_ratio=1.0
    )
    q_meta = build_pack_metadata(lengths_q, QUERY_SEQ_LEN, block_n=block_n)

    if has_cache:
        cache_max_len = max(1, context_len)
        lengths_kv = make_varlen_lengths(
            context_batch_size, cache_max_len, min_ratio=0.5, max_ratio=1.0
        )
    else:
        lengths_kv = torch.zeros(context_batch_size, device=device, dtype=torch.int32)

    cu_seqlens_kv = build_cu_seqlens(lengths_kv)
    total_context_len = int(cu_seqlens_kv[-1].item())

    if is_mla:
        k_cache = torch.randn((total_context_len, head_dim), device=device, dtype=dtype)
        v_cache = torch.randn_like(k_cache)
    else:
        k_cache = torch.randn(
            (num_heads * total_context_len, head_dim), device=device, dtype=dtype
        )
        v_cache = torch.randn_like(k_cache)

    sm_scale = 1.0 / math.sqrt(head_dim)

    fn = lambda: flash_attn_mlm_compressed(
        q,
        k,
        v,
        k_cache,
        v_cache,
        num_heads=num_heads,
        q_meta=q_meta,
        total_context_len=total_context_len,
        cu_seqlens_kv=cu_seqlens_kv,
        scale=sm_scale,
        is_mla=is_mla,
        context_batch_size=context_batch_size,
        block_m=block_m,
        block_n=block_n,
    )
    ms = triton.testing.do_bench(fn)

    if return_latency_ms:
        return ms

    return estimate_tflops(lengths_q, lengths_kv, num_heads, head_dim, ms)
