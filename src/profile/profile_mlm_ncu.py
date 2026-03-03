import argparse

import torch

from flash_mlm import (
    InferenceCache,
    build_pack_metadata,
    flash_attn_mlm,
    flash_attn_mlm_compressed,
)


def _run_dense(
    *,
    batch_size: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    dtype: torch.dtype,
    warmup: int,
    iters: int,
    prefill: bool,
    is_mla: bool,
) -> None:
    q = torch.randn(
        batch_size, num_heads, seq_len, head_dim, device="cuda", dtype=dtype
    )
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    cache = InferenceCache()
    layer_id = 0
    scale = 1.0 / (head_dim**0.5)

    if prefill:
        flash_attn_mlm(
            q,
            k,
            v,
            scale=scale,
            inference_cache=cache,
            layer_id=layer_id,
            is_mla=is_mla,
            prefill=True,
        )

    for _ in range(warmup):
        flash_attn_mlm(
            q,
            k,
            v,
            scale=scale,
            inference_cache=cache,
            layer_id=layer_id,
            is_mla=is_mla,
            prefill=False,
        )
    torch.cuda.synchronize()

    for _ in range(iters):
        flash_attn_mlm(
            q,
            k,
            v,
            scale=scale,
            inference_cache=cache,
            layer_id=layer_id,
            is_mla=is_mla,
            prefill=False,
        )
    torch.cuda.synchronize()


def _run_compressed(
    *,
    batch_size: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    active_tokens: int,
    block_n: int,
    dtype: torch.dtype,
    warmup: int,
    iters: int,
    prefill: bool,
    is_mla: bool,
    causal_query_seq_attn: bool,
) -> None:
    q = torch.randn(
        batch_size, num_heads, seq_len, head_dim, device="cuda", dtype=dtype
    )
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    lengths = torch.full(
        (batch_size,),
        fill_value=active_tokens,
        device="cuda",
        dtype=torch.int32,
    )
    q_meta = build_pack_metadata(lengths, N=seq_len, block_n=block_n)

    cache = InferenceCache()
    layer_id = 0
    scale = 1.0 / (head_dim**0.5)

    if prefill:
        flash_attn_mlm_compressed(
            q,
            k,
            v,
            num_heads=num_heads,
            q_meta=q_meta,
            scale=scale,
            inference_cache=cache,
            layer_id=layer_id,
            is_mla=is_mla,
            prefill=True,
            causal_query_seq_attn=causal_query_seq_attn,
            block_n=block_n,
        )

    for _ in range(warmup):
        flash_attn_mlm_compressed(
            q,
            k,
            v,
            num_heads=num_heads,
            q_meta=q_meta,
            scale=scale,
            inference_cache=cache,
            layer_id=layer_id,
            is_mla=is_mla,
            prefill=False,
            causal_query_seq_attn=causal_query_seq_attn,
            block_n=block_n,
        )
    torch.cuda.synchronize()

    for _ in range(iters):
        flash_attn_mlm_compressed(
            q,
            k,
            v,
            num_heads=num_heads,
            q_meta=q_meta,
            scale=scale,
            inference_cache=cache,
            layer_id=layer_id,
            is_mla=is_mla,
            prefill=False,
            causal_query_seq_attn=causal_query_seq_attn,
            block_n=block_n,
        )
    torch.cuda.synchronize()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Nsight Compute driver for flash-mlm")
    parser.add_argument("--mode", choices=["dense", "compressed"], default="compressed")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--active-tokens", type=int, default=512)
    parser.add_argument("--block-n", type=int, default=32)
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="bf16")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--prefill", action="store_true")
    parser.add_argument("--is-mla", action="store_true")
    parser.add_argument("--causal-query-seq-attn", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16

    if args.mode == "dense":
        _run_dense(
            batch_size=args.batch_size,
            num_heads=args.num_heads,
            seq_len=args.seq_len,
            head_dim=args.head_dim,
            dtype=dtype,
            warmup=args.warmup,
            iters=args.iters,
            prefill=args.prefill,
            is_mla=args.is_mla,
        )
        return

    if args.active_tokens < 1 or args.active_tokens > args.seq_len:
        raise ValueError("active-tokens must be in [1, seq-len]")

    _run_compressed(
        batch_size=args.batch_size,
        num_heads=args.num_heads,
        seq_len=args.seq_len,
        head_dim=args.head_dim,
        active_tokens=args.active_tokens,
        block_n=args.block_n,
        dtype=dtype,
        warmup=args.warmup,
        iters=args.iters,
        prefill=args.prefill,
        is_mla=args.is_mla,
        causal_query_seq_attn=args.causal_query_seq_attn,
    )


if __name__ == "__main__":
    main()
