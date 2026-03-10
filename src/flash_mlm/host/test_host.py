from flash_mlm.host import (
    flash_attn_mlm,
    flash_attn_mlm_compressed,
    flash_attn_mlm_precompressed,
)
from flash_mlm.host.host_utils import (
    build_pack_metadata,
    unpack_from_kernel,
    pack_for_kernel,
)
from flash_mlm.host.cache import InferenceCache
import torch
import pytest
import triton

from flash_mlm.kernel_utils import is_hip, is_blackwell, is_hopper

try:
    from flash_mlm.kernel import _attention

    attention = _attention.apply
except Exception:
    _attention = None
    attention = None

TORCH_HAS_FP8 = hasattr(torch, "float8_e5m2")
DEVICE = triton.runtime.driver.active.get_active_torch_device()


def reference_attention(q, k, v, lengths, scale, is_mla):
    """Reference for per-batch variable-length self-attention on padded [B, H, N, D]."""
    bsz, n_heads, n_ctx, d = q.shape
    out = torch.zeros_like(q)
    for b in range(bsz):
        L = int(lengths[b].item())
        if L == 0:
            continue
        for h in range(n_heads):
            q_bh = q[b, h, :L, :]
            k_bh = k[b, 0 if is_mla else h, :L, :]
            v_bh = k_bh if is_mla else v[b, h, :L, :]
            logits = torch.matmul(q_bh, k_bh.transpose(0, 1)) * scale
            probs = torch.softmax(logits.float(), dim=-1).to(q.dtype)
            out[b, h, :L, :] = torch.matmul(probs, v_bh)
    return out


def _reference_varlen_attention_with_cache(
    q,
    k,
    v,
    lengths_q,
    k_cache,
    v_cache,
    cu_seqlens_kv,
    scale,
    is_mla,
    causal_query_seq_attn=False,
):
    """Reference for per-batch varlen attention with packed cache + current tokens."""
    bsz, n_heads, _, _ = q.shape
    total_context_len = int(cu_seqlens_kv[-1].item())
    out = torch.zeros_like(q)
    for b in range(bsz):
        Lq = int(lengths_q[b].item())
        if Lq == 0:
            continue
        c_start = int(cu_seqlens_kv[b].item())
        c_end = int(cu_seqlens_kv[b + 1].item())
        for h in range(n_heads):
            q_bh = q[b, h, :Lq, :]
            if is_mla:
                if causal_query_seq_attn:
                    k_main = torch.cat(
                        [k[j, 0, : int(lengths_q[j].item()), :] for j in range(b + 1)],
                        dim=0,
                    )
                else:
                    k_main = k[b, 0, :Lq, :]
                v_main = k_main
                k_ctx = k_cache[c_start:c_end, :]
                v_ctx = k_ctx
            else:
                if causal_query_seq_attn:
                    k_main = torch.cat(
                        [k[j, h, : int(lengths_q[j].item()), :] for j in range(b + 1)],
                        dim=0,
                    )
                    v_main = torch.cat(
                        [v[j, h, : int(lengths_q[j].item()), :] for j in range(b + 1)],
                        dim=0,
                    )
                else:
                    k_main = k[b, h, :Lq, :]
                    v_main = v[b, h, :Lq, :]
                row_offset = h * total_context_len
                k_ctx = k_cache[row_offset + c_start : row_offset + c_end, :]
                v_ctx = v_cache[row_offset + c_start : row_offset + c_end, :]

            k_all = torch.cat([k_ctx, k_main], dim=0)
            v_all = torch.cat([v_ctx, v_main], dim=0)
            logits = torch.matmul(q_bh, k_all.transpose(0, 1)) * scale
            probs = torch.softmax(logits.float(), dim=-1).to(q.dtype)
            out[b, h, :Lq, :] = torch.matmul(probs, v_all)
    return out


def _reference_dense_attention_with_flat_cache(
    q,
    k,
    v,
    k_cache,
    v_cache,
    context_len,
    scale,
    is_mla,
    context_batch_size,
):
    bsz, n_heads, n_ctx, _ = q.shape
    out = torch.zeros_like(q)
    for b in range(bsz):
        ctx_batch = 0 if context_batch_size == 1 else b
        for h in range(n_heads):
            q_bh = q[b, h, :n_ctx, :]
            if is_mla:
                k_main = k[b, h, :n_ctx, :]
                v_main = k_main
                c_start = ctx_batch * context_len
                c_end = c_start + context_len
                k_ctx = k_cache[c_start:c_end, :]
                v_ctx = k_ctx
            else:
                c_start = (ctx_batch * n_heads + h) * context_len
                c_end = c_start + context_len
                k_main = k[b, h, :n_ctx, :]
                v_main = v[b, h, :n_ctx, :]
                k_ctx = k_cache[c_start:c_end, :]
                v_ctx = v_cache[c_start:c_end, :]

            k_all = torch.cat([k_ctx, k_main], dim=0)
            v_all = torch.cat([v_ctx, v_main], dim=0)
            logits = torch.matmul(q_bh, k_all.transpose(0, 1)) * scale
            probs = torch.softmax(logits.float(), dim=-1).to(q.dtype)
            out[b, h, :n_ctx, :] = torch.matmul(probs, v_all)
    return out


@pytest.mark.parametrize("is_mla", [False, True])
def test_mlm_dense_prefill_requires_cache(is_mla):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    torch.manual_seed(21)
    B, H, N, D = 2, 2, 16, 64
    q = torch.randn((B, H, N, D), device=DEVICE, dtype=torch.float16)
    k = torch.randn((B, H, N, D), device=DEVICE, dtype=torch.float16)
    v = k if is_mla else torch.randn((B, H, N, D), device=DEVICE, dtype=torch.float16)

    with pytest.raises(ValueError, match="prefill=True requires inference_cache"):
        flash_attn_mlm(
            q,
            k,
            v,
            scale=0.5,
            inference_cache=None,
            layer_id=None,
            is_mla=is_mla,
            block_m=32,
            block_n=32,
            prefill=True,
        )


@pytest.mark.parametrize("is_mla", [False, True])
def test_mlm_dense_no_cache_matches_reference(is_mla):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    torch.manual_seed(22)
    B, H, N, D = 2, 3, 24, 64
    scale = 0.6
    q = torch.randn((B, H, N, D), device=DEVICE, dtype=torch.float16)
    k = torch.randn((B, H, N, D), device=DEVICE, dtype=torch.float16)
    v = k if is_mla else torch.randn((B, H, N, D), device=DEVICE, dtype=torch.float16)

    out = flash_attn_mlm(
        q,
        k,
        v,
        scale=scale,
        inference_cache=None,
        layer_id=None,
        is_mla=is_mla,
        block_m=32,
        block_n=32,
    )

    empty_rows = 0 if is_mla else H * 0
    k_cache = torch.empty((empty_rows, D), device=DEVICE, dtype=torch.float16)
    v_cache = torch.empty_like(k_cache)
    ref = _reference_dense_attention_with_flat_cache(
        q,
        k,
        v,
        k_cache,
        v_cache,
        0,
        scale,
        is_mla,
        context_batch_size=B,
    )
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=0)


@pytest.mark.parametrize("is_mla", [False, True])
def test_mlm_dense_prefill_then_use_cached_context_matches_reference(is_mla):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    torch.manual_seed(23)
    B, H, N, D = 2, 2, 20, 64
    scale = 0.55

    q1 = torch.randn((B, H, N, D), device=DEVICE, dtype=torch.float16)
    k1 = torch.randn((B, H, N, D), device=DEVICE, dtype=torch.float16)
    v1 = k1 if is_mla else torch.randn((B, H, N, D), device=DEVICE, dtype=torch.float16)

    q2 = torch.randn((B, H, N, D), device=DEVICE, dtype=torch.float16)
    k2 = torch.randn((B, H, N, D), device=DEVICE, dtype=torch.float16)
    v2 = k2 if is_mla else torch.randn((B, H, N, D), device=DEVICE, dtype=torch.float16)

    cache = InferenceCache()
    flash_attn_mlm(
        q1,
        k1,
        v1,
        scale=scale,
        inference_cache=cache,
        layer_id=3,
        is_mla=is_mla,
        block_m=32,
        block_n=32,
        prefill=True,
    )

    out = flash_attn_mlm(
        q2,
        k2,
        v2,
        scale=scale,
        inference_cache=cache,
        layer_id=3,
        is_mla=is_mla,
        context_batch_size=B,
        block_m=32,
        block_n=32,
    )

    entry = cache.get_kv_cache(
        3,
        is_mla=is_mla,
        num_heads=H,
        head_dim=D,
        dtype=q2.dtype,
        device=q2.device,
    )
    context_len = entry.total_context_len // entry.context_batch_size
    ref = _reference_dense_attention_with_flat_cache(
        q2,
        k2,
        v2,
        entry.k_cache,
        entry.v_cache,
        context_len,
        scale,
        is_mla,
        entry.context_batch_size,
    )
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=0)


@pytest.mark.parametrize("is_mla", [False, True])
def test_mlm_compressed_prefill_requires_cache(is_mla):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    torch.manual_seed(11)
    B, H, N, D = 2, 2, 16, 64
    lengths = torch.tensor([16, 9], device=DEVICE, dtype=torch.int32)

    q = torch.randn((B, H, N, D), device=DEVICE, dtype=torch.float16)
    k = torch.randn((B, H, N, D), device=DEVICE, dtype=torch.float16)
    v = torch.randn((B, H, N, D), device=DEVICE, dtype=torch.float16)
    q_meta = build_pack_metadata(lengths, N, block_n=32)

    with pytest.raises(ValueError, match="prefill=True requires inference_cache"):
        flash_attn_mlm_compressed(
            q,
            k,
            v,
            num_heads=H,
            scale=0.5,
            q_meta=q_meta,
            inference_cache=None,
            layer_id=None,
            is_mla=is_mla,
            block_m=32,
            block_n=32,
            prefill=True,
        )


@pytest.mark.parametrize("is_mla", [False, True])
def test_mlm_compressed_no_prefill_empty_context_same_with_or_without_cache(is_mla):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    torch.manual_seed(12)
    B, H, N, D = 2, 3, 24, 64
    lengths = torch.tensor([24, 15], device=DEVICE, dtype=torch.int32)
    scale = 0.6

    q = torch.randn((B, H, N, D), device=DEVICE, dtype=torch.float16)
    k = torch.randn((B, H, N, D), device=DEVICE, dtype=torch.float16)
    v = torch.randn((B, H, N, D), device=DEVICE, dtype=torch.float16)
    q_meta = build_pack_metadata(lengths, N, block_n=32)

    out_no_cache = flash_attn_mlm_compressed(
        q,
        k,
        v,
        num_heads=H,
        scale=scale,
        q_meta=q_meta,
        inference_cache=None,
        layer_id=None,
        is_mla=is_mla,
        block_m=32,
        block_n=32,
        prefill=False,
    )

    cache = InferenceCache()
    out_with_cache = flash_attn_mlm_compressed(
        q,
        k,
        v,
        num_heads=H,
        scale=scale,
        q_meta=q_meta,
        inference_cache=cache,
        layer_id=0,
        is_mla=is_mla,
        block_m=32,
        block_n=32,
        prefill=False,
    )

    torch.testing.assert_close(out_no_cache, out_with_cache, atol=2e-2, rtol=0)

    out_no_cache_unpacked = unpack_from_kernel(out_no_cache, q_meta, H=H)
    out_with_cache_unpacked = unpack_from_kernel(out_with_cache, q_meta, H=H)
    ref = reference_attention(q, k, v, lengths, scale, is_mla=is_mla)
    torch.testing.assert_close(out_no_cache_unpacked, ref, atol=2e-2, rtol=0)
    torch.testing.assert_close(out_with_cache_unpacked, ref, atol=2e-2, rtol=0)


@pytest.mark.parametrize("is_mla", [False, True])
def test_mlm_compressed_prefill_with_cache_stores_entry(is_mla):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    torch.manual_seed(13)
    B, H, N, D = 2, 2, 20, 64
    lengths = torch.tensor([20, 7], device=DEVICE, dtype=torch.int32)

    q = torch.randn((B, H, N, D), device=DEVICE, dtype=torch.float16)
    k = torch.randn((B, H, N, D), device=DEVICE, dtype=torch.float16)
    v = torch.randn((B, H, N, D), device=DEVICE, dtype=torch.float16)
    q_meta = build_pack_metadata(lengths, N, block_n=32)

    cache = InferenceCache()
    packed_out = flash_attn_mlm_compressed(
        q,
        k,
        v,
        num_heads=H,
        scale=0.55,
        q_meta=q_meta,
        inference_cache=cache,
        layer_id=7,
        is_mla=is_mla,
        block_m=32,
        block_n=32,
        prefill=True,
    )

    stored = cache.get_kv_cache(
        7,
        is_mla=is_mla,
        num_heads=H,
        head_dim=D,
        dtype=q.dtype,
        device=q.device,
    )
    # Cached storage length is padded for descriptor-safe tail loads.
    expected_unpadded_total_q_len = int(lengths.sum().item())
    assert int(stored.cu_seqlens_kv[-1].item()) == expected_unpadded_total_q_len
    assert stored.total_context_len >= expected_unpadded_total_q_len
    if is_mla:
        assert stored.k_cache.shape[0] == stored.total_context_len
    else:
        assert stored.k_cache.shape[0] == H * stored.total_context_len
    assert stored.context_batch_size == B


@pytest.mark.parametrize("is_mla", [False, True])
def test_mlm_compressed_matches_reference(is_mla):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    torch.manual_seed(0)
    B, H, N, D = 2, 3, 48, 64
    lengths = torch.tensor([48, 37], device=DEVICE, dtype=torch.int32)
    scale = 0.7

    q = torch.randn((B, H, N, D), device=DEVICE, dtype=torch.float16)
    k = torch.randn((B, H, N, D), device=DEVICE, dtype=torch.float16)
    v = torch.randn((B, H, N, D), device=DEVICE, dtype=torch.float16)

    total_context_len = 0
    if is_mla:
        k_cache = torch.empty(
            (total_context_len, D), device=DEVICE, dtype=torch.float16
        )
    else:
        k_cache = torch.empty(
            (H * total_context_len, D), device=DEVICE, dtype=torch.float16
        )
    v_cache = torch.empty_like(k_cache)
    cu_seqlens_kv = torch.zeros(B + 1, device=DEVICE, dtype=torch.int32)
    cache = InferenceCache()
    cache.prefill_kv_cache(
        layer_id=0,
        k_cache=k_cache,
        v_cache=v_cache,
        total_context_len=total_context_len,
        cu_seqlens_kv=cu_seqlens_kv,
        context_batch_size=B,
        is_mla=is_mla,
        num_heads=H,
        head_dim=D,
    )

    q_meta = build_pack_metadata(lengths, N, block_n=32)
    packed_out = flash_attn_mlm_compressed(
        q,
        k,
        v,
        num_heads=H,
        scale=scale,
        q_meta=q_meta,
        inference_cache=cache,
        layer_id=0,
        is_mla=is_mla,
        block_m=32,
        block_n=32,
    )
    out = unpack_from_kernel(packed_out, q_meta, H=H)

    ref = reference_attention(q, k, v, lengths, scale, is_mla=is_mla)
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=0)


@pytest.mark.parametrize("is_mla", [False, True])
def test_mlm_compressed_prev_subseq_attention_matches_reference(is_mla):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    torch.manual_seed(2)
    B, H, N, D = 3, 2, 32, 64
    lengths = torch.tensor([9, 7, 5], device=DEVICE, dtype=torch.int32)
    scale = 0.65

    q = torch.randn((B, H, N, D), device=DEVICE, dtype=torch.float16)
    k = torch.randn((B, H, N, D), device=DEVICE, dtype=torch.float16)
    v = torch.randn((B, H, N, D), device=DEVICE, dtype=torch.float16)

    total_context_len = 0
    if is_mla:
        k_cache = torch.empty(
            (total_context_len, D), device=DEVICE, dtype=torch.float16
        )
    else:
        k_cache = torch.empty(
            (H * total_context_len, D),
            device=DEVICE,
            dtype=torch.float16,
        )
    v_cache = torch.empty_like(k_cache)
    cu_seqlens_kv = torch.zeros(B + 1, device=DEVICE, dtype=torch.int32)

    cache = InferenceCache()
    cache.prefill_kv_cache(
        layer_id=0,
        k_cache=k_cache,
        v_cache=v_cache,
        total_context_len=total_context_len,
        cu_seqlens_kv=cu_seqlens_kv,
        context_batch_size=B,
        is_mla=is_mla,
        num_heads=H,
        head_dim=D,
    )

    q_meta = build_pack_metadata(lengths, N, block_n=32)
    packed_out = flash_attn_mlm_compressed(
        q,
        k,
        v,
        num_heads=H,
        scale=scale,
        q_meta=q_meta,
        inference_cache=cache,
        layer_id=0,
        is_mla=is_mla,
        block_m=32,
        block_n=32,
        causal_query_seq_attn=True,
    )
    out = unpack_from_kernel(packed_out, q_meta, H=H)

    ref = _reference_varlen_attention_with_cache(
        q,
        k,
        v,
        lengths,
        k_cache,
        v_cache,
        cu_seqlens_kv,
        scale,
        is_mla,
        causal_query_seq_attn=True,
    )
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=0)


@pytest.mark.parametrize("is_mla", [False, True])
def test_mlm_compressed_matches_reference_with_cache(is_mla):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    torch.manual_seed(1)
    B, H, N, D = 2, 3, 48, 64
    lengths = torch.tensor([48, 37], device=DEVICE, dtype=torch.int32)
    context_lengths = torch.tensor([13, 9], device=DEVICE, dtype=torch.int32)
    scale = 0.5

    q = torch.randn((B, H, N, D), device=DEVICE, dtype=torch.float16)
    k = torch.randn((B, H, N, D), device=DEVICE, dtype=torch.float16)
    v = torch.randn((B, H, N, D), device=DEVICE, dtype=torch.float16)

    cu_seqlens_kv = torch.zeros(B + 1, device=DEVICE, dtype=torch.int32)
    cu_seqlens_kv[1:] = context_lengths.cumsum(0)
    total_context_len = int(cu_seqlens_kv[-1].item())

    if is_mla:
        k_cache = torch.randn(
            (total_context_len, D), device=DEVICE, dtype=torch.float16
        )
        v_cache = torch.randn_like(k_cache)
    else:
        k_cache = torch.randn(
            (H * total_context_len, D), device=DEVICE, dtype=torch.float16
        )
        v_cache = torch.randn_like(k_cache)
    cache = InferenceCache()
    cache.prefill_kv_cache(
        layer_id=0,
        k_cache=k_cache,
        v_cache=v_cache,
        total_context_len=total_context_len,
        cu_seqlens_kv=cu_seqlens_kv,
        context_batch_size=B,
        is_mla=is_mla,
        num_heads=H,
        head_dim=D,
    )

    q_meta = build_pack_metadata(lengths, N, block_n=32)
    packed_out = flash_attn_mlm_compressed(
        q,
        k,
        v,
        num_heads=H,
        scale=scale,
        q_meta=q_meta,
        inference_cache=cache,
        layer_id=0,
        is_mla=is_mla,
        block_m=32,
        block_n=32,
    )
    out = unpack_from_kernel(packed_out, q_meta, H=H)

    ref = _reference_varlen_attention_with_cache(
        q,
        k,
        v,
        lengths,
        k_cache,
        v_cache,
        cu_seqlens_kv,
        scale,
        is_mla,
    )
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=0)


# @pytest.mark.parametrize("Z", [1, 4])
# @pytest.mark.parametrize("H", [2, 48])
# @pytest.mark.parametrize("N_CTX", [128, 1024, (2 if is_hip() else 4) * 1024])
# @pytest.mark.parametrize("HEAD_DIM", [64, 128])
# @pytest.mark.parametrize("causal", [False, True])
# @pytest.mark.parametrize(
#     "warp_specialize", [False, True] if is_blackwell() else [False]
# )
# @pytest.mark.parametrize("mode", ["fwd", "bwd"])
# @pytest.mark.parametrize(
#     "provider", ["triton-fp16"] + (["triton-fp8"] if TORCH_HAS_FP8 else [])
# )
# def test_op(
#     Z, H, N_CTX, HEAD_DIM, causal, warp_specialize, mode, provider, dtype=torch.float16
# ):
#     if attention is None:
#         pytest.skip(
#             "python/kernel.py reference path unavailable; skipping legacy attention test"
#         )
#     if mode == "fwd" and "fp16" in provider:
#         pytest.skip("Avoid running the forward computation twice.")
#     if mode == "bwd" and "fp8" in provider:
#         pytest.skip("Backward pass with FP8 is not supported.")
#     torch.manual_seed(20)
#     q = (
#         torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE)
#         .normal_(mean=0.0, std=0.5)
#         .requires_grad_()
#     )
#     k = (
#         torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE)
#         .normal_(mean=0.0, std=0.5)
#         .requires_grad_()
#     )
#     v = (
#         torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE)
#         .normal_(mean=0.0, std=0.5)
#         .requires_grad_()
#     )
#     sm_scale = 0.5
#     # reference implementation
#     ref_dtype = dtype
#     if mode == "fwd" and "fp8" in provider:
#         ref_dtype = torch.float32
#     q = q.to(ref_dtype)
#     k = k.to(ref_dtype)
#     v = v.to(ref_dtype)
#     M = torch.tril(torch.ones((N_CTX, N_CTX), device=DEVICE))
#     p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
#     if causal:
#         p[:, :, M == 0] = float("-inf")
#     p = torch.softmax(p.float(), dim=-1)
#     p = p.to(ref_dtype)
#     # p = torch.exp(p)
#     ref_out = torch.matmul(p, v).half()
#     if mode == "bwd":
#         dout = torch.randn_like(q)
#         ref_out.backward(dout)
#         ref_dv, v.grad = v.grad.clone(), None
#         ref_dk, k.grad = k.grad.clone(), None
#         ref_dq, q.grad = q.grad.clone(), None
#     # triton implementation
#     if mode == "fwd" and "fp8" in provider:
#         q = q.to(torch.float8_e5m2)
#         k = k.to(torch.float8_e5m2)
#         v = v.permute(0, 1, 3, 2).contiguous()
#         v = v.permute(0, 1, 3, 2)
#         v = v.to(torch.float8_e5m2)
#     tri_out = attention(q, k, v, causal, sm_scale, warp_specialize).half()
#     if mode == "fwd":
#         atol = 3 if "fp8" in provider else 1e-2
#         torch.testing.assert_close(tri_out, ref_out, atol=atol, rtol=0)
#         return
#     tri_out.backward(dout)
#     tri_dv, v.grad = v.grad.clone(), None
#     tri_dk, k.grad = k.grad.clone(), None
#     tri_dq, q.grad = q.grad.clone(), None
#     # compare
#     torch.testing.assert_close(tri_out, ref_out, atol=1e-2, rtol=0)
#     rtol = 0.0
#     # Relative tolerance workaround for known hardware limitation of CDNA2 GPU.
#     # For details see https://pytorch.org/docs/stable/notes/numerical_accuracy.html#reduced-precision-fp16-and-bf16-gemms-and-convolutions-on-amd-instinct-mi200-devices
#     if (
#         torch.version.hip is not None
#         and triton.runtime.driver.active.get_current_target().arch == "gfx90a"
#     ):
#         rtol = 1e-2
#     torch.testing.assert_close(tri_dv, ref_dv, atol=1e-2, rtol=rtol)
#     torch.testing.assert_close(tri_dk, ref_dk, atol=1e-2, rtol=rtol)
#     torch.testing.assert_close(tri_dq, ref_dq, atol=1e-2, rtol=rtol)


# try:
#     from flash_attn.flash_attn_interface import (
#         flash_attn_qkvpacked_func as flash_attn_func,
#     )

#     HAS_FLASH = True
# except BaseException:
#     HAS_FLASH = False

# TORCH_HAS_FP8 = hasattr(torch, "float8_e5m2")
# BATCH, N_HEADS = 4, 32
# # vary seq length for fixed head and batch=4
# configs = []
# for HEAD_DIM in [64, 128]:
#     for mode in ["fwd", "bwd"]:
#         for causal in [True, False]:
#             # Enable warpspec for causal fwd on Hopper
#             enable_ws = mode == "fwd" and (
#                 is_blackwell() or (is_hopper() and not causal)
#             )
#             for warp_specialize in [False, True] if enable_ws else [False]:
#                 configs.append(
#                     triton.testing.Benchmark(
#                         x_names=["N_CTX"],
#                         x_vals=[2**i for i in range(10, 15)],
#                         line_arg="provider",
#                         line_vals=["triton-fp16"]
#                         + (["triton-fp8"] if TORCH_HAS_FP8 else [])
#                         + (["flash"] if HAS_FLASH else []),
#                         line_names=["Triton [FP16]"]
#                         + (["Triton [FP8]"] if TORCH_HAS_FP8 else [])
#                         + (["Flash-2"] if HAS_FLASH else []),
#                         styles=[("red", "-"), ("blue", "-"), ("green", "-")],
#                         ylabel="TFLOPS",
#                         plot_name=f"fused-attention-batch{BATCH}-head{N_HEADS}-d{HEAD_DIM}-{mode}-causal={causal}-warp_specialize={warp_specialize}",
#                         args={
#                             "H": N_HEADS,
#                             "BATCH": BATCH,
#                             "HEAD_DIM": HEAD_DIM,
#                             "mode": mode,
#                             "causal": causal,
#                             "warp_specialize": warp_specialize,
#                         },
#                     )
#                 )


# @triton.testing.perf_report(configs)
# def bench_flash_attention(
#     BATCH, H, N_CTX, HEAD_DIM, causal, warp_specialize, mode, provider, device=DEVICE
# ):
#     if attention is None and "triton" in provider:
#         pytest.skip(
#             "python/kernel.py reference path unavailable; skipping Triton benchmark"
#         )
#     assert mode in ["fwd", "bwd"]
#     dtype = torch.float16
#     if "triton" in provider:
#         q = torch.randn(
#             (BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True
#         )
#         k = torch.randn(
#             (BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True
#         )
#         v = torch.randn(
#             (BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True
#         )
#         if mode == "fwd" and "fp8" in provider:
#             q = q.to(torch.float8_e5m2)
#             k = k.to(torch.float8_e5m2)
#             v = v.permute(0, 1, 3, 2).contiguous()
#             v = v.permute(0, 1, 3, 2)
#             v = v.to(torch.float8_e5m2)
#         sm_scale = 1.3
#         fn = lambda: attention(q, k, v, causal, sm_scale, warp_specialize)
#         if mode == "bwd":
#             o = fn()
#             do = torch.randn_like(o)
#             fn = lambda: o.backward(do, retain_graph=True)
#         ms = triton.testing.do_bench(fn)

#     if provider == "flash":
#         qkv = torch.randn(
#             (BATCH, N_CTX, 3, H, HEAD_DIM),
#             dtype=dtype,
#             device=device,
#             requires_grad=True,
#         )
#         fn = lambda: flash_attn_func(qkv, causal=causal)
#         if mode == "bwd":
#             o = fn()
#             do = torch.randn_like(o)
#             fn = lambda: o.backward(do, retain_graph=True)
#         ms = triton.testing.do_bench(fn)
#     flops_per_matmul = 2.0 * BATCH * H * N_CTX * N_CTX * HEAD_DIM
#     total_flops = 2 * flops_per_matmul
#     if causal:
#         total_flops *= 0.5
#     if mode == "bwd":
#         total_flops *= 2.5  # 2.0(bwd) + 0.5(recompute)
#     return total_flops * 1e-12 / (ms * 1e-3)


@pytest.mark.parametrize("is_mla", [False, True])
def test_mlm_precompressed_matches_compressed_no_cache(is_mla):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    torch.manual_seed(50)
    B, H, N, D = 2, 3, 48, 64
    lengths = torch.tensor([48, 37], device=DEVICE, dtype=torch.int32)
    scale = 0.7
    block_m, block_n = 32, 32

    q = torch.randn((B, H, N, D), device=DEVICE, dtype=torch.float16)
    k = torch.randn((B, H, N, D), device=DEVICE, dtype=torch.float16)
    v = k if is_mla else torch.randn((B, H, N, D), device=DEVICE, dtype=torch.float16)

    total_context_len = 0
    k_cache = torch.empty(
        (total_context_len, D) if is_mla else (H * total_context_len, D),
        device=DEVICE,
        dtype=torch.float16,
    )
    v_cache = torch.empty_like(k_cache)
    cu_seqlens_kv = torch.zeros(B + 1, device=DEVICE, dtype=torch.int32)
    cache = InferenceCache()
    cache.prefill_kv_cache(
        layer_id=0,
        k_cache=k_cache,
        v_cache=v_cache,
        total_context_len=total_context_len,
        cu_seqlens_kv=cu_seqlens_kv,
        context_batch_size=B,
        is_mla=is_mla,
        num_heads=H,
        head_dim=D,
    )

    q_meta = build_pack_metadata(lengths, N, block_n=block_n)

    # Reference: flash_attn_mlm_compressed with padded inputs
    ref_out = flash_attn_mlm_compressed(
        q,
        k,
        v,
        num_heads=H,
        scale=scale,
        q_meta=q_meta,
        inference_cache=cache,
        layer_id=0,
        is_mla=is_mla,
        block_m=block_m,
        block_n=block_n,
    )

    # Pre-pack q, k, v manually and call precompressed
    q_packed = pack_for_kernel(q, q_meta, flatten_for_kernel=True)
    if is_mla:
        k_packed = pack_for_kernel(
            k[:, :1, :, :], q_meta, flatten_for_kernel=False
        ).squeeze(0)
        v_packed = k_packed
    else:
        k_packed = pack_for_kernel(k, q_meta, flatten_for_kernel=True)
        v_packed = pack_for_kernel(v, q_meta, flatten_for_kernel=True)

    cache2 = InferenceCache()
    cache2.prefill_kv_cache(
        layer_id=0,
        k_cache=k_cache,
        v_cache=v_cache,
        total_context_len=total_context_len,
        cu_seqlens_kv=cu_seqlens_kv,
        context_batch_size=B,
        is_mla=is_mla,
        num_heads=H,
        head_dim=D,
    )

    total_q_len = q_packed.shape[0] // H
    pre_out = flash_attn_mlm_precompressed(
        q_packed,
        k_packed,
        v_packed,
        num_heads=H,
        total_q_len=total_q_len,
        cu_seqlens_q=q_meta.cu_seqlens,
        batch_ids_q=q_meta.batch_ids_q,
        q_tile_starts_q=q_meta.q_tile_starts_q,
        scale=scale,
        inference_cache=cache2,
        layer_id=0,
        is_mla=is_mla,
        block_m=block_m,
        block_n=block_n,
    )

    torch.testing.assert_close(pre_out, ref_out, atol=1e-5, rtol=0)


@pytest.mark.parametrize("is_mla", [False, True])
def test_mlm_precompressed_prefill_stores_cache_entry(is_mla):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    torch.manual_seed(52)
    B, H, N, D = 2, 2, 20, 64
    lengths = torch.tensor([20, 7], device=DEVICE, dtype=torch.int32)
    block_m, block_n = 32, 32

    q = torch.randn((B, H, N, D), device=DEVICE, dtype=torch.float16)
    k = torch.randn((B, H, N, D), device=DEVICE, dtype=torch.float16)
    v = k if is_mla else torch.randn((B, H, N, D), device=DEVICE, dtype=torch.float16)

    q_meta = build_pack_metadata(lengths, N, block_n=block_n)

    q_packed = pack_for_kernel(q, q_meta, flatten_for_kernel=True)
    if is_mla:
        k_packed = pack_for_kernel(
            k[:, :1, :, :], q_meta, flatten_for_kernel=False
        ).squeeze(0)
        v_packed = k_packed
    else:
        k_packed = pack_for_kernel(k, q_meta, flatten_for_kernel=True)
        v_packed = pack_for_kernel(v, q_meta, flatten_for_kernel=True)

    total_q_len = q_packed.shape[0] // H

    cache = InferenceCache()
    flash_attn_mlm_precompressed(
        q_packed,
        k_packed,
        v_packed,
        num_heads=H,
        total_q_len=total_q_len,
        cu_seqlens_q=q_meta.cu_seqlens,
        batch_ids_q=q_meta.batch_ids_q,
        q_tile_starts_q=q_meta.q_tile_starts_q,
        scale=0.55,
        inference_cache=cache,
        layer_id=7,
        is_mla=is_mla,
        block_m=block_m,
        block_n=block_n,
        prefill=True,
    )

    stored = cache.get_kv_cache(
        7,
        is_mla=is_mla,
        num_heads=H,
        head_dim=D,
        dtype=q.dtype,
        device=q.device,
    )
    expected_unpadded_total_q_len = int(lengths.sum().item())
    assert int(stored.cu_seqlens_kv[-1].item()) == expected_unpadded_total_q_len
    assert stored.total_context_len >= expected_unpadded_total_q_len
    if is_mla:
        assert stored.k_cache.shape[0] == stored.total_context_len
    else:
        assert stored.k_cache.shape[0] == H * stored.total_context_len
    assert stored.context_batch_size == B


@pytest.mark.parametrize("is_mla", [False, True])
def test_mlm_precompressed_matches_compressed_with_cache(is_mla):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    torch.manual_seed(51)
    B, H, N, D = 2, 3, 48, 64
    lengths = torch.tensor([48, 37], device=DEVICE, dtype=torch.int32)
    context_lengths = torch.tensor([13, 9], device=DEVICE, dtype=torch.int32)
    scale = 0.5
    block_m, block_n = 32, 32

    q = torch.randn((B, H, N, D), device=DEVICE, dtype=torch.float16)
    k = torch.randn((B, H, N, D), device=DEVICE, dtype=torch.float16)
    v = k if is_mla else torch.randn((B, H, N, D), device=DEVICE, dtype=torch.float16)

    cu_seqlens_kv = torch.zeros(B + 1, device=DEVICE, dtype=torch.int32)
    cu_seqlens_kv[1:] = context_lengths.cumsum(0)
    total_context_len = int(cu_seqlens_kv[-1].item())

    if is_mla:
        k_cache = torch.randn(
            (total_context_len, D), device=DEVICE, dtype=torch.float16
        )
    else:
        k_cache = torch.randn(
            (H * total_context_len, D), device=DEVICE, dtype=torch.float16
        )
    v_cache = torch.randn_like(k_cache)

    def _make_cache():
        c = InferenceCache()
        c.prefill_kv_cache(
            layer_id=0,
            k_cache=k_cache,
            v_cache=v_cache,
            total_context_len=total_context_len,
            cu_seqlens_kv=cu_seqlens_kv,
            context_batch_size=B,
            is_mla=is_mla,
            num_heads=H,
            head_dim=D,
        )
        return c

    q_meta = build_pack_metadata(lengths, N, block_n=block_n)

    ref_out = flash_attn_mlm_compressed(
        q,
        k,
        v,
        num_heads=H,
        scale=scale,
        q_meta=q_meta,
        inference_cache=_make_cache(),
        layer_id=0,
        is_mla=is_mla,
        block_m=block_m,
        block_n=block_n,
    )

    q_packed = pack_for_kernel(q, q_meta, flatten_for_kernel=True)
    if is_mla:
        k_packed = pack_for_kernel(
            k[:, :1, :, :], q_meta, flatten_for_kernel=False
        ).squeeze(0)
        v_packed = k_packed
    else:
        k_packed = pack_for_kernel(k, q_meta, flatten_for_kernel=True)
        v_packed = pack_for_kernel(v, q_meta, flatten_for_kernel=True)

    total_q_len = q_packed.shape[0] // H
    pre_out = flash_attn_mlm_precompressed(
        q_packed,
        k_packed,
        v_packed,
        num_heads=H,
        total_q_len=total_q_len,
        cu_seqlens_q=q_meta.cu_seqlens,
        batch_ids_q=q_meta.batch_ids_q,
        q_tile_starts_q=q_meta.q_tile_starts_q,
        scale=scale,
        inference_cache=_make_cache(),
        layer_id=0,
        is_mla=is_mla,
        block_m=block_m,
        block_n=block_n,
    )

    torch.testing.assert_close(pre_out, ref_out, atol=1e-5, rtol=0)


if __name__ == "__main__":
    # only works on post-Ampere GPUs right now
    bench_flash_attention.run(save_path=".", print_data=True)
