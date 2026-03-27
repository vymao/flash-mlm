import triton
import triton.language as tl

from flash_mlm.kernel_utils import (
    _maybe_make_tensor_desc,
    _maybe_write_tensor_descriptor,
)


@triton.jit
def _mlm_inner_attention(
    Q_block,
    q_start_offsets,
    offset_y,
    desc_k,
    desc_v,
    l_i,
    m_i,
    o_i,
    scale,
    q_valid_start,
    q_valid_end,
    K_START,
    K_END,
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    warp_specialize: tl.constexpr,
    IS_MLA: tl.constexpr,
):
    scale_log2 = scale * 1.4426950408889634  # 1 / ln(2)
    for block_start in tl.range(
        K_START, K_END, BLOCK_M, warp_specialize=warp_specialize
    ):
        kv_start_offsets = block_start + tl.arange(0, BLOCK_M)

        K_block = desc_k.load([offset_y + block_start, 0])
        if IS_MLA:
            V_block = K_block
        else:
            V_block = desc_v.load([offset_y + block_start, 0])

        attn = tl.dot(Q_block, tl.trans(K_block)) * scale_log2

        kv_valid = kv_start_offsets < K_END
        q_valid = (q_start_offsets >= q_valid_start) & (q_start_offsets < q_valid_end)
        attn_mask = q_valid[:, None] & kv_valid[None, :]
        attn = tl.where(attn_mask, attn, float("-inf"))

        max_val_block = tl.max(attn, axis=1)
        m_new = tl.maximum(m_i, max_val_block)

        corrected_max = tl.math.exp2(m_i - m_new)
        adjusted_attn = tl.math.exp2(attn - m_new[:, None])
        l_i = l_i * corrected_max + tl.sum(adjusted_attn, axis=1)

        o_i = o_i * corrected_max[:, None] + tl.dot(
            adjusted_attn.to(V_block.dtype), V_block
        )

        m_i = m_new

    return o_i, l_i, m_i


@triton.jit
def _mlm_main_kernel_impl(
    desc_q,
    desc_k,
    desc_v,
    desc_o,
    desc_k_cache,
    desc_v_cache,
    batch_size,
    context_batch_size,
    num_heads,
    context_len,
    scale,
    seq_len,
    is_mla: tl.constexpr,
    prefill: tl.constexpr,
    force_desc_input: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    warp_specialize: tl.constexpr,
):
    out_ptr = desc_o
    start_block_q = tl.program_id(0)
    off_hz = tl.program_id(1)
    batch_idx = off_hz // num_heads
    head_idx = off_hz % num_heads

    y_dim = batch_size * num_heads * seq_len
    if not force_desc_input:
        desc_q = _maybe_make_tensor_desc(
            desc_q,
            shape=[y_dim, HEAD_DIM],
            strides=[HEAD_DIM, 1],
            block_shape=[BLOCK_N, HEAD_DIM],
        )
        desc_k = _maybe_make_tensor_desc(
            desc_k,
            shape=[y_dim, HEAD_DIM],
            strides=[HEAD_DIM, 1],
            block_shape=[BLOCK_M, HEAD_DIM],
        )
        if is_mla:
            desc_v = desc_k
        else:
            desc_v = _maybe_make_tensor_desc(
                desc_v,
                shape=[y_dim, HEAD_DIM],
                strides=[HEAD_DIM, 1],
                block_shape=[BLOCK_M, HEAD_DIM],
            )
        desc_o = _maybe_make_tensor_desc(
            desc_o,
            shape=[y_dim, HEAD_DIM],
            strides=[HEAD_DIM, 1],
            block_shape=[BLOCK_N, HEAD_DIM],
        )

    start_block_q = start_block_q * BLOCK_N
    offset_y = batch_idx * (seq_len * num_heads) + head_idx * seq_len
    if start_block_q >= seq_len:
        return

    q_start_offsets = start_block_q + tl.arange(0, BLOCK_N)
    q_valid = q_start_offsets < seq_len
    Q_block = desc_q.load([offset_y + start_block_q, 0])

    multi_batch_context = context_batch_size > 1
    if is_mla:
        context_batch_offset = multi_batch_context * (batch_idx * context_len)
        context_head_offset = 0
    else:
        context_batch_offset = multi_batch_context * (
            batch_idx * num_heads * context_len
        )
        context_head_offset = head_idx * context_len
    offset_y_context = context_batch_offset + context_head_offset

    m_i = tl.full([BLOCK_N], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_N], dtype=tl.float32)
    o_i = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)

    if not prefill:
        if is_mla:
            y_dim_context = context_batch_size * context_len
        else:
            y_dim_context = context_batch_size * num_heads * context_len
        if not force_desc_input:
            desc_k_cache = _maybe_make_tensor_desc(
                desc_k_cache,
                shape=[y_dim_context, HEAD_DIM],
                strides=[HEAD_DIM, 1],
                block_shape=[BLOCK_M, HEAD_DIM],
            )
            if is_mla:
                desc_v_cache = desc_k_cache
            else:
                desc_v_cache = _maybe_make_tensor_desc(
                    desc_v_cache,
                    shape=[y_dim_context, HEAD_DIM],
                    strides=[HEAD_DIM, 1],
                    block_shape=[BLOCK_M, HEAD_DIM],
                )
        o_i, l_i, m_i = _mlm_inner_attention(
            Q_block,
            q_start_offsets,
            offset_y_context,
            desc_k_cache,
            desc_v_cache,
            l_i,
            m_i,
            o_i,
            scale,
            0,
            seq_len,
            0,
            context_len,
            BLOCK_M,
            HEAD_DIM,
            warp_specialize,
            is_mla,
        )

    o_i, l_i, m_i = _mlm_inner_attention(
        Q_block,
        q_start_offsets,
        offset_y,
        desc_k,
        desc_v,
        l_i,
        m_i,
        o_i,
        scale,
        0,
        seq_len,
        0,
        seq_len,
        BLOCK_M,
        HEAD_DIM,
        warp_specialize,
        is_mla,
    )

    l_safe = tl.where(l_i > 0, l_i, 1.0)
    o_i = o_i / l_safe[:, None]
    _maybe_write_tensor_descriptor(
        out_ptr,
        offset_y + start_block_q,
        o_i,
        q_valid,
        HEAD_DIM,
    )


@triton.jit
def _mlm_compressed_kernel_impl(
    desc_q,
    desc_k,
    desc_v,
    desc_o,
    desc_k_cache,
    desc_v_cache,
    context_batch_size,
    num_heads,
    total_context_len,
    scale,
    total_q_len,
    batch_ids_q,
    q_tile_starts_q,
    cu_seqlens_q,
    cu_seqlens_kv,
    is_mla: tl.constexpr,
    prefill: tl.constexpr,
    causal_query_seq_attn: tl.constexpr,
    force_desc_input: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    warp_specialize: tl.constexpr,
):
    out_ptr = desc_o
    tile_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    batch_idx = tl.load(batch_ids_q + tile_idx)
    start_block_q = tl.load(q_tile_starts_q + tile_idx)

    y_dim_qo = num_heads * total_q_len
    if is_mla:
        y_dim_kv_main = total_q_len
    else:
        y_dim_kv_main = num_heads * total_q_len
    if not force_desc_input:
        desc_q = _maybe_make_tensor_desc(
            desc_q,
            shape=[y_dim_qo, HEAD_DIM],
            strides=[HEAD_DIM, 1],
            block_shape=[BLOCK_N, HEAD_DIM],
        )
        desc_k = _maybe_make_tensor_desc(
            desc_k,
            shape=[y_dim_kv_main, HEAD_DIM],
            strides=[HEAD_DIM, 1],
            block_shape=[BLOCK_M, HEAD_DIM],
        )
        if is_mla:
            desc_v = desc_k
        else:
            desc_v = _maybe_make_tensor_desc(
                desc_v,
                shape=[y_dim_kv_main, HEAD_DIM],
                strides=[HEAD_DIM, 1],
                block_shape=[BLOCK_M, HEAD_DIM],
            )

    q_head_offset = head_idx * total_q_len
    q_tile_block_offset = start_block_q + q_head_offset

    q_start = tl.load(cu_seqlens_q + batch_idx)
    q_end = tl.load(cu_seqlens_q + batch_idx + 1)
    q_start_offsets = start_block_q + tl.arange(0, BLOCK_N)
    q_valid_start = q_start
    q_valid_end = q_end
    if causal_query_seq_attn:
        kv_main_k_start = 0
        kv_main_k_end = q_end
    else:
        kv_main_k_start = q_start
        kv_main_k_end = q_end

    Q_block = desc_q.load([q_tile_block_offset, 0])

    if is_mla:
        context_head_offset = 0
    else:
        context_head_offset = head_idx * total_context_len

    multi_batch_context = context_batch_size > 1
    context_batch_offset = multi_batch_context * batch_idx

    kv_context_start = tl.load(cu_seqlens_kv + context_batch_offset)
    kv_context_end = tl.load(cu_seqlens_kv + context_batch_offset + 1)
    kv_context_len = kv_context_end - kv_context_start

    kv_context_start_offset = context_head_offset
    if is_mla:
        kv_main_start_offset = 0
    else:
        kv_main_start_offset = q_head_offset

    m_i = tl.full([BLOCK_N], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_N], dtype=tl.float32)
    o_i = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)

    if not prefill:
        if is_mla:
            y_dim_context = total_context_len
        else:
            y_dim_context = num_heads * total_context_len
        if not force_desc_input:
            desc_k_cache = _maybe_make_tensor_desc(
                desc_k_cache,
                shape=[y_dim_context, HEAD_DIM],
                strides=[HEAD_DIM, 1],
                block_shape=[BLOCK_M, HEAD_DIM],
            )
            if is_mla:
                desc_v_cache = desc_k_cache
            else:
                desc_v_cache = _maybe_make_tensor_desc(
                    desc_v_cache,
                    shape=[y_dim_context, HEAD_DIM],
                    strides=[HEAD_DIM, 1],
                    block_shape=[BLOCK_M, HEAD_DIM],
                )
        o_i, l_i, m_i = _mlm_inner_attention(
            Q_block,
            q_start_offsets,
            kv_context_start_offset,
            desc_k_cache,
            desc_v_cache,
            l_i,
            m_i,
            o_i,
            scale,
            q_valid_start,
            q_valid_end,
            kv_context_start,
            kv_context_end,
            BLOCK_M,
            HEAD_DIM,
            warp_specialize,
            is_mla,
        )

    # Main
    o_i, l_i, m_i = _mlm_inner_attention(
        Q_block,
        q_start_offsets,
        kv_main_start_offset,
        desc_k,
        desc_v,
        l_i,
        m_i,
        o_i,
        scale,
        q_valid_start,
        q_valid_end,
        kv_main_k_start,
        kv_main_k_end,
        BLOCK_M,
        HEAD_DIM,
        warp_specialize,
        is_mla,
    )

    l_safe = tl.where(l_i > 0, l_i, 1.0)
    o_i = o_i / l_safe[:, None]
    q_valid = (q_start_offsets >= q_start) & (q_start_offsets < q_end)
    _maybe_write_tensor_descriptor(
        out_ptr,
        q_tile_block_offset,
        o_i,
        q_valid,
        HEAD_DIM,
    )
