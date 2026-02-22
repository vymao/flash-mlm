import triton
import triton.language as tl

from python.utils import _maybe_make_tensor_desc


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
    N,
    K_START,
    K_END,
    BLOCK_M: tl.constexpr,  # K/V
    HEAD_DIM: tl.constexpr,
):
    for block_start in tl.range(K_START, K_END, BLOCK_M):
        kv_start_offsets = block_start + tl.arange(0, BLOCK_M)

        K_block = desc_k.load([offset_y + block_start, 0])
        V_block = desc_v.load([offset_y + block_start, 0])

        # Compute attention
        attn = tl.dot(Q_block, tl.trans(K_block)) * scale

        # Apply masking for out of bounds positions
        kv_valid = (kv_start_offsets < N) & (kv_start_offsets < K_END)
        attn_mask = (q_start_offsets[:, None] < N) & kv_valid[None, :]
        attn = tl.where(attn_mask, attn, float("-inf"))

        # Apply online softmax update
        max_val_block = tl.max(attn, axis=1)
        m_new = tl.maximum(m_i, max_val_block)

        corrected_max = tl.exp(m_i - m_new)
        adjusted_attn = tl.exp(attn - m_new[:, None])
        l_i = l_i * corrected_max + tl.sum(adjusted_attn, axis=1)

        o_i = o_i * corrected_max[:, None] + tl.dot(
            adjusted_attn.to(V_block.dtype), V_block
        )

        m_i = m_new

    return o_i, l_i, m_i


@triton.jit
def _mlm_main_kernel(
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
    cu_seqlens_q,
    cu_seqlens_kv,
    is_mla,
    BLOCK_M: tl.constexpr,  # K/V
    BLOCK_N: tl.constexpr,  # Query
    HEAD_DIM: tl.constexpr,
):
    start_block_q = tl.program_id(0)
    off_hz = tl.program_id(1)
    batch_idx = off_hz // num_heads
    head_idx = off_hz % num_heads

    is_mla_i = is_mla.to(tl.int32)
    not_mla_i = 1 - is_mla_i

    multi_batch_context = context_batch_size > 1

    y_dim = batch_size * num_heads * seq_len
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

    # Cache dimensions. Only one of the two will be non-zero.
    standard_y_dim_context = (context_batch_size * num_heads * context_len) * not_mla_i
    mla_y_dim_context = context_batch_size * context_len * is_mla_i
    y_dim_context = standard_y_dim_context + mla_y_dim_context
    desc_k_cache = _maybe_make_tensor_desc(
        desc_k_cache,
        shape=[y_dim_context, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_M, HEAD_DIM],
    )
    desc_v_cache = _maybe_make_tensor_desc(
        desc_v_cache,
        shape=[y_dim_context, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_M, HEAD_DIM],
    )

    # ============================================================
    # Step 1: Load Q block — stays in registers for the entire loop
    # ============================================================
    # Get the corresponding pointers for the block edges for query
    start_block_q = start_block_q * BLOCK_N
    offset_y = batch_idx * (seq_len * num_heads) + head_idx * seq_len

    # Context offset depends on if batch size is 1 or not. Only one of the two will be non-zero.
    standard_context_batch_offset = (
        multi_batch_context * (batch_idx * num_heads * context_len) * not_mla_i
    )
    mla_context_batch_offset = (
        multi_batch_context * (batch_idx * context_len) * is_mla_i
    )

    # Head offset needed for non-MLA.
    standard_context_head_offset = (head_idx * context_len) * not_mla_i

    offset_y_context = (
        standard_context_batch_offset
        + mla_context_batch_offset
        + standard_context_head_offset
    )

    q_start_offsets = start_block_q + tl.arange(0, BLOCK_N)
    Q_block = desc_q.load([offset_y + start_block_q, 0])

    # Initialize state across all context
    m_i = tl.full([BLOCK_N], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_N], dtype=tl.float32)
    o_i = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)

    # ============================================================
    # Step 2: Compute Attention
    # ============================================================
    # Cache
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
        context_len,
        0,
        context_len,
        BLOCK_M,
        HEAD_DIM,
    )

    # Main
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
        seq_len,
        0,
        seq_len,
        BLOCK_M,
        HEAD_DIM,
    )

    l_safe = tl.where(l_i > 0, l_i, 1.0)
    o_i = o_i / l_safe[:, None]
    desc_o.store([offset_y + start_block_q, 0], o_i)
