import triton
import triton.language as tl

from python.kernel_utils import _maybe_make_tensor_desc


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
    IS_MLA: tl.constexpr,
):
    for block_start in tl.range(K_START, K_END, BLOCK_M):
        kv_start_offsets = block_start + tl.arange(0, BLOCK_M)

        K_block = desc_k.load([offset_y + block_start, 0])
        if IS_MLA:
            V_block = K_block
        else:
            V_block = desc_v.load([offset_y + block_start, 0])

        # Compute attention
        attn = tl.dot(Q_block, tl.trans(K_block)) * scale

        # Apply masking for out of bounds positions
        kv_valid = kv_start_offsets < K_END
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
    is_mla: tl.constexpr,
    BLOCK_M: tl.constexpr,  # K/V
    BLOCK_N: tl.constexpr,  # Query
    HEAD_DIM: tl.constexpr,
):
    start_block_q = tl.program_id(0)
    off_hz = tl.program_id(1)
    batch_idx = off_hz // num_heads
    head_idx = off_hz % num_heads

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

    if is_mla:
        y_dim_context = context_batch_size * context_len
    else:
        y_dim_context = context_batch_size * num_heads * context_len
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

    q_start_offsets = start_block_q + tl.arange(0, BLOCK_N)
    Q_block = desc_q.load([offset_y + start_block_q, 0])

    # ============================================================
    # Step 2: Get parameters for KV block
    # ============================================================
    # Context offsets differ between standard per-head KV cache and MLA shared cache.
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

    # ============================================================
    # Step 3: Compute Attention
    # ============================================================
    # Initialize state across all context
    m_i = tl.full([BLOCK_N], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_N], dtype=tl.float32)
    o_i = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)

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
        seq_len,  # For query
        0,
        context_len,
        BLOCK_M,
        HEAD_DIM,
        is_mla,
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
        seq_len,  # For query
        0,
        seq_len,
        BLOCK_M,
        HEAD_DIM,
        is_mla,
    )

    l_safe = tl.where(l_i > 0, l_i, 1.0)
    o_i = o_i / l_safe[:, None]
    desc_o.store([offset_y + start_block_q, 0], o_i)


@triton.jit
def _mlm_compressed_kernel(
    desc_q,  # Assumes layout of (batch_size, num_heads, seq_len, head_dim)
    desc_k,
    desc_v,
    desc_o,
    desc_k_cache,
    desc_v_cache,
    context_batch_size,
    num_heads,
    total_context_len,  # Length of the packed context, across batches, per head
    scale,
    total_q_len,  # Length of the packed query, across batches, per head
    batch_ids_q,  # Per tile batch index
    cu_seqlens_q,  # Cumulative sequence lengths for query, per batch, per head
    cu_seqlens_kv,  # Cumulative sequence lengths for key/value, per batch, per head
    is_mla: tl.constexpr,
    BLOCK_M: tl.constexpr,  # K/V
    BLOCK_N: tl.constexpr,  # Query
    HEAD_DIM: tl.constexpr,
):
    start_block_q = tl.program_id(0)  # Concatenated sequence dimension (batch-wise)
    head_idx = tl.program_id(1)  # Head dimension

    ## Parameters
    # Load and get batch index and start offset for query
    batch_idx = tl.load(
        batch_ids_q + start_block_q
    )  # Indexes start_block_q into batch_ids_q

    y_dim_qo = num_heads * total_q_len
    if is_mla:
        y_dim_kv_main = total_q_len
    else:
        y_dim_kv_main = num_heads * total_q_len
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
    desc_v = _maybe_make_tensor_desc(
        desc_v,
        shape=[y_dim_kv_main, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_M, HEAD_DIM],
    )
    desc_o = _maybe_make_tensor_desc(
        desc_o,
        shape=[y_dim_qo, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_N, HEAD_DIM],
    )

    if is_mla:
        y_dim_context = total_context_len
    else:
        y_dim_context = num_heads * total_context_len
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
    # Load per-sequence token offset
    start_block_q = start_block_q * BLOCK_N
    q_head_offset = head_idx * total_q_len
    q_tile_block_offset = start_block_q + q_head_offset

    q_start = tl.load(cu_seqlens_q + batch_idx)
    q_end = tl.load(cu_seqlens_q + batch_idx + 1)
    q_len = q_end - q_start
    q_start_offsets = (start_block_q - q_start) + tl.arange(0, BLOCK_N)

    Q_block = desc_q.load([q_tile_block_offset, 0])

    # ============================================================
    # Step 2: Get parameters for KV block
    # ============================================================
    if is_mla:
        context_head_offset = 0
    else:
        context_head_offset = head_idx * total_context_len

    multi_batch_context = context_batch_size > 1
    context_batch_offset = multi_batch_context * batch_idx

    kv_context_start = tl.load(cu_seqlens_kv + context_batch_offset)
    kv_context_end = tl.load(cu_seqlens_kv + context_batch_offset + 1)
    kv_context_len = kv_context_end - kv_context_start

    kv_context_start_offset = kv_context_start + context_head_offset
    if is_mla:
        kv_main_start_offset = q_start
    else:
        kv_main_start_offset = q_start + q_head_offset

    # ============================================================
    # Step 3: Compute Attention
    # ============================================================
    m_i = tl.full([BLOCK_N], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_N], dtype=tl.float32)
    o_i = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)

    # Cache
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
        q_len,
        0,
        kv_context_len,
        BLOCK_M,
        HEAD_DIM,
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
        q_len,
        0,
        q_len,
        BLOCK_M,
        HEAD_DIM,
        is_mla,
    )

    l_safe = tl.where(l_i > 0, l_i, 1.0)
    o_i = o_i / l_safe[:, None]
    desc_o.store([q_tile_block_offset, 0], o_i)
