import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from flash_mlm.kernel_utils import is_hip
from flash_mlm.mlm_kernel_impl import (
    _mlm_compressed_kernel_impl,
    _mlm_main_kernel_impl,
)


if is_hip():
    _MLM_NUM_STAGES_OPTIONS = [1]
else:
    _MLM_NUM_STAGES_OPTIONS = [2, 3, 4]

_MLM_BLOCK_M_OPTIONS = [32, 64, 128]
_MLM_BLOCK_N_OPTIONS = [32, 64, 128]


def _set_desc_block_shape_if_needed(nargs, name, block_shape):
    desc = nargs[name]
    if isinstance(desc, TensorDescriptor):
        desc.block_shape = block_shape


def _mlm_main_host_descriptor_pre_hook(nargs):
    block_m = nargs["BLOCK_M"]
    block_n = nargs["BLOCK_N"]
    head_dim = nargs["HEAD_DIM"]
    _set_desc_block_shape_if_needed(nargs, "desc_q", [block_n, head_dim])
    _set_desc_block_shape_if_needed(nargs, "desc_k", [block_m, head_dim])
    _set_desc_block_shape_if_needed(nargs, "desc_v", [block_m, head_dim])
    _set_desc_block_shape_if_needed(nargs, "desc_o", [block_n, head_dim])
    _set_desc_block_shape_if_needed(nargs, "desc_k_cache", [block_m, head_dim])
    _set_desc_block_shape_if_needed(nargs, "desc_v_cache", [block_m, head_dim])


def _mlm_compressed_host_descriptor_pre_hook(nargs):
    block_m = nargs["BLOCK_M"]
    block_n = nargs["BLOCK_N"]
    head_dim = nargs["HEAD_DIM"]
    _set_desc_block_shape_if_needed(nargs, "desc_q", [block_n, head_dim])
    _set_desc_block_shape_if_needed(nargs, "desc_k", [block_m, head_dim])
    _set_desc_block_shape_if_needed(nargs, "desc_v", [block_m, head_dim])
    _set_desc_block_shape_if_needed(nargs, "desc_o", [block_n, head_dim])
    _set_desc_block_shape_if_needed(nargs, "desc_k_cache", [block_m, head_dim])
    _set_desc_block_shape_if_needed(nargs, "desc_v_cache", [block_m, head_dim])


def _mlm_prune_invalid_configs(configs, named_args, **kwargs):
    block_n = int(kwargs.get("BLOCK_N", 0))
    head_dim = int(kwargs.get("HEAD_DIM", 0))
    q_len = int(kwargs.get("seq_len", kwargs.get("total_q_len", 0)))
    kv_len = int(kwargs.get("context_len", kwargs.get("total_context_len", 0)))
    workload = max(q_len, kv_len)

    pruned = []
    for conf in configs:
        conf_block_m = int(conf.kwargs.get("BLOCK_M", kwargs.get("BLOCK_M", 0)))
        conf_block_n = int(conf.kwargs.get("BLOCK_N", block_n))
        tile_area = conf_block_m * conf_block_n

        if conf_block_n <= 32 and conf.num_warps > 4:
            continue
        if head_dim <= 32 and conf.num_warps > 4:
            continue
        if workload <= 64 and conf.num_warps > 2:
            continue
        if workload <= 128 and conf.num_stages >= 4:
            continue
        if conf_block_m >= 128 and head_dim >= 128 and conf.num_warps >= 8:
            continue
        if tile_area >= 16384 and head_dim >= 128 and conf.num_warps >= 8:
            continue
        pruned.append(conf)

    return pruned if pruned else configs[:1]


_MLM_COMPRESSED_AUTOTUNE_CONFIGS = [
    triton.Config(
        {},
        num_stages=s,
        num_warps=w,
        pre_hook=_mlm_compressed_host_descriptor_pre_hook,
    )
    for s in _MLM_NUM_STAGES_OPTIONS
    for w in [2, 4, 8]
]

_MLM_MAIN_AUTOTUNE_CONFIGS = [
    triton.Config(
        {},
        num_stages=s,
        num_warps=w,
        pre_hook=_mlm_main_host_descriptor_pre_hook,
    )
    for s in _MLM_NUM_STAGES_OPTIONS
    for w in [2, 4, 8]
]

_MLM_MAIN_AUTOTUNE_TILE_CONFIGS = [
    triton.Config(
        {"BLOCK_M": bm},
        num_stages=s,
        num_warps=w,
        pre_hook=_mlm_main_host_descriptor_pre_hook,
    )
    for bm in _MLM_BLOCK_M_OPTIONS
    for s in _MLM_NUM_STAGES_OPTIONS
    for w in [2, 4, 8]
]

_MLM_COMPRESSED_AUTOTUNE_TILE_CONFIGS = [
    triton.Config(
        {"BLOCK_M": bm},
        num_stages=s,
        num_warps=w,
        pre_hook=_mlm_compressed_host_descriptor_pre_hook,
    )
    for bm in _MLM_BLOCK_M_OPTIONS
    for s in _MLM_NUM_STAGES_OPTIONS
    for w in [2, 4, 8]
]


@triton.autotune(
    configs=_MLM_MAIN_AUTOTUNE_CONFIGS,
    key=["HEAD_DIM", "BLOCK_M", "BLOCK_N", "is_mla", "seq_len", "context_len"],
    prune_configs_by={"early_config_prune": _mlm_prune_invalid_configs},
)
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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    _mlm_main_kernel_impl(
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
        is_mla,
        BLOCK_M,
        BLOCK_N,
        HEAD_DIM,
    )


@triton.autotune(
    configs=_MLM_MAIN_AUTOTUNE_TILE_CONFIGS,
    key=["HEAD_DIM", "BLOCK_N", "is_mla", "seq_len", "context_len"],
    prune_configs_by={"early_config_prune": _mlm_prune_invalid_configs},
)
@triton.jit
def _mlm_main_kernel_auto_tile(
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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    _mlm_main_kernel_impl(
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
        is_mla,
        BLOCK_M,
        BLOCK_N,
        HEAD_DIM,
    )


@triton.autotune(
    configs=_MLM_COMPRESSED_AUTOTUNE_CONFIGS,
    key=[
        "HEAD_DIM",
        "BLOCK_M",
        "BLOCK_N",
        "is_mla",
        "total_q_len",
        "total_context_len",
    ],
    prune_configs_by={"early_config_prune": _mlm_prune_invalid_configs},
)
@triton.jit
def _mlm_compressed_kernel(
    desc_q,
    desc_k,
    desc_v,
    desc_o,
    desc_k_cache,
    desc_v_cache,
    context_batch_size,  # number of context batches represented by cu_seqlens_kv
    num_heads,  # attention heads
    total_context_len,  # packed context token count across context batches
    scale,  # softmax scaling factor
    total_q_len,  # packed (and possibly padded) query token count
    batch_ids_q,  # tile_idx -> batch_idx mapping for query tiles
    q_tile_starts_q,  # packed query start offset per tile
    cu_seqlens_q,  # cumulative packed query lengths [B+1]
    cu_seqlens_kv,  # cumulative packed context lengths [context_batch_size+1]
    is_mla: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    _mlm_compressed_kernel_impl(
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
        is_mla,
        BLOCK_M,
        BLOCK_N,
        HEAD_DIM,
    )


@triton.autotune(
    configs=_MLM_COMPRESSED_AUTOTUNE_TILE_CONFIGS,
    key=[
        "HEAD_DIM",
        "BLOCK_N",
        "is_mla",
        "total_q_len",
        "total_context_len",
    ],
    prune_configs_by={"early_config_prune": _mlm_prune_invalid_configs},
)
@triton.jit
def _mlm_compressed_kernel_auto_block_m(
    desc_q,
    desc_k,
    desc_v,
    desc_o,
    desc_k_cache,
    desc_v_cache,
    context_batch_size,  # number of context batches represented by cu_seqlens_kv
    num_heads,  # attention heads
    total_context_len,  # packed context token count across context batches
    scale,  # softmax scaling factor
    total_q_len,  # packed (and possibly padded) query token count
    batch_ids_q,  # tile_idx -> batch_idx mapping for query tiles
    q_tile_starts_q,  # packed query start offset per tile
    cu_seqlens_q,  # cumulative packed query lengths [B+1]
    cu_seqlens_kv,  # cumulative packed context lengths [context_batch_size+1]
    is_mla: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    _mlm_compressed_kernel_impl(
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
        is_mla,
        BLOCK_M,
        BLOCK_N,
        HEAD_DIM,
    )
