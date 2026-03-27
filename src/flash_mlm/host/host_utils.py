from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

try:
    from triton.tools.tensor_descriptor import TensorDescriptor
except ImportError:
    TensorDescriptor = None

from flash_mlm.host.cache import InferenceCache

if TYPE_CHECKING:
    from flash_mlm.host.cache import LayerKVEntry, PackingCache


@dataclass
class PackMetadata:
    """Reusable metadata for fast pack/unpack across layers."""

    B: int
    N: int
    lengths: torch.Tensor  # (B,) int tensor with active tokens per batch entry.
    cu_seqlens: torch.Tensor  # (B+1) int tensor with cumulative sequence lengths.
    token_indices: torch.Tensor  # (T,) Valid token indices.
    total_tokens: int  # Total number of valid tokens.
    batch_ids_block_n: int  # BLOCK_N
    batch_ids_q: torch.Tensor  # (num_q_tiles,) tile-to-batch mapping
    q_tile_starts_q: torch.Tensor  # (num_q_tiles,) packed query start offsets


@dataclass
class CacheContext:
    k_cache: torch.Tensor
    v_cache: torch.Tensor | None
    context_batch_size: int
    context_len: int
    total_context_len: int
    cu_seqlens_kv: torch.Tensor | None


def build_pack_metadata(lengths: torch.Tensor, N: int, block_n: int) -> PackMetadata:
    """Precompute reusable indexing for variable-length packed tensors.

    Args:
        lengths: (B,) int tensor with active tokens per batch entry.
        N: padded max sequence length.
        block_n: Q tile size used to precompute and store tile-to-batch ids
            for packed kernel launches.
    """
    if lengths.ndim != 1:
        raise ValueError("lengths must be rank-1 [B]")
    if N < 0:
        raise ValueError("N must be >= 0")
    if block_n < 1:
        raise ValueError("block_n must be >= 1")
    if torch.any(lengths < 0) or torch.any(lengths > N):
        raise ValueError("lengths values must be in [0, N]")

    lengths_i32 = lengths.to(dtype=torch.int32)
    B = int(lengths_i32.numel())
    device = lengths.device

    cu_seqlens = torch.zeros(B + 1, dtype=torch.int32, device=device)
    cu_seqlens[1:] = lengths_i32.cumsum(0)

    token_in_seq = torch.arange(N, device=device, dtype=torch.int32)[None, :]
    valid = token_in_seq < lengths_i32[:, None]
    flat_base = (torch.arange(B, device=device, dtype=torch.int32) * N)[:, None]
    token_indices = (flat_base + token_in_seq)[valid].to(dtype=torch.int64)

    q_tile_starts_q, batch_ids_q = _build_query_tile_map_from_cu_seqlens(
        cu_seqlens, block_n
    )

    return PackMetadata(
        B=B,
        N=N,
        lengths=lengths_i32,
        cu_seqlens=cu_seqlens,
        token_indices=token_indices,
        total_tokens=int(cu_seqlens[-1].item()),
        batch_ids_block_n=block_n,
        batch_ids_q=batch_ids_q,
        q_tile_starts_q=q_tile_starts_q,
    )


def _build_query_tile_map_from_cu_seqlens(cu_seqlens_q: torch.Tensor, block_n: int):
    """Build per-batch query tile starts and tile-to-batch mapping."""
    if cu_seqlens_q.ndim != 1:
        raise ValueError("cu_seqlens_q must be rank-1")
    if cu_seqlens_q.numel() < 2:
        raise ValueError("cu_seqlens_q must have at least 2 elements")
    if block_n < 1:
        raise ValueError("block_n must be >= 1")

    device = cu_seqlens_q.device
    lengths = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
    tiles_per_batch = torch.div(lengths + (block_n - 1), block_n, rounding_mode="floor")
    batch_ids = torch.repeat_interleave(
        torch.arange(lengths.numel(), device=device, dtype=torch.int32),
        tiles_per_batch.to(torch.int64),
    )

    total_tiles = int(tiles_per_batch.sum().item())
    if total_tiles == 0:
        empty = torch.empty((0,), device=device, dtype=torch.int32)
        return empty, empty

    tile_prefix = torch.cumsum(tiles_per_batch, dim=0) - tiles_per_batch
    intra_tile_idx = torch.arange(
        total_tiles, device=device, dtype=torch.int32
    ) - torch.repeat_interleave(
        tile_prefix,
        tiles_per_batch.to(torch.int64),
    )
    q_tile_starts_q = (
        cu_seqlens_q[:-1].index_select(0, batch_ids.to(torch.int64))
        + intra_tile_idx * block_n
    ).to(torch.int32)
    return q_tile_starts_q.contiguous(), batch_ids.contiguous()


def build_batch_ids_from_cu_seqlens(cu_seqlens_q: torch.Tensor, block_n: int):
    """Build tile-to-batch mapping for packed query tiles."""
    _, batch_ids = _build_query_tile_map_from_cu_seqlens(cu_seqlens_q, block_n)
    return batch_ids


def build_q_tile_starts_from_cu_seqlens(cu_seqlens_q: torch.Tensor, block_n: int):
    """Build packed query start offsets for each per-batch query tile."""
    q_tile_starts_q, _ = _build_query_tile_map_from_cu_seqlens(cu_seqlens_q, block_n)
    return q_tile_starts_q


def num_query_tiles_from_cu_seqlens(cu_seqlens_q: torch.Tensor, block_n: int) -> int:
    """Return number of per-batch query tiles for a given block_n."""
    q_tile_starts_q = build_q_tile_starts_from_cu_seqlens(cu_seqlens_q, block_n)
    return int(q_tile_starts_q.numel())


def num_query_tiles_from_meta(meta: PackMetadata) -> int:
    """Return number of precomputed per-batch query tiles stored in metadata."""
    return int(meta.q_tile_starts_q.numel())


def make_host_desc(x: torch.Tensor, rows: int, head_dim: int) -> TensorDescriptor:
    return TensorDescriptor(
        x,
        shape=[rows, head_dim],
        strides=[head_dim, 1],
        block_shape=[1, 1],
    )


def validate_cache_request(
    *,
    prefill: bool,
    inference_cache: InferenceCache | None,
    layer_id: int | str | None,
) -> None:
    if prefill and inference_cache is None:
        raise ValueError("prefill=True requires inference_cache")
    if (inference_cache is None) != (layer_id is None):
        raise ValueError(
            "inference_cache and layer_id must be provided together or both omitted"
        )


def _maybe_get_cache_context(
    *,
    inference_cache: InferenceCache | None,
    layer_id: int | str | None,
    batch_size: int,
    num_heads: int,
    head_dim: int,
    is_mla: bool,
    context_batch_size: int | None,
    device: torch.device,
    dtype: torch.dtype,
    compressed: bool,
):
    entry: "LayerKVEntry | None" = None
    if inference_cache is not None:
        try:
            entry = inference_cache.get_kv_cache(
                layer_id,
                is_mla=is_mla,
                num_heads=num_heads,
                head_dim=head_dim,
                dtype=dtype,
                device=device,
            )
        except ValueError:
            entry = None

    context_len = 0
    if entry is None:
        resolved_context_batch_size = (
            batch_size if context_batch_size is None else context_batch_size
        )
        total_context_len = 0
        kv_rows = total_context_len if is_mla else num_heads * total_context_len
        k_cache = torch.empty((kv_rows, head_dim), device=device, dtype=dtype)
        v_cache = torch.empty_like(k_cache)
        cu_seqlens_kv = (
            torch.zeros(
                resolved_context_batch_size + 1,
                device=device,
                dtype=torch.int32,
            )
            if compressed
            else None
        )

    else:
        k_cache = entry.k_cache
        v_cache = entry.v_cache

        resolved_context_batch_size = entry.context_batch_size
        if (
            context_batch_size is not None
            and context_batch_size != resolved_context_batch_size
        ):
            raise ValueError(
                "context_batch_size must match cached entry context_batch_size"
            )
        if resolved_context_batch_size < 1:
            raise ValueError("context_batch_size must be >= 1")

        total_context_len = int(entry.total_context_len)
        if compressed:
            cu_seqlens_kv = entry.cu_seqlens_kv
            if cu_seqlens_kv is None:
                if total_context_len % resolved_context_batch_size != 0:
                    raise ValueError(
                        "compressed path requires cu_seqlens_kv or uniform per-batch context lengths"
                    )
                context_len = total_context_len // resolved_context_batch_size
                cu_seqlens_kv = torch.arange(
                    resolved_context_batch_size + 1,
                    device=device,
                    dtype=torch.int32,
                ) * int(context_len)
        else:
            cu_seqlens_kv = None
            if total_context_len % resolved_context_batch_size != 0:
                raise ValueError(
                    "dense flash_attn_mlm requires uniform per-batch context lengths"
                )

            context_len = total_context_len // resolved_context_batch_size
            expected_rows = (
                total_context_len if is_mla else num_heads * total_context_len
            )
            if k_cache.shape[0] != expected_rows or v_cache.shape[0] != expected_rows:
                raise ValueError(
                    "cached dense k/v rows mismatch expected layout for is_mla/num_heads"
                )

    return CacheContext(
        k_cache=k_cache,
        v_cache=v_cache,
        context_batch_size=resolved_context_batch_size,
        context_len=context_len,
        total_context_len=total_context_len,
        cu_seqlens_kv=cu_seqlens_kv,
    )


def pack_for_kernel(
    x: torch.Tensor,
    meta: PackMetadata,
    flatten_for_kernel: bool = False,
):
    """Pack padded (B, H, N, D) into token-packed representation using cached indices.

    Returns:
        packed: (H, total_tokens, D) or (H * total_tokens, D) if flatten_for_kernel=True
    """
    if x.ndim != 4:
        raise ValueError("x must be rank-4 [B, H, N, D]")
    B, H, N, D = x.shape
    if B != meta.B or N != meta.N:
        raise ValueError("x shape mismatch with PackMetadata")

    x_hbnd = x.permute(1, 0, 2, 3).contiguous()  # (H, B, N, D)
    x_hbnd = x_hbnd.view(H, B * N, D)
    packed = x_hbnd.index_select(dim=1, index=meta.token_indices)

    if flatten_for_kernel:
        packed = packed.reshape(H * meta.total_tokens, D).contiguous()

    return packed


def _compute_total_q_len_padded(
    total_q_len_unpadded: int,
    cu_seqlens_q: torch.Tensor,
    block_m: int,
    block_n: int,
) -> int:
    tile_align = max(block_m, block_n)
    seq_lengths_q = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
    seq_tile_ends_q = (
        cu_seqlens_q[:-1]
        + torch.div(
            seq_lengths_q + (tile_align - 1),
            tile_align,
            rounding_mode="floor",
        )
        * tile_align
    )
    max_required_q_len = (
        int(torch.max(seq_tile_ends_q).item()) if seq_tile_ends_q.numel() > 0 else 0
    )
    return max(total_q_len_unpadded, max_required_q_len)


def require_cuda_tensors(
    *tensors: torch.Tensor, message: str = "All tensors must be CUDA tensors"
):
    if not all(t is None or t.is_cuda for t in tensors):
        raise ValueError(message)


def validate_qkv_same_shape_rank4(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    error_prefix: str,
) -> tuple[int, int, int, int]:
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError(f"{error_prefix} must be rank-4 tensors [B, H, N, D]")
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q, k, v must have the same shape")
    b, h, n, d = q.shape
    return int(b), int(h), int(n), int(d)


def validate_head_dim_supported(head_dim: int):
    if head_dim not in (16, 32, 64, 128, 256):
        raise ValueError("HEAD_DIM must be one of {16, 32, 64, 128, 256}")


def validate_cu_seqlens_rank1_min2(*cu_seqlens: torch.Tensor):
    for x in cu_seqlens:
        if x.ndim != 1:
            raise ValueError("cu_seqlens tensors must be rank-1")
        if x.numel() < 2:
            raise ValueError("cu_seqlens tensors must have at least 2 elements")


def validate_packed_cache_shapes(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor | None,
    *,
    num_heads: int,
    total_context_len: int,
    head_dim: int,
    is_mla: bool,
):
    if k_cache.ndim != 2:
        raise ValueError("k_cache must be a rank-2 packed tensor")
    if k_cache.shape[1] != head_dim:
        raise ValueError("k_cache second dim must match q/k/v head_dim")
    if is_mla:
        if v_cache is not None:
            if v_cache.ndim != 2:
                raise ValueError("MLA v_cache must be a rank-2 packed tensor")
            if v_cache.shape != k_cache.shape:
                raise ValueError("MLA v_cache shape must match k_cache shape")
    else:
        if v_cache is None:
            raise ValueError("non-MLA cache requires v_cache")
        if v_cache.ndim != 2:
            raise ValueError("v_cache must be a rank-2 packed tensor")
        if v_cache.shape != k_cache.shape:
            raise ValueError("v_cache shape must match k_cache shape")
        if v_cache.shape[1] != head_dim:
            raise ValueError("v_cache second dim must match q/k/v head_dim")
    expected_context_rows = (
        total_context_len if is_mla else num_heads * total_context_len
    )
    if k_cache.shape[0] != expected_context_rows:
        raise ValueError(
            "k_cache first dim mismatch: expected "
            f"{expected_context_rows} rows for is_mla={is_mla}"
        )
    if not is_mla and v_cache.shape[0] != expected_context_rows:
        raise ValueError(
            "v_cache first dim mismatch: expected "
            f"{expected_context_rows} rows for is_mla={is_mla}"
        )


def make_contiguous(*tensors: torch.Tensor | None) -> tuple[torch.Tensor | None, ...]:
    return tuple(t if t is None else t.contiguous() for t in tensors)


def _pack_flatten_for_kernel_into(
    x: torch.Tensor,
    meta: PackMetadata,
    expected_heads: int,
    total_q_len_padded: int,
    out: torch.Tensor,
) -> torch.Tensor:
    if x.ndim != 4:
        raise ValueError("x must be rank-4 [B, H, N, D]")
    b, h, n, d = x.shape
    if b != meta.B or n != meta.N:
        raise ValueError("x shape mismatch with PackMetadata")
    if h != expected_heads:
        raise ValueError("x head dimension mismatch for requested packed output")
    if out.shape != (expected_heads * total_q_len_padded, d):
        raise ValueError("out has unexpected shape for packed output")

    x_hbnd = x.permute(1, 0, 2, 3).contiguous().view(expected_heads, b * n, d)
    out_h = out.view(expected_heads, total_q_len_padded, d)
    packed_main = out_h[:, : meta.total_tokens, :]
    torch.index_select(x_hbnd, dim=1, index=meta.token_indices, out=packed_main)
    if total_q_len_padded > meta.total_tokens:
        out_h[:, meta.total_tokens :, :].zero_()
    return out


def pack_and_pad_main_tensors_for_mlm_compressed(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    q_meta: PackMetadata,
    num_heads: int,
    is_mla: bool,
    block_m: int,
    block_n: int,
    packing_cache: "PackingCache | None" = None,
    q_buffer_name: str = "q",
    k_buffer_name: str = "k",
    v_buffer_name: str = "v",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Fused pack + pad into final flattened q/k/v buffers for compressed kernel."""

    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q/k/v must be rank-4 padded tensors [B, H, N, D]")
    b, h, n, d = q.shape
    if q_meta.B != b or q_meta.N != n:
        raise ValueError("q_meta shape metadata does not match q/k/v padded shape")
    if h != num_heads:
        raise ValueError("q/k/v head dimension must match num_heads")

    total_q_len_unpadded = q_meta.total_tokens
    total_q_len_padded = _compute_total_q_len_padded(
        total_q_len_unpadded,
        q_meta.cu_seqlens,
        block_m,
        block_n,
    )

    def _alloc(name: str, rows: int, cols: int, template: torch.Tensor) -> torch.Tensor:
        if packing_cache is None:
            return torch.empty(
                (rows, cols), dtype=template.dtype, device=template.device
            )
        return packing_cache.get_2d(
            name,
            rows,
            cols,
            dtype=template.dtype,
            device=template.device,
        )

    q_out = _alloc(q_buffer_name, num_heads * total_q_len_padded, d, q)
    q_out = _pack_flatten_for_kernel_into(
        q, q_meta, num_heads, total_q_len_padded, q_out
    )

    if is_mla:
        k_src = k[:, :1, :, :]
        k_out = _alloc(k_buffer_name, total_q_len_padded, d, k_src)
        k_out = _pack_flatten_for_kernel_into(
            k_src, q_meta, 1, total_q_len_padded, k_out
        )
        v_out = k_out
    else:
        k_out = _alloc(k_buffer_name, num_heads * total_q_len_padded, d, k)
        v_out = _alloc(v_buffer_name, num_heads * total_q_len_padded, d, v)
        k_out = _pack_flatten_for_kernel_into(
            k, q_meta, num_heads, total_q_len_padded, k_out
        )
        v_out = _pack_flatten_for_kernel_into(
            v, q_meta, num_heads, total_q_len_padded, v_out
        )

    return q_out, k_out, v_out, total_q_len_padded


def pad_packed_main_tensors_for_mlm_compressed(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    num_heads: int,
    is_mla: bool,
    total_q_len_unpadded: int,
    cu_seqlens_q: torch.Tensor,
    block_m: int,
    block_n: int,
    packing_cache: "PackingCache | None" = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Pad flattened packed main q/k/v rows for descriptor-safe tile loads.

    This helper is specific to the compressed kernel launch contract where
    q/k/v are flattened as [heads * packed_len, D] for q and either
    [heads * packed_len, D] (non-MLA) or [packed_len, D] (MLA) for k/v.
    """

    total_q_len_padded = _compute_total_q_len_padded(
        total_q_len_unpadded,
        cu_seqlens_q,
        block_m,
        block_n,
    )

    if total_q_len_padded == total_q_len_unpadded:
        return q, k, v, total_q_len_unpadded

    def _pad_packed_per_head(
        x: torch.Tensor,
        h: int,
        old_len: int,
        new_len: int,
    ):
        d_local = x.shape[1]
        x_h = x.view(h, old_len, d_local)
        pad_rows = new_len - old_len
        pad = x.new_zeros((h, pad_rows, d_local))
        return torch.cat([x_h, pad], dim=1).reshape(h * new_len, d_local).contiguous()

    def _pad_packed_per_head_with_workspace(
        x: torch.Tensor,
        h: int,
        old_len: int,
        new_len: int,
        name: str,
    ):
        d_local = x.shape[1]
        out = packing_cache.get_2d(
            name,
            h * new_len,
            d_local,
            dtype=x.dtype,
            device=x.device,
        )
        x_h = x.reshape(h, old_len, d_local)
        out_h = out.reshape(h, new_len, d_local)
        out_h[:, :old_len, :].copy_(x_h)
        out_h[:, old_len:, :].zero_()
        return out

    kv_heads = 1 if is_mla else num_heads
    if packing_cache is None:
        q = _pad_packed_per_head(q, num_heads, total_q_len_unpadded, total_q_len_padded)
        k = _pad_packed_per_head(k, kv_heads, total_q_len_unpadded, total_q_len_padded)
        v = _pad_packed_per_head(v, kv_heads, total_q_len_unpadded, total_q_len_padded)
    else:
        q = _pad_packed_per_head_with_workspace(
            q,
            num_heads,
            total_q_len_unpadded,
            total_q_len_padded,
            "q",
        )
        k = _pad_packed_per_head_with_workspace(
            k,
            kv_heads,
            total_q_len_unpadded,
            total_q_len_padded,
            "k",
        )
        v = _pad_packed_per_head_with_workspace(
            v,
            kv_heads,
            total_q_len_unpadded,
            total_q_len_padded,
            "v",
        )
    return q, k, v, total_q_len_padded


def unpack_from_kernel(packed: torch.Tensor, meta: PackMetadata, H: int):
    """Unpack token-packed representation back to padded (B, H, N, D)."""
    if packed.ndim == 3:
        _, total_tokens, D = packed.shape
        packed_h = packed
    elif packed.ndim == 2:
        total_tokens = meta.total_tokens
        if total_tokens == 0:
            D = packed.shape[1]
            packed_h = packed.new_zeros((H, 0, D))
        else:
            if packed.shape[0] != H * total_tokens:
                raise ValueError("packed first dim must be H * total_tokens")
            D = packed.shape[1]
            packed_h = packed.view(H, total_tokens, D)
    else:
        raise ValueError("packed must be rank-2 or rank-3")

    if total_tokens != meta.total_tokens:
        raise ValueError("packed token count mismatch with PackMetadata")

    x = torch.zeros(
        (H, meta.B * meta.N, D),
        device=packed_h.device,
        dtype=packed_h.dtype,
    )
    x.index_copy_(dim=1, index=meta.token_indices, source=packed_h)
    x = x.view(H, meta.B, meta.N, D)
    return x.permute(1, 0, 2, 3).contiguous()  # (B, H, N, D)
