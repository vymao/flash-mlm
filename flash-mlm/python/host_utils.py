from dataclasses import dataclass

import torch


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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Pad flattened packed main q/k/v rows for descriptor-safe tile loads.

    This helper is specific to the compressed kernel launch contract where
    q/k/v are flattened as [heads * packed_len, D] for q and either
    [heads * packed_len, D] (non-MLA) or [packed_len, D] (MLA) for k/v.
    """

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
    total_q_len_padded = max(total_q_len_unpadded, max_required_q_len)

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

    kv_heads = 1 if is_mla else num_heads
    q = _pad_packed_per_head(q, num_heads, total_q_len_unpadded, total_q_len_padded)
    k = _pad_packed_per_head(k, kv_heads, total_q_len_unpadded, total_q_len_padded)
    v = _pad_packed_per_head(v, kv_heads, total_q_len_unpadded, total_q_len_padded)
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
