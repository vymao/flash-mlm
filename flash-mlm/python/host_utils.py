from dataclasses import dataclass

import torch


@dataclass
class PackMetadata:
    """Reusable metadata for fast pack/unpack across layers."""

    B: int
    N: int
    lengths: torch.Tensor
    cu_seqlens: torch.Tensor
    token_indices: torch.Tensor
    total_tokens: int


def build_pack_metadata(lengths: torch.Tensor, N: int) -> PackMetadata:
    """Precompute reusable indexing for variable-length packed tensors.

    Args:
        lengths: (B,) int tensor with active tokens per batch entry.
        N: padded max sequence length.
    """
    if lengths.ndim != 1:
        raise ValueError("lengths must be rank-1 [B]")
    if N < 0:
        raise ValueError("N must be >= 0")
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

    return PackMetadata(
        B=B,
        N=N,
        lengths=lengths_i32,
        cu_seqlens=cu_seqlens,
        token_indices=token_indices,
        total_tokens=int(cu_seqlens[-1].item()),
    )


def pack_for_kernel(
    x: torch.Tensor, meta: PackMetadata, flatten_for_kernel: bool = False
):
    """Pack padded (B, H, N, D) into token-packed representation using cached indices.

    Returns:
        packed: (H, total_tokens, D) or (H * total_tokens, D) if flatten_for_kernel=True
        cu_seqlens: (B+1,) int32
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

    return packed, meta.cu_seqlens


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
