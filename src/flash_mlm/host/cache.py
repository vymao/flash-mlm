from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from flash_mlm.host.host_utils import PackMetadata


class PackingCache:
    """Reusable cache for pack metadata and packed main tensor workspaces."""

    def __init__(self):
        self._buffers: dict[
            tuple[str, str, int | None, torch.dtype, int], torch.Tensor
        ] = {}
        self._pack_meta: dict[
            tuple[str, int | None, int, int, tuple[int, ...]], "PackMetadata"
        ] = {}

    def get_2d(
        self,
        name: str,
        rows: int,
        cols: int,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        if rows < 0:
            raise ValueError("rows must be >= 0")
        if cols < 1:
            raise ValueError("cols must be >= 1")

        key = (name, device.type, device.index, dtype, cols)
        buf = self._buffers.get(key)
        if buf is None or buf.shape[0] < rows:
            buf = torch.empty((rows, cols), dtype=dtype, device=device)
            self._buffers[key] = buf
        return buf[:rows]

    def get_pack_metadata(
        self,
        lengths: torch.Tensor,
        *,
        N: int,
        block_n: int,
    ) -> "PackMetadata":
        lengths_key = tuple(lengths.to(device="cpu", dtype=torch.int32).tolist())
        key = (lengths.device.type, lengths.device.index, N, block_n, lengths_key)
        meta = self._pack_meta.get(key)
        if meta is not None:
            return meta

        from flash_mlm.host.host_utils import build_pack_metadata

        meta = build_pack_metadata(lengths, N, block_n)
        self._pack_meta[key] = meta
        return meta

    def clear(self):
        self._buffers.clear()
        self._pack_meta.clear()


@dataclass
class LayerKVEntry:
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    total_context_len: int
    cu_seqlens_kv: torch.Tensor | None
    context_batch_size: int
    is_mla: bool
    num_heads: int
    head_dim: int
    dtype: torch.dtype
    device: torch.device


class InferenceCache:
    """Layer-scoped inference cache for packed KV context plus PackingCache reuse."""

    def __init__(self):
        self.packing = PackingCache()
        self._layer_kv: dict[str, LayerKVEntry] = {}

    @staticmethod
    def _layer_key(layer_id: int | str) -> str:
        return str(layer_id)

    @staticmethod
    def _validate_kv_common_inputs(
        *,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        total_context_len: int,
        is_mla: bool,
        num_heads: int,
        head_dim: int,
    ) -> None:
        if k_cache.ndim != 2 or v_cache.ndim != 2:
            raise ValueError("k_cache/v_cache must be rank-2 packed tensors")
        if v_cache.shape != k_cache.shape:
            raise ValueError("v_cache shape must match k_cache shape")
        if k_cache.shape[1] != head_dim:
            raise ValueError("k_cache/v_cache second dim must match head_dim")
        expected_rows = total_context_len if is_mla else num_heads * total_context_len
        if k_cache.shape[0] != expected_rows:
            raise ValueError(
                "k_cache/v_cache first dim mismatch: expected "
                f"{expected_rows} rows for is_mla={is_mla}"
            )

    @staticmethod
    def _validate_kv_entry_inputs(
        *,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        total_context_len: int,
        cu_seqlens_kv: torch.Tensor,
        context_batch_size: int,
        is_mla: bool,
        num_heads: int,
        head_dim: int,
    ) -> None:
        InferenceCache._validate_kv_common_inputs(
            k_cache=k_cache,
            v_cache=v_cache,
            total_context_len=total_context_len,
            is_mla=is_mla,
            num_heads=num_heads,
            head_dim=head_dim,
        )
        if cu_seqlens_kv.ndim != 1 or cu_seqlens_kv.numel() < 2:
            raise ValueError("cu_seqlens_kv must be rank-1 with at least 2 elements")
        if context_batch_size < 1:
            raise ValueError("context_batch_size must be >= 1")
        if context_batch_size + 1 > cu_seqlens_kv.numel():
            raise ValueError(
                "context_batch_size is incompatible with cu_seqlens_kv length"
            )

    def prefill_kv_cache(
        self,
        layer_id: int | str,
        *,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        total_context_len: int | None = None,
        cu_seqlens_kv: torch.Tensor | None = None,
        context_batch_size: int,
        is_mla: bool,
        num_heads: int,
        head_dim: int,
    ) -> None:
        """Store pre-packed KV cache buffers for a layer without copying."""
        if context_batch_size < 1:
            raise ValueError("context_batch_size must be >= 1")
        if num_heads < 1:
            raise ValueError("num_heads must be >= 1")

        if total_context_len is None:
            if is_mla:
                total_context_len = int(k_cache.shape[0])
            else:
                if k_cache.shape[0] % num_heads != 0:
                    raise ValueError(
                        "non-MLA k_cache rows must be divisible by num_heads"
                    )
                total_context_len = int(k_cache.shape[0] // num_heads)
        else:
            total_context_len = int(total_context_len)

        if cu_seqlens_kv is None:
            self._validate_kv_common_inputs(
                k_cache=k_cache,
                v_cache=v_cache,
                total_context_len=total_context_len,
                is_mla=is_mla,
                num_heads=num_heads,
                head_dim=head_dim,
            )
            if total_context_len % context_batch_size != 0:
                raise ValueError(
                    "when cu_seqlens_kv is omitted, total_context_len must be divisible by context_batch_size"
                )
        else:
            self._validate_kv_entry_inputs(
                k_cache=k_cache,
                v_cache=v_cache,
                total_context_len=total_context_len,
                cu_seqlens_kv=cu_seqlens_kv,
                context_batch_size=context_batch_size,
                is_mla=is_mla,
                num_heads=num_heads,
                head_dim=head_dim,
            )

        layer_key = self._layer_key(layer_id)
        self._layer_kv[layer_key] = LayerKVEntry(
            k_cache=k_cache,
            v_cache=v_cache,
            total_context_len=int(total_context_len),
            cu_seqlens_kv=(
                cu_seqlens_kv.contiguous() if cu_seqlens_kv is not None else None
            ),
            context_batch_size=int(context_batch_size),
            is_mla=bool(is_mla),
            num_heads=int(num_heads),
            head_dim=int(head_dim),
            dtype=k_cache.dtype,
            device=k_cache.device,
        )

    def get_kv_cache(
        self,
        layer_id: int | str,
        *,
        is_mla: bool,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> LayerKVEntry:
        layer_key = self._layer_key(layer_id)
        entry = self._layer_kv.get(layer_key)
        if entry is None:
            raise ValueError(f"No cached KV context for layer_id={layer_id}")
        if entry.is_mla != is_mla:
            raise ValueError("Cached KV mode (is_mla) does not match request")
        if entry.num_heads != num_heads:
            raise ValueError("Cached num_heads does not match request")
        if entry.head_dim != head_dim:
            raise ValueError("Cached head_dim does not match request")
        if entry.dtype != dtype:
            raise ValueError("Cached KV dtype does not match request")
        if entry.device != device:
            raise ValueError("Cached KV device does not match request")
        return entry

    def clear_layer(self, layer_id: int | str):
        self._layer_kv.pop(self._layer_key(layer_id), None)

    def clear(self):
        self.packing.clear()
        self._layer_kv.clear()
