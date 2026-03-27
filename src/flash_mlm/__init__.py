import torch
import warnings

from flash_mlm.host.cache import InferenceCache, PackMetadata, PackingCache
from flash_mlm.host.host_utils import (
    build_pack_metadata,
    unpack_from_kernel,
)

# Guard triton imports for GPU-only functions
_TRITON_AVAILABLE = False
if torch.cuda.is_available():
    try:
        import triton
        from flash_mlm.host import (
            flash_attn_mlm,
            flash_attn_mlm_compressed,
            flash_attn_mlm_precompressed,
        )

        def _triton_alloc_fn(size: int, alignment: int, stream):
            return torch.empty(size, device="cuda", dtype=torch.int8)

        triton.set_allocator(_triton_alloc_fn)
        _TRITON_AVAILABLE = True
    except ImportError:
        warnings.warn(
            "GPU detected but triton is not installed. "
            "GPU attention kernels will not be available. "
            "Install with: pip install flash-mlm[gpu]",
            ImportWarning,
            stacklevel=2,
        )
else:
    # No GPU available, skip triton import silently
    pass

if not _TRITON_AVAILABLE:
    # Provide stub functions that raise helpful errors
    def flash_attn_mlm(*args, **kwargs):
        raise RuntimeError(
            "flash_attn_mlm requires GPU and triton. "
            "Install with: pip install flash-mlm[gpu]"
        )

    def flash_attn_mlm_compressed(*args, **kwargs):
        raise RuntimeError(
            "flash_attn_mlm_compressed requires GPU and triton. "
            "Install with: pip install flash-mlm[gpu]"
        )

    def flash_attn_mlm_precompressed(*args, **kwargs):
        raise RuntimeError(
            "flash_attn_mlm_precompressed requires GPU and triton. "
            "Install with: pip install flash-mlm[gpu]"
        )


__all__ = [
    "PackMetadata",
    "PackingCache",
    "InferenceCache",
    "build_pack_metadata",
    "unpack_from_kernel",
    "flash_attn_mlm",
    "flash_attn_mlm_compressed",
    "flash_attn_mlm_precompressed",
]
