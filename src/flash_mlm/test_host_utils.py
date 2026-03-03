import pytest
import torch

from flash_mlm.host_utils import (
    build_batch_ids_from_cu_seqlens,
    build_pack_metadata,
    build_q_tile_starts_from_cu_seqlens,
    num_query_tiles_from_cu_seqlens,
    num_query_tiles_from_meta,
    pack_and_pad_main_tensors_for_mlm_compressed,
    pad_packed_main_tensors_for_mlm_compressed,
    pack_for_kernel,
    unpack_from_kernel,
)
from flash_mlm.cache import PackingCache


def test_build_pack_metadata_precomputes_indices_and_batch_ids():
    lengths = torch.tensor([4, 2], dtype=torch.int32)
    meta = build_pack_metadata(lengths, N=4, block_n=3)

    assert meta.B == 2
    assert meta.N == 4
    assert meta.total_tokens == 6
    assert meta.batch_ids_block_n == 3

    torch.testing.assert_close(
        meta.cu_seqlens,
        torch.tensor([0, 4, 6], dtype=torch.int32),
    )
    torch.testing.assert_close(
        meta.token_indices,
        torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.int64),
    )
    torch.testing.assert_close(
        meta.batch_ids_q,
        torch.tensor([0, 0, 1], dtype=torch.int32),
    )
    torch.testing.assert_close(
        meta.q_tile_starts_q,
        torch.tensor([0, 3, 4], dtype=torch.int32),
    )


def test_build_pack_metadata_requires_valid_block_n():
    lengths = torch.tensor([1, 1], dtype=torch.int32)
    with pytest.raises(ValueError, match="block_n must be >= 1"):
        build_pack_metadata(lengths, N=2, block_n=0)


def test_pack_for_kernel_nonflatten_and_flatten_values():
    lengths = torch.tensor([4, 2], dtype=torch.int32)
    meta = build_pack_metadata(lengths, N=4, block_n=3)

    x = torch.arange(2 * 2 * 4 * 1, dtype=torch.float32).view(2, 2, 4, 1)

    packed = pack_for_kernel(x, meta, flatten_for_kernel=False)
    expected_nonflatten = torch.tensor(
        [
            [[0.0], [1.0], [2.0], [3.0], [8.0], [9.0]],
            [[4.0], [5.0], [6.0], [7.0], [12.0], [13.0]],
        ]
    )
    torch.testing.assert_close(packed, expected_nonflatten)

    packed_flat = pack_for_kernel(x, meta, flatten_for_kernel=True)
    expected_flat = torch.tensor(
        [
            [0.0],
            [1.0],
            [2.0],
            [3.0],
            [8.0],
            [9.0],
            [4.0],
            [5.0],
            [6.0],
            [7.0],
            [12.0],
            [13.0],
        ]
    )
    torch.testing.assert_close(packed_flat, expected_flat)


def test_unpack_from_kernel_restores_valid_tokens_and_zeros_padding():
    lengths = torch.tensor([4, 2], dtype=torch.int32)
    meta = build_pack_metadata(lengths, N=4, block_n=2)

    x = torch.arange(2 * 2 * 4 * 1, dtype=torch.float32).view(2, 2, 4, 1)
    packed = pack_for_kernel(x, meta, flatten_for_kernel=True)
    restored = unpack_from_kernel(packed, meta, H=2)

    expected = x.clone()
    expected[1, :, 2:, :] = 0
    torch.testing.assert_close(restored, expected)


def test_build_batch_ids_from_cu_seqlens_matches_expected_tiles():
    cu_seqlens_q = torch.tensor([0, 5, 7], dtype=torch.int32)
    batch_ids = build_batch_ids_from_cu_seqlens(cu_seqlens_q, block_n=4)
    torch.testing.assert_close(batch_ids, torch.tensor([0, 0, 1], dtype=torch.int32))


def test_build_q_tile_starts_and_num_tiles_from_cu_seqlens():
    cu_seqlens_q = torch.tensor([0, 5, 7], dtype=torch.int32)
    q_tile_starts = build_q_tile_starts_from_cu_seqlens(cu_seqlens_q, block_n=4)
    torch.testing.assert_close(
        q_tile_starts, torch.tensor([0, 4, 5], dtype=torch.int32)
    )
    assert num_query_tiles_from_cu_seqlens(cu_seqlens_q, block_n=4) == 3


def test_num_query_tiles_from_meta_uses_precomputed_tile_starts():
    lengths = torch.tensor([5, 2], dtype=torch.int32)
    meta = build_pack_metadata(lengths, N=5, block_n=4)
    assert num_query_tiles_from_meta(meta) == 3


def test_pad_packed_main_tensors_for_mlm_compressed_pads_to_required_len():
    num_heads = 2
    total_q_len_unpadded = 6
    d = 1
    q = torch.arange(num_heads * total_q_len_unpadded, dtype=torch.float32).view(-1, d)
    k = torch.arange(num_heads * total_q_len_unpadded, dtype=torch.float32).view(-1, d)
    v = k.clone()
    cu_seqlens_q = torch.tensor([0, 4, 6], dtype=torch.int32)

    q_p, k_p, v_p, total_q_len_padded = pad_packed_main_tensors_for_mlm_compressed(
        q,
        k,
        v,
        num_heads=num_heads,
        is_mla=False,
        total_q_len_unpadded=total_q_len_unpadded,
        cu_seqlens_q=cu_seqlens_q,
        block_m=4,
        block_n=4,
    )

    assert total_q_len_padded == 8
    assert q_p.shape == (num_heads * total_q_len_padded, d)
    assert k_p.shape == (num_heads * total_q_len_padded, d)
    assert v_p.shape == (num_heads * total_q_len_padded, d)
    torch.testing.assert_close(
        q_p[:, 0],
        torch.tensor(
            [0, 1, 2, 3, 4, 5, 0, 0, 6, 7, 8, 9, 10, 11, 0, 0], dtype=torch.float32
        ),
    )


def test_packing_cache_reuses_allocation_for_smaller_rows():
    cache = PackingCache()
    first = cache.get_2d(
        "q",
        rows=8,
        cols=4,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    second = cache.get_2d(
        "q",
        rows=6,
        cols=4,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    assert first.untyped_storage().data_ptr() == second.untyped_storage().data_ptr()
    assert second.shape == (6, 4)


def test_pad_packed_main_tensors_for_mlm_compressed_uses_packing_cache():
    cache = PackingCache()
    num_heads = 2
    total_q_len_unpadded = 6
    d = 1
    q = torch.arange(num_heads * total_q_len_unpadded, dtype=torch.float32).view(-1, d)
    k = torch.arange(num_heads * total_q_len_unpadded, dtype=torch.float32).view(-1, d)
    v = k.clone()
    cu_seqlens_q = torch.tensor([0, 4, 6], dtype=torch.int32)

    q_p, k_p, v_p, total_q_len_padded = pad_packed_main_tensors_for_mlm_compressed(
        q,
        k,
        v,
        num_heads=num_heads,
        is_mla=False,
        total_q_len_unpadded=total_q_len_unpadded,
        cu_seqlens_q=cu_seqlens_q,
        block_m=4,
        block_n=4,
        packing_cache=cache,
    )

    assert total_q_len_padded == 8
    assert (
        q_p.untyped_storage().data_ptr()
        == cache.get_2d(
            "q", rows=16, cols=1, dtype=torch.float32, device=torch.device("cpu")
        )
        .untyped_storage()
        .data_ptr()
    )
    assert (
        k_p.untyped_storage().data_ptr()
        == cache.get_2d(
            "k", rows=16, cols=1, dtype=torch.float32, device=torch.device("cpu")
        )
        .untyped_storage()
        .data_ptr()
    )
    assert (
        v_p.untyped_storage().data_ptr()
        == cache.get_2d(
            "v", rows=16, cols=1, dtype=torch.float32, device=torch.device("cpu")
        )
        .untyped_storage()
        .data_ptr()
    )


def test_packing_cache_reuses_pack_metadata_for_same_lengths_signature():
    cache = PackingCache()
    lengths = torch.tensor([4, 2], dtype=torch.int32)

    meta_1 = cache.get_pack_metadata(lengths, N=4, block_n=3)
    meta_2 = cache.get_pack_metadata(lengths.clone(), N=4, block_n=3)

    assert meta_1 is meta_2


def test_pack_and_pad_main_tensors_for_mlm_compressed_matches_two_step_non_mla():
    lengths = torch.tensor([4, 2], dtype=torch.int32)
    meta = build_pack_metadata(lengths, N=4, block_n=4)

    q = torch.arange(2 * 2 * 4 * 1, dtype=torch.float32).view(2, 2, 4, 1)
    k = q + 100
    v = q + 200

    q_ref = pack_for_kernel(q, meta, flatten_for_kernel=True)
    k_ref = pack_for_kernel(k, meta, flatten_for_kernel=True)
    v_ref = pack_for_kernel(v, meta, flatten_for_kernel=True)
    q_ref, k_ref, v_ref, total_ref = pad_packed_main_tensors_for_mlm_compressed(
        q_ref,
        k_ref,
        v_ref,
        num_heads=2,
        is_mla=False,
        total_q_len_unpadded=meta.total_tokens,
        cu_seqlens_q=meta.cu_seqlens,
        block_m=4,
        block_n=4,
    )

    q_new, k_new, v_new, total_new = pack_and_pad_main_tensors_for_mlm_compressed(
        q,
        k,
        v,
        q_meta=meta,
        num_heads=2,
        is_mla=False,
        block_m=4,
        block_n=4,
    )

    assert total_new == total_ref
    torch.testing.assert_close(q_new, q_ref)
    torch.testing.assert_close(k_new, k_ref)
    torch.testing.assert_close(v_new, v_ref)


def test_pack_and_pad_main_tensors_for_mlm_compressed_uses_packing_cache_mla():
    cache = PackingCache()
    lengths = torch.tensor([4, 2], dtype=torch.int32)
    meta = cache.get_pack_metadata(lengths, N=4, block_n=4)

    q = torch.arange(2 * 2 * 4 * 1, dtype=torch.float32).view(2, 2, 4, 1)
    k = q + 10
    v = q + 20

    q_out, k_out, v_out, total_q_len = pack_and_pad_main_tensors_for_mlm_compressed(
        q,
        k,
        v,
        q_meta=meta,
        num_heads=2,
        is_mla=True,
        block_m=4,
        block_n=4,
        packing_cache=cache,
    )

    assert total_q_len == 8
    assert q_out.shape == (16, 1)
    assert k_out.shape == (8, 1)
    assert v_out is k_out
    assert (
        q_out.untyped_storage().data_ptr()
        == cache.get_2d("q", 16, 1, dtype=torch.float32, device=torch.device("cpu"))
        .untyped_storage()
        .data_ptr()
    )
    assert (
        k_out.untyped_storage().data_ptr()
        == cache.get_2d("k", 8, 1, dtype=torch.float32, device=torch.device("cpu"))
        .untyped_storage()
        .data_ptr()
    )
