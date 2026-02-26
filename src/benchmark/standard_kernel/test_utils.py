import torch

from benchmark.standard_kernel.utils import _build_shared_inputs


def _build_kwargs(
    *,
    has_cache: bool,
    context_batch_size: int,
    batch_size: int = 3,
    num_context_seqs_per_query: int = 3,
):
    context_subseq_len_active = 3
    context_subseq_len_padded = 6
    return dict(
        batch_size=batch_size,
        num_heads=2,
        query_seq_len_active=4,
        query_seq_len_padded=6,
        context_len_active=context_subseq_len_active * num_context_seqs_per_query,
        context_subseq_len_active=context_subseq_len_active,
        context_subseq_len_padded=context_subseq_len_padded,
        context_batch_size=context_batch_size,
        num_context_seqs_per_query=num_context_seqs_per_query,
        head_dim=16,
        has_cache=has_cache,
        block_n=2,
        q_ratio_min=1.0,
        q_ratio_max=1.0,
        kv_ratio_min=1.0,
        kv_ratio_max=1.0,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )


def test_build_shared_inputs_no_cache_shapes_and_zero_lengths():
    kwargs = _build_kwargs(has_cache=False, context_batch_size=1)
    payload = _build_shared_inputs(**kwargs)

    assert payload["q"].shape == (
        kwargs["batch_size"],
        kwargs["num_heads"],
        kwargs["query_seq_len_padded"],
        kwargs["head_dim"],
    )
    assert payload["k_cache_dense"].shape == (1, 2, 0, 16)
    assert payload["v_cache_dense"].shape == (1, 2, 0, 16)

    torch.testing.assert_close(
        payload["lengths_kv_varlen"], torch.tensor([0], dtype=torch.int32)
    )
    torch.testing.assert_close(
        payload["lengths_kv_dense"], torch.tensor([0], dtype=torch.int32)
    )
    torch.testing.assert_close(
        payload["cu_seqlens_kv"], torch.tensor([0, 0], dtype=torch.int32)
    )

    assert payload["k_cache_compact"].shape == (0, 16)
    assert payload["v_cache_compact"].shape == (0, 16)
    assert payload["k_cache_compact_mla"].shape == (0, 16)
    assert payload["v_cache_compact_mla"].shape == (0, 16)


def test_build_shared_inputs_cache_single_shapes_and_lengths():
    kwargs = _build_kwargs(has_cache=True, context_batch_size=1)
    payload = _build_shared_inputs(**kwargs)

    expected_active_context_len = (
        kwargs["context_subseq_len_active"] * kwargs["num_context_seqs_per_query"]
    )
    expected_padded_context_len = (
        kwargs["context_subseq_len_padded"] * kwargs["num_context_seqs_per_query"]
    )

    torch.testing.assert_close(
        payload["lengths_kv_varlen"],
        torch.tensor([expected_active_context_len], dtype=torch.int32),
    )
    torch.testing.assert_close(
        payload["lengths_kv_dense"],
        torch.tensor([expected_active_context_len], dtype=torch.int32),
    )
    torch.testing.assert_close(
        payload["cu_seqlens_kv"],
        torch.tensor([0, expected_active_context_len], dtype=torch.int32),
    )

    assert payload["k_cache_dense"].shape == (1, 2, expected_padded_context_len, 16)
    assert payload["v_cache_dense"].shape == (1, 2, expected_padded_context_len, 16)
    assert payload["k_cache_compact"].shape == (2 * expected_active_context_len, 16)
    assert payload["v_cache_compact"].shape == (2 * expected_active_context_len, 16)
    assert payload["k_cache_compact_mla"].shape == (expected_active_context_len, 16)
    assert payload["v_cache_compact_mla"].shape == (expected_active_context_len, 16)


def test_build_shared_inputs_cache_full_shapes_and_cu_seqlens():
    kwargs = _build_kwargs(has_cache=True, context_batch_size=3)
    payload = _build_shared_inputs(**kwargs)

    expected_active_context_len = (
        kwargs["context_subseq_len_active"] * kwargs["num_context_seqs_per_query"]
    )
    expected_padded_context_len = (
        kwargs["context_subseq_len_padded"] * kwargs["num_context_seqs_per_query"]
    )
    expected_total_context_len = (
        kwargs["context_batch_size"] * expected_active_context_len
    )

    torch.testing.assert_close(
        payload["lengths_kv_varlen"],
        torch.full((3,), expected_active_context_len, dtype=torch.int32),
    )
    torch.testing.assert_close(
        payload["lengths_kv_dense"],
        torch.full((3,), expected_active_context_len, dtype=torch.int32),
    )
    torch.testing.assert_close(
        payload["cu_seqlens_kv"],
        torch.tensor(
            [
                0,
                expected_active_context_len,
                2 * expected_active_context_len,
                expected_total_context_len,
            ],
            dtype=torch.int32,
        ),
    )

    assert payload["k_cache_dense"].shape == (3, 2, expected_padded_context_len, 16)
    assert payload["v_cache_dense"].shape == (3, 2, expected_padded_context_len, 16)
    assert payload["k_cache_compact"].shape == (2 * expected_total_context_len, 16)
    assert payload["v_cache_compact"].shape == (2 * expected_total_context_len, 16)
    assert payload["k_cache_compact_mla"].shape == (expected_total_context_len, 16)
    assert payload["v_cache_compact_mla"].shape == (expected_total_context_len, 16)

    assert payload["q_meta"].B == 3
    assert payload["q_meta"].N == kwargs["query_seq_len_padded"]
