import math
from collections import Counter

import numpy as np
import pytest

from cs336_basics.BPE import BYTE_TOKENS
from cs336_basics.Prepare import encode_dataset
from cs336_basics.Tokenizer import Tokenizer

from .adapters import run_get_batch


def test_get_batch():
    dataset = np.arange(0, 100)
    context_length = 7
    batch_size = 32
    device = "cpu"

    # Sanity check to make sure that the random samples are indeed somewhat random.
    starting_indices = Counter()
    num_iters = 1000
    for _ in range(num_iters):
        x, y = run_get_batch(
            dataset=dataset,
            batch_size=batch_size,
            context_length=context_length,
            device=device,
        )

        # Make sure the shape is correct
        assert x.shape == (batch_size, context_length)
        assert y.shape == (batch_size, context_length)

        # Make sure the y's are always offset by 1
        np.testing.assert_allclose((x + 1).detach().numpy(), y.detach().numpy())

        starting_indices.update(x[:, 0].tolist())

    # Make sure we never sample an invalid start index
    num_possible_starting_indices = len(dataset) - context_length
    assert max(starting_indices) == num_possible_starting_indices - 1
    assert min(starting_indices) == 0
    # Expected # of times that we see each starting index
    expected_count = (num_iters * batch_size) / num_possible_starting_indices
    standard_deviation = math.sqrt(
        (num_iters * batch_size) * (1 / num_possible_starting_indices) * (1 - (1 / num_possible_starting_indices))
    )
    # Range for expected outcomes (mu +/- 5sigma). For a given index,
    # this should happen 99.99994% of the time of the time.
    # So, in the case where we have 93 possible start indices,
    # the entire test should pass with 99.9944202% of the time
    occurrences_lower_bound = expected_count - 5 * standard_deviation
    occurrences_upper_bound = expected_count + 5 * standard_deviation

    for starting_index, count in starting_indices.items():
        if count < occurrences_lower_bound:
            raise ValueError(
                f"Starting index {starting_index} occurs {count} times, but expected at least {occurrences_lower_bound}"
            )
        if count > occurrences_upper_bound:
            raise ValueError(
                f"Starting index {starting_index} occurs {count} times, but expected at most {occurrences_upper_bound}"
            )

    with pytest.raises((RuntimeError, AssertionError)) as excinfo:
        # We're assuming that cuda:99 is an invalid device ordinal.
        # Just adding this here to make sure that the device flag is
        # being handled.
        run_get_batch(
            dataset=dataset,
            batch_size=batch_size,
            context_length=context_length,
            device="cuda:99",
        )
        assert "CUDA error" in str(excinfo.value) or "Torch not compiled with CUDA enabled" in str(excinfo.value)


def test_parallel_encode_dataset_matches_single_worker(tmp_path):
    input_path = tmp_path / "input.txt"
    single_output_path = tmp_path / "single.npy"
    parallel_output_path = tmp_path / "parallel.npy"
    input_path.write_text("hello world\nhello<|endoftext|>\nbye world\n", encoding="utf-8")

    vocab = {token_id: token for token_id, token in enumerate(BYTE_TOKENS)}
    tokenizer = Tokenizer(vocab, [], special_tokens=["<|endoftext|>"])

    single_count, single_dtype = encode_dataset(
        input_path,
        single_output_path,
        tokenizer,
        chunk_size=3,
        num_workers=1,
    )
    parallel_count, parallel_dtype = encode_dataset(
        input_path,
        parallel_output_path,
        tokenizer,
        chunk_size=3,
        num_workers=4,
    )

    assert single_count == parallel_count
    assert single_dtype == parallel_dtype
    np.testing.assert_array_equal(np.load(single_output_path), np.load(parallel_output_path))
