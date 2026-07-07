from __future__ import annotations

# train the tokenizer & use the tokenizer to encode the whole dataset



import argparse
import math
import os
import tempfile
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .BPE import train_bpe
from .Tokenizer import Tokenizer

DEFAULT_SPECIAL_TOKEN = "<|endoftext|>"


def train_tokenizer(
    input_path: str | os.PathLike,
    vocab_size: int,
    vocab_path: str | os.PathLike,
    merges_path: str | os.PathLike,
    special_tokens: list[str],
) -> int:
    Path(vocab_path).parent.mkdir(parents=True, exist_ok=True)
    Path(merges_path).parent.mkdir(parents=True, exist_ok=True)
    vocab, merges = train_bpe(os.fspath(input_path), vocab_size, special_tokens)
    tokenizer = Tokenizer(vocab, merges, special_tokens)
    tokenizer.save(os.fspath(vocab_path), os.fspath(merges_path))
    return len(tokenizer.vocab)


def _iter_text_lines(input_path: str | os.PathLike) -> Iterator[str]:
    with open(input_path, encoding="utf-8") as input_file:
        yield from input_file


def _choose_dtype(vocab_size: int) -> np.dtype:
    if vocab_size <= np.iinfo(np.uint16).max + 1:
        return np.dtype(np.uint16)
    if vocab_size <= np.iinfo(np.uint32).max + 1:
        return np.dtype(np.uint32)
    return np.dtype(np.uint64)


def _write_encoded_tokens(
    token_ids: Iterator[int],
    raw_file,
    dtype: np.dtype,
    chunk_size: int,
) -> int:
    token_count = 0
    buffer: list[int] = []

    for token_id in token_ids:
        buffer.append(token_id)
        if len(buffer) >= chunk_size:
            values = np.asarray(buffer, dtype=dtype)
            values.tofile(raw_file)
            token_count += len(values)
            buffer.clear()

    if buffer:
        values = np.asarray(buffer, dtype=dtype)
        values.tofile(raw_file)
        token_count += len(values)

    return token_count


def _byte_ranges_by_line(input_path: str | os.PathLike, num_workers: int) -> list[tuple[int, int]]:
    file_size = os.path.getsize(input_path)
    if file_size == 0:
        return []

    target_chunk_size = math.ceil(file_size / num_workers)
    ranges: list[tuple[int, int]] = []
    start = 0

    with open(input_path, "rb") as input_file:
        while start < file_size:
            end = min(file_size, start + target_chunk_size)
            if end < file_size:
                input_file.seek(end)
                input_file.readline()
                end = input_file.tell()
            if end > start:
                ranges.append((start, end))
            start = end

    return ranges


def _iter_text_lines_from_byte_range(
    input_path: str | os.PathLike,
    start: int,
    end: int,
) -> Iterator[str]:
    with open(input_path, "rb") as input_file:
        input_file.seek(start)
        while input_file.tell() < end:
            line = input_file.readline()
            if not line:
                break
            yield line.decode("utf-8")


def _encode_dataset_range(
    input_path: str,
    start: int,
    end: int,
    output_dir: str,
    dtype_name: str,
    chunk_size: int,
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str],
) -> tuple[str, int]:
    dtype = np.dtype(dtype_name)
    tokenizer = Tokenizer(vocab, merges, special_tokens)

    with tempfile.NamedTemporaryFile(dir=output_dir, delete=False) as raw_file:
        raw_path = raw_file.name
        token_count = _write_encoded_tokens(
            tokenizer.encode_iterable(
                _iter_text_lines_from_byte_range(input_path, start, end)
            ),
            raw_file,
            dtype,
            chunk_size,
        )
        raw_file.flush()

    return raw_path, token_count


def _copy_raw_parts_to_npy(
    part_paths: list[str],
    part_counts: list[int],
    output_path: Path,
    dtype: np.dtype,
) -> None:
    token_count = sum(part_counts)
    if token_count == 0:
        raise ValueError("input produced no tokens")

    destination = np.lib.format.open_memmap(
        output_path, mode="w+", dtype=dtype, shape=(token_count,)
    )
    offset = 0
    try:
        for part_path, part_count in zip(part_paths, part_counts, strict=True):
            if part_count == 0:
                continue
            source = np.memmap(part_path, dtype=dtype, mode="r", shape=(part_count,))
            destination[offset : offset + part_count] = source[:]
            offset += part_count
            del source
        destination.flush()
    finally:
        del destination


def _encode_dataset_parallel(
    input_path: str | os.PathLike,
    output_path: Path,
    tokenizer: Tokenizer,
    dtype: np.dtype,
    chunk_size: int,
    num_workers: int,
) -> int:
    ranges = _byte_ranges_by_line(input_path, num_workers)
    if not ranges:
        raise ValueError("input produced no tokens")

    merges = [
        pair
        for pair, _ in sorted(tokenizer.merge_rank.items(), key=lambda item: item[1])
    ]
    output_dir = os.fspath(output_path.parent)
    part_paths: list[str] = []
    part_counts: list[int] = []

    try:
        with ProcessPoolExecutor(max_workers=min(num_workers, len(ranges))) as executor:
            futures = [
                executor.submit(
                    _encode_dataset_range,
                    os.fspath(input_path),
                    start,
                    end,
                    output_dir,
                    dtype.name,
                    chunk_size,
                    tokenizer.vocab,
                    merges,
                    tokenizer.special_tokens,
                )
                for start, end in ranges
            ]
            for future in futures:
                part_path, part_count = future.result()
                part_paths.append(part_path)
                part_counts.append(part_count)

        _copy_raw_parts_to_npy(part_paths, part_counts, output_path, dtype)
        return sum(part_counts)
    finally:
        for part_path in part_paths:
            Path(part_path).unlink(missing_ok=True)


def encode_dataset(
    input_path: str | os.PathLike,
    output_path: str | os.PathLike,
    tokenizer: Tokenizer,
    chunk_size: int = 1_000_000,
    num_workers: int = 1,
) -> tuple[int, np.dtype]:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if num_workers <= 0:
        raise ValueError("num_workers must be greater than 0")

    dtype = _choose_dtype(len(tokenizer.vocab))

    if num_workers > 1:
        token_count = _encode_dataset_parallel(
            input_path,
            output_path,
            tokenizer,
            dtype,
            chunk_size,
            num_workers,
        )
        return token_count, dtype

    with tempfile.NamedTemporaryFile(dir=output_path.parent, delete=False) as raw_file:
        raw_path = Path(raw_file.name)
        try:
            token_count = _write_encoded_tokens(
                tokenizer.encode_iterable(_iter_text_lines(input_path)),
                raw_file,
                dtype,
                chunk_size,
            )
            raw_file.flush()
            _copy_raw_parts_to_npy([os.fspath(raw_path)], [token_count], output_path, dtype)
        finally:
            raw_path.unlink(missing_ok=True)

    return token_count, dtype


def _special_tokens(args: argparse.Namespace) -> list[str]:
    return args.special_token or [DEFAULT_SPECIAL_TOKEN]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare tokenizer and token datasets for CS336 training.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train-tokenizer")
    train_parser.add_argument("--input", required=True)
    train_parser.add_argument("--vocab-size", type=int, required=True)
    train_parser.add_argument("--vocab-out", required=True)
    train_parser.add_argument("--merges-out", required=True)
    train_parser.add_argument("--special-token", action="append")

    encode_parser = subparsers.add_parser("encode")
    encode_parser.add_argument("--input", required=True)
    encode_parser.add_argument("--output", required=True)
    encode_parser.add_argument("--vocab", required=True)
    encode_parser.add_argument("--merges", required=True)
    encode_parser.add_argument("--special-token", action="append")
    encode_parser.add_argument("--chunk-size", type=int, default=1_000_000)
    encode_parser.add_argument("--num-workers", type=int, default=1)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    special_tokens = _special_tokens(args)

    if args.command == "train-tokenizer":
        actual_vocab_size = train_tokenizer(
            args.input,
            args.vocab_size,
            args.vocab_out,
            args.merges_out,
            special_tokens,
        )
        print(f"saved tokenizer with vocab_size={actual_vocab_size}")
        return

    tokenizer = Tokenizer.from_files(args.vocab, args.merges, special_tokens)
    token_count, dtype = encode_dataset(
        args.input,
        args.output,
        tokenizer,
        chunk_size=args.chunk_size,
        num_workers=args.num_workers,
    )
    print(f"saved {token_count} tokens to {args.output} with dtype={dtype.name}")


if __name__ == "__main__":
    main()
