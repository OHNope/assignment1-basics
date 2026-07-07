from collections import defaultdict
from collections.abc import Iterable, Iterator
import heapq
from multiprocessing import Pool
import os
import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
BYTE_TOKENS = tuple(bytes([i]) for i in range(256))

READ_CHARS = 8 * 1024 * 1024
TARGET_CHARS = 32 * 1024 * 1024


def pre_tokenizer(
    corpus: str, special_tokens: list[str], word_freq: dict[tuple[bytes, ...], int]
):  # corpus -> chunk + special tokens -> token_seq s + special token
    if special_tokens:
        special_pat = "(" + "|".join(re.escape(tok) for tok in sorted(special_tokens, key=len, reverse=True)) + ")"
        corpus_with_specials: list[str] = re.split(special_pat, corpus)
    else:
        corpus_with_specials: list[str] = [corpus]

    for chunk in corpus_with_specials:
        if chunk in special_tokens:
            continue

        for match in re.finditer(PAT, chunk):
            piece: str = match.group(0)
            piece_bytes = piece.encode("utf-8")

            token_seq = tuple(BYTE_TOKENS[b] for b in piece_bytes)
            word_freq[token_seq] += 1
    return


def pretokenize_chunk(texts: list[str], special_tokens: list[str]) -> dict[tuple[bytes, ...], int]:
    word_freq: dict[tuple[bytes, ...], int] = defaultdict(int)
    for text in texts:
        pre_tokenizer(text, special_tokens, word_freq)
    return word_freq


def batched_special_token_chunks(
    input_path: str,
    special_tokens: list[str],
    target_chars: int = TARGET_CHARS,
) -> Iterator[list[str]]:
    special_tokens = sorted(special_tokens, key=len, reverse=True)
    special_pat = re.compile("(" + "|".join(re.escape(tok) for tok in special_tokens) + ")")
    batch: list[str] = []
    batch_chars = 0
    pending = ""

    with open(input_path, encoding="utf-8") as f:
        while True:
            block = f.read(READ_CHARS)
            if not block:
                break

            pieces = special_pat.split(pending + block)
            pending = pieces.pop()

            for piece in pieces:
                if not piece or piece in special_tokens:
                    continue

                batch.append(piece)
                batch_chars += len(piece)
                if batch_chars >= target_chars:
                    yield batch
                    batch = []
                    batch_chars = 0

            while len(pending) >= target_chars:
                split_at = pending.rfind("\n", 0, target_chars)
                if split_at < 0:
                    break

                batch.append(pending[: split_at + 1])
                batch_chars += split_at + 1
                pending = pending[split_at + 1 :]
                if batch_chars >= target_chars:
                    yield batch
                    batch = []
                    batch_chars = 0

    if pending:
        batch.append(pending)

    if batch:
        yield batch


def line_chunks(input_path: str, target_chars: int = TARGET_CHARS) -> Iterator[list[str]]:
    batch: list[str] = []
    batch_chars = 0

    with open(input_path, encoding="utf-8") as f:
        for line in f:
            batch.append(line)
            batch_chars += len(line)
            if batch_chars >= target_chars:
                yield batch
                batch = []
                batch_chars = 0

    if batch:
        yield batch


def iter_text_chunks(input_path: str, special_tokens: list[str]) -> Iterator[list[str]]:
    if special_tokens:
        yield from batched_special_token_chunks(input_path, special_tokens)
    else:
        yield from line_chunks(input_path)


def merge_word_freqs(
    word_freqs: Iterable[dict[tuple[bytes, ...], int]],
) -> dict[tuple[bytes, ...], int]:
    word_freq: dict[tuple[bytes, ...], int] = defaultdict(int)
    for partial_word_freq in word_freqs:
        for token_seq, freq in partial_word_freq.items():
            word_freq[token_seq] += freq
    return word_freq


def build_word_freq(
    input_path: str,
    special_tokens: list[str],
    num_workers: int | None = None,
) -> dict[tuple[bytes, ...], int]:
    if num_workers is None:
        input_size = os.path.getsize(input_path)
        num_workers = 1 if input_size < TARGET_CHARS else min(8, os.cpu_count() or 1)

    chunks = iter_text_chunks(input_path, special_tokens)
    if num_workers <= 1:
        return merge_word_freqs(pretokenize_chunk(chunk, special_tokens) for chunk in chunks)

    with Pool(num_workers) as pool:
        return merge_word_freqs(pool.starmap(pretokenize_chunk, ((chunk, special_tokens) for chunk in chunks)))


def iter_pairs(token_seq: tuple[bytes, ...]) -> Iterator[tuple[bytes, bytes]]:
    for i in range(len(token_seq) - 1):
        yield (token_seq[i], token_seq[i + 1])


PairHeap = list[tuple[int, tuple[bytes, bytes]]]


def update_pair_index(
    token_seq: tuple[bytes, ...],
    freq_delta: int,
    pair_counts: dict[tuple[bytes, bytes], int],
    pair_to_words: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]],
    pair_heap: PairHeap | None = None,
) -> None:
    pair_multiplicity: dict[tuple[bytes, bytes], int] = defaultdict(int)
    for pair in iter_pairs(token_seq):
        pair_multiplicity[pair] += 1

    for pair, occurrences in pair_multiplicity.items():
        if freq_delta > 0:
            pair_to_words[pair].add(token_seq)
        else:
            words = pair_to_words.get(pair)
            if words is not None:
                words.discard(token_seq)
                if not words:
                    pair_to_words.pop(pair, None)

        next_count = pair_counts.get(pair, 0) + freq_delta * occurrences
        if next_count > 0:
            pair_counts[pair] = next_count
            if pair_heap is not None:
                heapq.heappush(pair_heap, (-next_count, pair))
        else:
            pair_counts.pop(pair, None)


def build_pair_index(
    word_freq: dict[tuple[bytes, ...], int],
) -> tuple[
    dict[tuple[bytes, bytes], int],
    dict[tuple[bytes, bytes], set[tuple[bytes, ...]]],
    PairHeap,
]:
    pair_counts: dict[tuple[bytes, bytes], int] = defaultdict(int)
    pair_to_words: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]] = defaultdict(set)

    for token_seq, freq in word_freq.items():
        update_pair_index(token_seq, freq, pair_counts, pair_to_words)

    pair_heap = [(-count, pair) for pair, count in pair_counts.items()]
    heapq.heapify(pair_heap)
    return pair_counts, pair_to_words, pair_heap


def pop_best_pair(
    pair_counts: dict[tuple[bytes, bytes], int],
    pair_heap: PairHeap,
) -> tuple[bytes, bytes] | None:
    while pair_heap:
        neg_count, pair = heapq.heappop(pair_heap)
        count = -neg_count
        if pair_counts.get(pair, 0) != count:
            continue

        candidates = {pair}
        while pair_heap and pair_heap[0][0] == neg_count:
            _, tied_pair = heapq.heappop(pair_heap)
            if pair_counts.get(tied_pair, 0) == count:
                candidates.add(tied_pair)

        best_pair = max(candidates)
        for tied_pair in candidates:
            if tied_pair != best_pair:
                heapq.heappush(pair_heap, (neg_count, tied_pair))
        return best_pair
    return None


def merge_token_seq(
    token_seq: tuple[bytes, ...],
    best_pair: tuple[bytes, bytes],
    new_token: bytes,
) -> tuple[bytes, ...]:
    new_seq: list[bytes] = []
    i = 0

    while i < len(token_seq) - 1:
        if token_seq[i] == best_pair[0] and token_seq[i + 1] == best_pair[1]:
            new_seq.append(new_token)
            i += 2
        else:
            new_seq.append(token_seq[i])
            i += 1
    if i == len(token_seq) - 1:
        new_seq.append(token_seq[i])

    return tuple(new_seq)


def best_ranked_pair(
    token_seq: tuple[bytes, ...],
    merge_rank: dict[tuple[bytes, bytes], int],
) -> tuple[bytes, bytes] | None:
    best_pair: tuple[bytes, bytes] | None = None
    best_rank: int | None = None

    for pair in iter_pairs(token_seq):
        rank = merge_rank.get(pair)
        if rank is not None and (best_rank is None or rank < best_rank):
            best_pair = pair
            best_rank = rank

    return best_pair


def apply_ranked_merges(
    token_seq: tuple[bytes, ...],
    merge_rank: dict[tuple[bytes, bytes], int],
) -> tuple[bytes, ...]:
    while True:
        best_pair = best_ranked_pair(token_seq, merge_rank)
        if best_pair is None:
            return token_seq
        token_seq = merge_token_seq(token_seq, best_pair, best_pair[0] + best_pair[1])


def merge_pair_and_count_next(
    word_freq: dict[tuple[bytes, ...], int],
    pair_counts: dict[tuple[bytes, bytes], int],
    pair_to_words: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]],
    pair_heap: PairHeap,
    best_pair: tuple[bytes, bytes],
    new_token: bytes,
) -> None:
    affected_words = list(pair_to_words.get(best_pair, ()))
    pending_additions: dict[tuple[bytes, ...], int] = defaultdict(int)

    for token_seq in affected_words:
        freq = word_freq.pop(token_seq, 0)
        if freq == 0:
            continue

        update_pair_index(token_seq, -freq, pair_counts, pair_to_words, pair_heap)
        pending_additions[merge_token_seq(token_seq, best_pair, new_token)] += freq

    for token_seq, freq in pending_additions.items():
        word_freq[token_seq] += freq  # add the new token
        update_pair_index(token_seq, freq, pair_counts, pair_to_words, pair_heap)


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    merges: list[tuple[bytes, bytes]] = []

    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    for special_token in special_tokens:
        vocab[len(vocab)] = special_token.encode("utf-8")

    word_freq = build_word_freq(input_path, special_tokens)
    pair_counts, pair_to_words, pair_heap = build_pair_index(word_freq)

    while len(vocab) < vocab_size and pair_counts:
        best_pair = pop_best_pair(pair_counts, pair_heap)
        if best_pair is None:
            break

        new_token: bytes = best_pair[0] + best_pair[1]
        merges.append(best_pair)
        vocab[len(vocab)] = new_token

        merge_pair_and_count_next(word_freq, pair_counts, pair_to_words, pair_heap, best_pair, new_token)

    return (vocab, merges)
