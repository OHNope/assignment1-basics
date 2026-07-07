from __future__ import annotations

from collections.abc import Iterable, Iterator
import json
import os
import regex as re

from cs336_basics.BPE import BYTE_TOKENS, PAT, iter_pairs, merge_token_seq


class Tokenizer:
    vocab: dict[int, bytes]

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = dict(vocab)
        self.bytes_to_id: dict[bytes, int] = {
            token_bytes: token_id for token_id, token_bytes in self.vocab.items()
        }
        self.merge_rank = {pair: rank for rank, pair in enumerate(merges)}

        self.special_tokens = sorted(special_tokens or [], key=len, reverse=True)
        for token in self.special_tokens:  # add special tokens into the map/vocab
            token_bytes = token.encode("utf-8")
            if token_bytes not in self.bytes_to_id:
                token_id = len(self.vocab)
                self.vocab[token_id] = token_bytes
                self.bytes_to_id[token_bytes] = token_id

        self.special_pattern = None  # compile the pattern to accelerate the speed
        if self.special_tokens:
            self.special_pattern = re.compile(
                "(" + "|".join(re.escape(token) for token in self.special_tokens) + ")"
            )

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ) -> Tokenizer:
        with open(vocab_filepath, encoding="utf-8") as vocab_file:
            raw_vocab = json.load(vocab_file)

        if raw_vocab.get("format") == "cs336_bytes_hex_v1":
            vocab = {
                int(token_id): bytes.fromhex(token_hex)
                for token_id, token_hex in raw_vocab["vocab"].items()
            }
        else:
            vocab = {
                int(token_id): token_bytes.encode("latin-1")
                for token_id, token_bytes in raw_vocab.items()
            }

        with open(merges_filepath, encoding="utf-8") as merges_file:
            raw_merges = json.load(merges_file) if os.fspath(merges_filepath).endswith(".json") else None

        if isinstance(raw_merges, dict) and raw_merges.get("format") == "cs336_bytes_hex_v1":
            merges = [
                (bytes.fromhex(left), bytes.fromhex(right))
                for left, right in raw_merges["merges"]
            ]
        else:
            merges = []
            with open(merges_filepath, encoding="utf-8") as merges_file:
                for line in merges_file:
                    parts = line.rstrip("\n").split(" ")
                    if len(parts) == 2:
                        merges.append(
                            (parts[0].encode("latin-1"), parts[1].encode("latin-1"))
                        )

        return cls(vocab, merges, special_tokens)

    def save(self, vocab_filepath: str, merges_filepath: str) -> None:
        vocab_data = {
            "format": "cs336_bytes_hex_v1",
            "vocab": {str(token_id): token_bytes.hex() for token_id, token_bytes in self.vocab.items()},
        }
        merges_data = {
            "format": "cs336_bytes_hex_v1",
            "merges": [[left.hex(), right.hex()] for left, right in self.merge_rank],
        }
        with open(vocab_filepath, "w", encoding="utf-8") as vocab_file:
            json.dump(vocab_data, vocab_file)
        with open(merges_filepath, "w", encoding="utf-8") as merges_file:
            json.dump(merges_data, merges_file)

    def _split_by_special_tokens(self, text: str) -> list[str]:
        if self.special_pattern is None:
            return [text]
        return self.special_pattern.split(text)

    def _apply_bpe(self, text: str) -> list[bytes]:
        tokens = tuple(BYTE_TOKENS[byte] for byte in text.encode("utf-8"))

        while True:
            best_pair: tuple[bytes, bytes] | None = None
            best_rank: int | None = None

            for pair in iter_pairs(tokens):
                rank = self.merge_rank.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_pair = pair
                    best_rank = rank

            if best_pair is None:
                return list(tokens)

            tokens = merge_token_seq(tokens, best_pair, best_pair[0] + best_pair[1])

    def _encode_chunk(self, text: str) -> list[int]:
        token_ids: list[int] = []
        for match in re.finditer(PAT, text):
            for token in self._apply_bpe(match.group(0)):
                token_ids.append(self.bytes_to_id[token])
        return token_ids

    def encode(self, text: str) -> list[int]:
        token_ids: list[int] = []
        for chunk in self._split_by_special_tokens(text):
            if not chunk:
                continue
            if chunk in self.special_tokens:
                token_ids.append(self.bytes_to_id[chunk.encode("utf-8")])
            else:
                token_ids.extend(self._encode_chunk(chunk))
        return token_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        token_bytes = b"".join(self.vocab[token_id] for token_id in ids)
        return token_bytes.decode("utf-8", errors="replace")


# TODO: tokenizer_experiments
