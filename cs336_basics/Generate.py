from __future__ import annotations

import argparse
import os
from typing import Any

import torch

from .Tokenizer import Tokenizer
from .Transformer import Transformer
from .Train import compile_model_if_enabled, normalize_model_state_dict


def top_p_sample(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> torch.Tensor:
    """Sample one token from each row of logits using nucleus sampling."""
    if temperature <= 0:
        raise ValueError("temperature must be greater than 0")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in the interval (0, 1]")

    probabilities = torch.softmax(logits / temperature, dim=-1)
    sorted_probabilities, sorted_indices = torch.sort(
        probabilities, dim=-1, descending=True
    )

    cumulative_probabilities = torch.cumsum(sorted_probabilities, dim=-1)

    # Remove a token only when the probability mass before it already reaches p.
    # This keeps the first token that makes the cumulative probability >= p.
    remove_mask = cumulative_probabilities - sorted_probabilities >= top_p
    sorted_probabilities = sorted_probabilities.masked_fill(remove_mask, 0.0)
    sorted_probabilities = sorted_probabilities / sorted_probabilities.sum(
        dim=-1, keepdim=True
    )

    sampled_sorted_index = torch.multinomial(sorted_probabilities, num_samples=1)
    return torch.gather(sorted_indices, dim=-1, index=sampled_sorted_index)


@torch.no_grad()
def generate(
    model: Transformer,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    context_length: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    eos_token_id: int | None = None,
) -> torch.Tensor:
    """Generate a completion for a batch of tokenized prompts."""
    if prompt_ids.ndim != 2:
        raise ValueError("prompt_ids must have shape (batch_size, sequence_length)")
    if prompt_ids.shape[1] == 0:
        raise ValueError("prompt_ids must contain at least one token")

    model.eval()
    token_ids = prompt_ids

    for _ in range(max_new_tokens):
        model_input = token_ids[:, -context_length:]
        logits = model(model_input)
        next_token_logits = logits[:, -1, :]
        next_token_id = top_p_sample(next_token_logits, temperature, top_p)
        token_ids = torch.cat((token_ids, next_token_id), dim=1)

        if eos_token_id is not None and torch.all(next_token_id == eos_token_id):
            break

    return token_ids


def load_model_checkpoint(
    checkpoint_path: str | os.PathLike,
    model: Transformer,
    device: str,
) -> dict[str, Any]:
    """Load the model state from a training checkpoint and return its metadata."""
    checkpoint: dict[str, Any] = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(normalize_model_state_dict(checkpoint["model"]))
    return checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate text from a trained CS336 Transformer."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--merges", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--eos-token", default="<|endoftext|>")
    parser.add_argument("--vocab-size", type=int)
    parser.add_argument("--context-length", type=int)
    parser.add_argument("--d-model", type=int)
    parser.add_argument("--num-layers", type=int)
    parser.add_argument("--num-heads", type=int)
    parser.add_argument("--d-ff", type=int)
    parser.add_argument("--rope-theta", type=float, default=10000.0)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--compile-model",
        dest="compile_model",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = Tokenizer.from_files(
        args.vocab,
        args.merges,
        special_tokens=[args.eos_token],
    )

    checkpoint: dict[str, Any] = torch.load(args.checkpoint, map_location=args.device)
    checkpoint_config = checkpoint.get("model_config", {})
    tokenizer_vocab_size = len(tokenizer.vocab)
    model_config = {
        "vocab_size": args.vocab_size
        or checkpoint_config.get("vocab_size")
        or tokenizer_vocab_size,
        "context_length": args.context_length
        or checkpoint_config.get("context_length"),
        "d_model": args.d_model or checkpoint_config.get("d_model"),
        "num_layers": args.num_layers or checkpoint_config.get("num_layers"),
        "num_heads": args.num_heads or checkpoint_config.get("num_heads"),
        "d_ff": args.d_ff or checkpoint_config.get("d_ff"),
        "rope_theta": checkpoint_config.get("rope_theta", args.rope_theta),
    }
    missing = [key for key, value in model_config.items() if value is None]
    if missing:
        raise ValueError(
            f"checkpoint has no model_config; provide: {', '.join(missing)}"
        )
    if model_config["vocab_size"] != tokenizer_vocab_size:
        raise ValueError(
            f"model vocab_size={model_config['vocab_size']} does not match tokenizer vocab_size={tokenizer_vocab_size}"
        )

    model = Transformer(**model_config).to(args.device)
    model.load_state_dict(normalize_model_state_dict(checkpoint["model"]))
    model = compile_model_if_enabled(model, args.compile_model, args.device)

    prompt_ids = tokenizer.encode(args.prompt)
    eos_token_id = tokenizer.bytes_to_id[args.eos_token.encode("utf-8")]
    input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=args.device)

    output_ids = generate(
        model,
        input_tensor,
        max_new_tokens=args.max_new_tokens,
        context_length=int(model_config["context_length"]),
        temperature=args.temperature,
        top_p=args.top_p,
        eos_token_id=eos_token_id,
    )
    print(tokenizer.decode(output_ids[0].tolist()))


if __name__ == "__main__":
    main()


"""

1. Download the raw text datasets:

    mkdir -p data artifacts checkpoints experiments
    wget -O data/TinyStoriesV2-GPT4-train.txt \
      https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
    wget -O data/TinyStoriesV2-GPT4-valid.txt \
      https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

2. Train and save a tokenizer:

    uv run python -m cs336_basics.Prepare train-tokenizer \
      --input data/TinyStoriesV2-GPT4-train.txt \
      --vocab-size 10000 \
      --vocab-out artifacts/vocab.json \
      --merges-out artifacts/merges.json

3. Encode text datasets into token ID arrays:

    uv run python -m cs336_basics.Prepare encode \
      --input data/TinyStoriesV2-GPT4-train.txt \
      --output artifacts/train.npy \
      --vocab artifacts/vocab.json \
      --merges artifacts/merges.json

    uv run python -m cs336_basics.Prepare encode \
      --input data/TinyStoriesV2-GPT4-valid.txt \
      --output artifacts/valid.npy \
      --vocab artifacts/vocab.json \
      --merges artifacts/merges.json

4. Train the language model and record an experiment:

    uv run python -m cs336_basics.Train \
      --train-data artifacts/train.npy \
      --val-data artifacts/valid.npy \
      --vocab-size 10000 \
      --context-length 256 \
      --d-model 512 \
      --num-layers 4 \
      --num-heads 16 \
      --d-ff 1344 \
      --batch-size 32 \
      --max-iters 10000 \
      --lr 0.001 \
      --min-lr 0.0001 \
      --warmup-iters 100 \
      --log-every 10 \
      --eval-every 100 \
      --val-iters 10 \
      --checkpoint-every 100 \
      --checkpoint-path checkpoints/tinystories.pt \
      --log-dir experiments \
      --run-name tinystories-baseline \
      --device cuda:0

    The experiment directory contains config.json, metrics.csv, loss_curves_steps.svg, and loss_curves_time.svg.
    Resume training by adding --resume-from checkpoints/tinystories.pt and a new run name.

5. Generate text. New checkpoints contain the model architecture configuration:

    uv run python -m cs336_basics.Generate \
      --checkpoint checkpoints/tinystories.pt \
      --vocab artifacts/vocab.json \
      --merges artifacts/merges.json \
      --prompt "Once upon a time" \
      --temperature 0.8 \
      --top-p 0.9 \
      --max-new-tokens 200 \
      --device cuda:0

Replace `python` with `pyinstrument` when profiling a command.
"""
