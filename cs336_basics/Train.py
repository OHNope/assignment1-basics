from dataclasses import asdict, dataclass
from typing import IO, BinaryIO

import argparse
import contextlib
import copy
import os
import numpy as np
import numpy.typing as npt
import torch

from .Transformer import Transformer
from .Experiment import ExperimentLogger
from .Optimizer import AdamW, cross_entropy, gradient_clipping, learning_rate_schedule


def data_loading(
    dataset: npt.NDArray,  # arrray with token ids
    batch_size: int,
    context_length: int,
    device: str,  # 'cpu' or 'cuda:0'
) -> tuple[
    torch.LongTensor, torch.LongTensor
]:  # (batch_size, context_length), (input_sequence, language modeling label)
    max_start = len(dataset) - context_length

    start_indices = np.random.randint(0, max_start, size=batch_size)

    x = np.stack([dataset[i : i + context_length] for i in start_indices])  # multi arrays = 2D tensor 
    # the current token

    y = np.stack([dataset[i + 1 : i + 1 + context_length] for i in start_indices])
    # the next token -> should be predicted by the large model 

    x = torch.from_numpy(x).to(device=device, dtype=torch.long)
    y = torch.from_numpy(y).to(device=device, dtype=torch.long)

    return x, y


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
    model_config: dict | None = None,
):
    model = _unwrap_compiled_model(model)
    obj = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }
    if model_config is not None:
        obj["model_config"] = model_config
    if isinstance(out, (str, os.PathLike)):
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    torch.save(obj, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:  # iteration num
    model = _unwrap_compiled_model(model)
    device = next(model.parameters()).device
    checkpoint = torch.load(src, map_location=device)
    model.load_state_dict(normalize_model_state_dict(checkpoint["model"]))
    optimizer_state = checkpoint["optimizer"]
    if isinstance(optimizer, torch.optim.AdamW):
        optimizer_state = _convert_custom_adamw_state(optimizer_state)
    elif isinstance(optimizer, AdamW):
        optimizer_state = _convert_torch_adamw_state(optimizer_state)
    optimizer.load_state_dict(optimizer_state)
    if isinstance(optimizer, torch.optim.AdamW):
        _ensure_torch_adamw_step_tensors(optimizer)
    return checkpoint["iteration"]


def _convert_custom_adamw_state(optimizer_state: dict) -> dict:
    converted = copy.deepcopy(optimizer_state)
    for state in converted.get("state", {}).values():
        if "m" in state and "v" in state:
            state["exp_avg"] = state.pop("m")
            state["exp_avg_sq"] = state.pop("v")
            step = state.get("step", 0)
            if not torch.is_tensor(step):
                state["step"] = torch.tensor(float(step), dtype=torch.float32)
    return converted


def _convert_torch_adamw_state(optimizer_state: dict) -> dict:
    converted = copy.deepcopy(optimizer_state)
    for state in converted.get("state", {}).values():
        if "exp_avg" in state and "exp_avg_sq" in state:
            state["m"] = state.pop("exp_avg")
            state["v"] = state.pop("exp_avg_sq")
            step = state.get("step", 0)
            if torch.is_tensor(step):
                state["step"] = int(step.item())
    return converted


def _ensure_torch_adamw_step_tensors(optimizer: torch.optim.Optimizer) -> None:
    for group in optimizer.param_groups:
        for param in group["params"]:
            state = optimizer.state.get(param)
            if not state or "step" not in state:
                continue
            if not torch.is_tensor(state["step"]):
                state["step"] = torch.tensor(float(state["step"]), dtype=torch.float32)


def _unwrap_compiled_model(model: torch.nn.Module) -> torch.nn.Module:
    """Return the original module when torch.compile wrapped it."""
    original = getattr(model, "_orig_mod", None)
    if isinstance(original, torch.nn.Module):
        return original
    return model


def normalize_model_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if not any(key.startswith("_orig_mod.") for key in state_dict):
        return state_dict
    return {
        key.removeprefix("_orig_mod."): value
        for key, value in state_dict.items()
    }


def compile_model_if_enabled(
    model: torch.nn.Module,
    enabled: bool,
    device: str,
) -> torch.nn.Module:
    if enabled and device.startswith("cuda"):
        return torch.compile(model)
    return model


"""
sampling from your dataset (i.e., a numpy array) during training, be sure to load the
dataset in memory-mapped mode (via np.memmap or the flag mmap_mode='r' to np.load, depending on
how you saved the array). """


@dataclass
class TrainConfig:
    train_data: str | os.PathLike
    vocab_size: int
    context_length: int
    d_model: int
    num_layers: int
    num_heads: int
    d_ff: int
    rope_theta: float = 10000.0
    batch_size: int = 32
    max_iters: int = 1000
    lr: float = 1e-3
    min_lr: float = 1e-4
    warmup_iters: int = 100
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    grad_clip: float | None = 1.0
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    val_data: str | os.PathLike | None = None
    val_iters: int = 10
    log_every: int = 10
    eval_every: int = 100
    checkpoint_every: int = 100
    checkpoint_path: str | os.PathLike | None = None
    resume_from: str | os.PathLike | None = None
    log_dir: str | os.PathLike = "experiments"
    run_name: str | None = None
    use_bf16: bool = True
    use_fused_adamw: bool = True
    compile_model: bool = True


def _load_dataset(path: str | os.PathLike) -> npt.NDArray:
    return np.load(path, mmap_mode="r")


@torch.no_grad()
def estimate_loss(
    model: torch.nn.Module,
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str,
    num_iters: int,
    use_bf16: bool = False,
) -> float:
    model.eval() # set the model setting to evaluate mode, no dropout & freeze the batchNorm !! 
    losses: list[float] = []
    for _ in range(num_iters):
        x, y = data_loading(dataset, batch_size, context_length, device)
        loss = _model_loss(model, x, y, device, use_bf16)
        losses.append(loss.item())
    model.train() # set the taining mode !! 
    return float(np.mean(losses))

# let the framework automatically determine which type to use: AMP
# optimize func
def _autocast_context(device: str, use_bf16: bool):
    enabled = use_bf16 and device.startswith("cuda")
    if enabled:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def _model_loss(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    device: str,
    use_bf16: bool,
) -> torch.Tensor:
    with _autocast_context(device, use_bf16):
        logits = model(x)
    return cross_entropy(logits.float().reshape(-1, logits.shape[-1]), y.reshape(-1))


def _build_optimizer(
    model: torch.nn.Module,
    config: TrainConfig,
) -> torch.optim.Optimizer:
    if config.use_fused_adamw and config.device.startswith("cuda"):
        try:
            return torch.optim.AdamW(
                model.parameters(),
                lr=config.lr,
                weight_decay=config.weight_decay,
                betas=config.betas,
                eps=config.eps,
                fused=True,
            )
        except TypeError:
            return torch.optim.AdamW(
                model.parameters(),
                lr=config.lr,
                weight_decay=config.weight_decay,
                betas=config.betas,
                eps=config.eps,
            )

    return AdamW(
        model.parameters(),  # return the all torch.nn.Parameter registered in model
        lr=config.lr,
        weight_decay=config.weight_decay,
        betas=config.betas,
        eps=config.eps,
    )


def train(config: TrainConfig) -> tuple[Transformer, torch.optim.Optimizer]:
    if config.device.startswith("cuda"):
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # get the dataset from the config
    train_dataset = _load_dataset(config.train_data)
    val_dataset = _load_dataset(config.val_data) if config.val_data is not None else None

# Initialize the every components
    model = Transformer(
        vocab_size=config.vocab_size,
        context_length=config.context_length,
        d_model=config.d_model,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
        rope_theta=config.rope_theta,
    ).to(config.device)

    use_bf16 = (
        config.use_bf16
        and config.device.startswith("cuda")
        and torch.cuda.is_bf16_supported()
    )
    optimizer = _build_optimizer(model, config)

    start_iter = 0
    if config.resume_from is not None:
        start_iter = load_checkpoint(config.resume_from, model, optimizer)

    if start_iter >= config.max_iters:
        print(
            f"checkpoint is already at iteration {start_iter}, "
            f"but max_iters is {config.max_iters}; no training or saving. "
            "Increase --max-iters to continue training."
        )
        return model, optimizer

    train_model: torch.nn.Module = model
    train_model = compile_model_if_enabled(model, config.compile_model, config.device)

    model_config = {
        key: value
        for key, value in asdict(config).items()
        if key
        in {
            "vocab_size",
            "context_length",
            "d_model",
            "num_layers",
            "num_heads",
            "d_ff",
            "rope_theta",
        }
    }

    logger = ExperimentLogger(config.log_dir, config.run_name, config)
    print(f"experiment logs: {logger.run_dir}")

    print(
        "speed options: "
        f"bf16={use_bf16}, "
        f"optimizer={type(optimizer).__name__}, "
        f"compiled={train_model is not model}, "
        "sdpa=True, tf32=True"
    )

# start training.
    train_model.train()  # Dropout ,BatchNorm on. the Trianing mode 
    last_completed_iter = start_iter
    for iteration in range(start_iter, config.max_iters):
        lr = learning_rate_schedule(
            iteration,
            config.lr,
            config.min_lr,
            config.warmup_iters,
            config.max_iters,
        )
        # for every components in the model, how to update all of the learnable params registered in nn.Parameters
        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y = data_loading(
            train_dataset,
            config.batch_size,
            config.context_length,
            config.device,
        )
        loss = _model_loss(train_model, x, y, config.device, use_bf16)

        optimizer.zero_grad(set_to_none=True)  # no write zero, free the mem.Next step: if None -> write the grad
        loss.backward() # calc the gradients
        if config.grad_clip is not None:
            gradient_clipping(model.parameters(), config.grad_clip)
        optimizer.step() # use the gradients calculated -> update the parameters

# precalc for the logger , debugger
        completed_iter = iteration + 1
        last_completed_iter = completed_iter
        should_log = completed_iter % config.log_every == 0
        should_eval = val_dataset is not None and completed_iter % config.eval_every == 0
        val_loss: float | None = None
        if should_eval:
            val_loss = estimate_loss(
                train_model,
                val_dataset,
                config.batch_size,
                config.context_length,
                config.device,
                config.val_iters,
                use_bf16,
            )

        # logger, debugger part
        if should_log or should_eval:
            if config.device.startswith("cuda"):
                torch.cuda.synchronize(config.device)
            train_loss = loss.item()
            tokens_seen = completed_iter * config.batch_size * config.context_length
            logger.log(
                step=completed_iter,
                tokens_seen=tokens_seen,
                learning_rate=lr,
                train_loss=train_loss,
                val_loss=val_loss,
            )
            message = (
                f"iter {completed_iter}: train loss {train_loss:.4f}, "
                f"lr {lr:.6g}, elapsed {logger.elapsed_seconds():.1f}s"
            )
            if val_loss is not None:
                message += f", val loss {val_loss:.4f}"
            print(message)

        # auto save data/params
        if config.checkpoint_path is not None and completed_iter % config.checkpoint_every == 0:
            save_checkpoint(model, optimizer, completed_iter, config.checkpoint_path, model_config)

    # final save
    if config.checkpoint_path is not None:
        save_checkpoint(model, optimizer, last_completed_iter, config.checkpoint_path, model_config)

    return model, optimizer


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train a CS336 Transformer language model.")
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--val-data")
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--context-length", type=int, required=True)
    parser.add_argument("--d-model", type=int, required=True)
    parser.add_argument("--num-layers", type=int, required=True)
    parser.add_argument("--num-heads", type=int, required=True)
    parser.add_argument("--d-ff", type=int, required=True)
    parser.add_argument("--rope-theta", type=float, default=10000.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-iters", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min-lr", type=float, default=1e-4)
    parser.add_argument("--warmup-iters", type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--val-iters", type=int, default=10)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--resume-from")
    parser.add_argument("--log-dir", default="experiments")
    parser.add_argument("--run-name")
    parser.add_argument("--bf16", dest="use_bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fused-adamw", dest="use_fused_adamw", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compile-model", dest="compile_model", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    return TrainConfig(**vars(args))


if __name__ == "__main__":
    train(parse_args())

"""
uv run python -m cs336_basics.Train \
    --train-data path/to/train.npy \
    --val-data path/to/valid.npy \
    --vocab-size 10000 \
    --context-length 256 \
    --d-model 512 \
    --num-layers 4 \
    --num-heads 16 \
    --d-ff 1344 \
    --batch-size 32 \
    --max-iters 10000 \
    --checkpoint-path checkpoints/latest.pt \
    --device gpu
"""
