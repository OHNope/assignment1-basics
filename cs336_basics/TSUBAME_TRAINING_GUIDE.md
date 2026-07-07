# CS336 assignment1-basics: tokenizer, training, and tuning on on-demand desktop

This note is for:

```sh
cd /work/7/uw07387/CS336/assignment1-basics
```

The Python package lives in `cs336_basics/`. I checked the repository:

- `uv` is available at `/home/7/uw07387/.local/bin/uv`.
- `.venv/bin/python` is Python 3.13.13.
- `import cs336_basics, torch` works.
- Full tests pass: `48 passed, 1 xfailed`.
- Login node CUDA is not usable: `torch.cuda.is_available()` is `False`.
- On your on-demand desktop, use the H100 MIG device directly with `--device cuda:0`.

Do not train on the login node. Run tokenizer encoding and training from the on-demand desktop when possible.

## Current state

As of the latest remote check, TinyStories preprocessing is already ready:

```text
artifacts/tinystories_vocab_10k.json
artifacts/tinystories_merges_10k.json
artifacts/tinystories_train_10k.npy
artifacts/tinystories_valid_10k.npy
```

You can start the baseline training directly from the on-demand desktop:

```sh
cd /work/7/uw07387/CS336/assignment1-basics
./scripts/train_tinystories_resume.sh
```

The latest checked `checkpoints/tinystories_baseline.pt` checkpoint was valid
at iteration `10000`.

Available training scripts:

```text
scripts/train_tinystories_resume.sh     continue from checkpoints/tinystories_baseline.pt
scripts/train_tinystories_fresh.sh      archive the old checkpoint and start from scratch
scripts/train_tinystories_fast_iter.sh  use batch_size=32 and less frequent eval/log for higher iter/sec
scripts/train_tinystories_desktop.sh    older auto-resume entrypoint; prefer the explicit scripts above
```

All three explicit scripts support environment-variable overrides. Example:

```sh
BATCH_SIZE=64 MAX_ITERS=12000 ./scripts/train_tinystories_resume.sh
```

Only rerun tokenizer training or encode if you change the dataset, vocabulary size, tokenizer files, or output `.npy` paths.

Speed note: compare throughput by tokens/sec, not only iter/sec.

```text
tokens/sec = iter/sec * batch_size * context_length
```

For example, `batch_size=32` at `10 iter/sec` and `batch_size=128` at
`2.5 iter/sec` are both about `81,920 tokens/sec`.

If you specifically want the displayed iteration rate to be higher, use:

```sh
./scripts/train_tinystories_fast_iter.sh
```

That script defaults to `batch_size=32`, `log_every=100`, `eval_every=1000`, and
`val_iters=3`. It should show faster iter/sec, but it may not improve tokens/sec.
By default it saves the trained model checkpoint to
`checkpoints/tinystories_baseline.pt` and writes experiment metrics under
`experiments/<RUN_NAME>/`.

For the exact continue-training and finetuning commands, see section 5.

## 0. On-demand desktop setup

Your on-demand desktop has:

- CPU cores: `nproc=48`
- GPU: NVIDIA H100 MIG 3g.47gb

Use 48 CPU threads for preprocessing and dataloading-related native libraries:

```sh
cd /work/7/uw07387/CS336/assignment1-basics
export OMP_NUM_THREADS=48
export OPENBLAS_NUM_THREADS=48
export MKL_NUM_THREADS=48
export NUMEXPR_NUM_THREADS=48
export TOKENIZERS_PARALLELISM=false
```

Check CUDA from the on-demand desktop:

```sh
UV_NO_SYNC=1 uv run python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY
```

If `cuda_available` is not `True`, fix the desktop environment before training.

## 1. Recommended directory layout

Run this once:

```sh
cd /work/7/uw07387/CS336/assignment1-basics
mkdir -p artifacts checkpoints experiments scripts
```

Keep large outputs under `/work/7/uw07387/CS336/assignment1-basics`, not under `/home`, because your home quota is smaller.

## 2. Quick sanity checks

Run:

```sh
cd /work/7/uw07387/CS336/assignment1-basics
UV_NO_SYNC=1 uv run pytest -q
```

Expected result from my check:

```text
48 passed, 1 xfailed
```

## 3. Train the tokenizer

The project provides:

```sh
uv run python -m cs336_basics.Prepare train-tokenizer ...
```

TinyStories baseline:

```sh
cd /work/7/uw07387/CS336/assignment1-basics
UV_NO_SYNC=1 uv run python -m cs336_basics.Prepare train-tokenizer \
  --input data/TinyStoriesV2-GPT4-train.txt \
  --vocab-size 10000 \
  --vocab-out artifacts/tinystories_vocab_10k.json \
  --merges-out artifacts/tinystories_merges_10k.json \
  --special-token '<|endoftext|>'
```

OpenWebText baseline:

```sh
cd /work/7/uw07387/CS336/assignment1-basics
UV_NO_SYNC=1 uv run python -m cs336_basics.Prepare train-tokenizer \
  --input data/owt_train.txt \
  --vocab-size 32000 \
  --vocab-out artifacts/owt_vocab_32k.json \
  --merges-out artifacts/owt_merges_32k.json \
  --special-token '<|endoftext|>'
```

Tokenizer training is CPU-heavy. It can use multiple CPU threads indirectly through native libraries, but this BPE implementation is mostly Python-side.

## 4. Encode datasets to `.npy`

`encode` supports CPU multiprocessing via `--num-workers`. On your on-demand
desktop with `nproc=48`, start with `--num-workers 48`. If the filesystem feels
sluggish or memory pressure rises, reduce it to `24` or `32`.

TinyStories:

```sh
cd /work/7/uw07387/CS336/assignment1-basics
UV_NO_SYNC=1 uv run python -m cs336_basics.Prepare encode \
  --input data/TinyStoriesV2-GPT4-train.txt \
  --output artifacts/tinystories_train_10k.npy \
  --vocab artifacts/tinystories_vocab_10k.json \
  --merges artifacts/tinystories_merges_10k.json \
  --special-token '<|endoftext|>' \
  --chunk-size 1000000 \
  --num-workers 48

UV_NO_SYNC=1 uv run python -m cs336_basics.Prepare encode \
  --input data/TinyStoriesV2-GPT4-valid.txt \
  --output artifacts/tinystories_valid_10k.npy \
  --vocab artifacts/tinystories_vocab_10k.json \
  --merges artifacts/tinystories_merges_10k.json \
  --special-token '<|endoftext|>' \
  --chunk-size 1000000 \
  --num-workers 48
```

OWT:

```sh
cd /work/7/uw07387/CS336/assignment1-basics
UV_NO_SYNC=1 uv run python -m cs336_basics.Prepare encode \
  --input data/owt_train.txt \
  --output artifacts/owt_train_32k.npy \
  --vocab artifacts/owt_vocab_32k.json \
  --merges artifacts/owt_merges_32k.json \
  --special-token '<|endoftext|>' \
  --chunk-size 1000000 \
  --num-workers 48

UV_NO_SYNC=1 uv run python -m cs336_basics.Prepare encode \
  --input data/owt_valid.txt \
  --output artifacts/owt_valid_32k.npy \
  --vocab artifacts/owt_vocab_32k.json \
  --merges artifacts/owt_merges_32k.json \
  --special-token '<|endoftext|>' \
  --chunk-size 1000000 \
  --num-workers 48
```

The training code loads these arrays with `np.load(..., mmap_mode="r")`, so it does not need to read the whole token array into RAM.

## 5. Train directly on the on-demand desktop

The shortest path is:

```sh
cd /work/7/uw07387/CS336/assignment1-basics
./scripts/train_tinystories_resume.sh
```

The resume script checks CUDA first, requires `checkpoints/tinystories_baseline.pt`,
then runs the baseline command below with `--resume-from`. It creates a fresh
timestamped experiment directory each time so `metrics.csv` is not overwritten.

To intentionally discard the current training run and start over:

```sh
./scripts/train_tinystories_fresh.sh
```

This does not delete `data/*.txt`, `artifacts/*.npy`, vocab, or merges. It moves
the old checkpoint under `checkpoints/archive/TIMESTAMP/`, then starts from
iteration 0.

To prioritize higher iter/sec:

```sh
./scripts/train_tinystories_fast_iter.sh
```

Default output locations:

```text
model checkpoint: checkpoints/tinystories_baseline.pt
metrics/logs:      experiments/<RUN_NAME>/metrics.csv and loss curves
```

`fast_iter` does not create a new checkpoint filename automatically. If
`CHECKPOINT_PATH` is not set, every run uses
`checkpoints/tinystories_baseline.pt`. The checkpoint file contains model
weights, optimizer state, the completed iteration number, and model config.

Current checked state:

```text
checkpoints/tinystories_baseline.pt
iteration: 10000
model_config: vocab_size=10000, context_length=256, d_model=512,
              num_layers=4, num_heads=16, d_ff=1344, rope_theta=10000.0
```

After one fast run finishes, start the next experiment by changing environment
variables before the script command. Always give the experiment a new
`RUN_NAME`; use a new `CHECKPOINT_PATH` when you want a separate checkpoint
instead of overwriting the baseline checkpoint.

Continue from the current checkpoint with changed runtime or finetuning
parameters:

```sh
RUN_NAME=tinystories_fast_ft_12k_$(date +%Y%m%d_%H%M%S) \
MAX_ITERS=13000 \
BATCH_SIZE=32 \
LR=0.003 \
MIN_LR=0.00003 \
LOG_EVERY=50 \
EVAL_EVERY=500 \
VAL_ITERS=5 \
./scripts/train_tinystories_fast_iter.sh
```

Because the current checkpoint is at iteration `10000`, `MAX_ITERS=12000` runs
2000 additional iterations. `--max-iters` is the final iteration number, not
"additional steps". The command above resumes from and then overwrites
`checkpoints/tinystories_baseline.pt`.

If `MAX_ITERS` is less than or equal to the checkpoint iteration, training now
exits without saving:

```text
checkpoint is already at iteration 10000, but max_iters is 2000; no training or saving.
```

This protects the checkpoint from accidentally being rewritten with a smaller
iteration number. Increase `MAX_ITERS` above the current checkpoint iteration
to continue training.

If you want to keep the current checkpoint before finetuning:

```sh
cp checkpoints/tinystories_baseline.pt checkpoints/tinystories_baseline_iter10000.pt

RUN_NAME=tinystories_fast_ft_12k_$(date +%Y%m%d_%H%M%S) \
MAX_ITERS=12000 \
LR=0.0003 \
MIN_LR=0.00003 \
./scripts/train_tinystories_fast_iter.sh
```

Start an independent new run from iteration 0:

```sh
RUN_NAME=tinystories_new_bs64_$(date +%Y%m%d_%H%M%S) \
CHECKPOINT_PATH=checkpoints/tinystories_new_bs64.pt \
BATCH_SIZE=64 \
MAX_ITERS=5000 \
./scripts/train_tinystories_fast_iter.sh
```

The fast script resumes if `CHECKPOINT_PATH` already exists. It starts from
scratch if that path does not exist.

To finetune into a separate checkpoint file while keeping the baseline file
untouched, first copy the baseline checkpoint, then point `CHECKPOINT_PATH` at
the copy:

```sh
cp checkpoints/tinystories_baseline.pt checkpoints/tinystories_ft_lr3e-4.pt

RUN_NAME=tinystories_ft_lr3e-4_$(date +%Y%m%d_%H%M%S) \
CHECKPOINT_PATH=checkpoints/tinystories_ft_lr3e-4.pt \
MAX_ITERS=12000 \
LR=0.0003 \
MIN_LR=0.00003 \
./scripts/train_tinystories_fast_iter.sh
```

In this case the script sees that `checkpoints/tinystories_ft_lr3e-4.pt`
already exists, resumes from it, and saves later checkpoints back to the same
file.

Safe parameters to change while resuming the same model:

```text
BATCH_SIZE
MAX_ITERS
LR
MIN_LR
LOG_EVERY
EVAL_EVERY
VAL_ITERS
CHECKPOINT_EVERY
RUN_NAME
CHECKPOINT_PATH, if it points to a copied compatible checkpoint
```

Model-shape parameters must match the checkpoint:

```text
vocab_size
context_length
d_model
num_layers
num_heads
d_ff
rope_theta
```

For short debugging runs, reduce compile overhead by running a manual training
command with `--no-compile-model`, or remove `--compile-model` from the script
temporarily.

First verify that both token arrays exist:

```sh
cd /work/7/uw07387/CS336/assignment1-basics
ls -lh artifacts/tinystories_train_10k.npy artifacts/tinystories_valid_10k.npy
```

Then train:

```sh
cd /work/7/uw07387/CS336/assignment1-basics

export OMP_NUM_THREADS=48
export OPENBLAS_NUM_THREADS=48
export MKL_NUM_THREADS=48
export NUMEXPR_NUM_THREADS=48
export TOKENIZERS_PARALLELISM=false

UV_NO_SYNC=1 uv run python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available on this on-demand desktop")
print("device:", torch.cuda.get_device_name(0))
PY

UV_NO_SYNC=1 uv run python -m cs336_basics.Train \
  --train-data artifacts/tinystories_train_10k.npy \
  --val-data artifacts/tinystories_valid_10k.npy \
  --vocab-size 10000 \
  --context-length 256 \
  --d-model 512 \
  --num-layers 4 \
  --num-heads 16 \
  --d-ff 1344 \
  --batch-size 128 \
  --max-iters 10000 \
  --lr 0.001 \
  --min-lr 0.0001 \
  --warmup-iters 100 \
  --weight-decay 0.01 \
  --grad-clip 1.0 \
  --log-every 10 \
  --eval-every 100 \
  --val-iters 10 \
  --checkpoint-every 500 \
  --checkpoint-path checkpoints/tinystories_baseline.pt \
  --log-dir experiments \
  --run-name tinystories_baseline \
  --bf16 \
  --fused-adamw \
  --compile-model \
  --device cuda:0
```

For H100 MIG 3g.47gb, this baseline should fit. If it OOMs, reduce `--batch-size` first:

```sh
--batch-size 64
```

## 6. Resume training

The checkpoint contains model weights, optimizer state, iteration, and model config. Prefer:

```sh
./scripts/train_tinystories_resume.sh
```

If you run the command manually, include `--resume-from checkpoints/tinystories_baseline.pt`.

```sh
UV_NO_SYNC=1 uv run python -m cs336_basics.Train \
  --train-data artifacts/tinystories_train_10k.npy \
  --val-data artifacts/tinystories_valid_10k.npy \
  --vocab-size 10000 \
  --context-length 256 \
  --d-model 512 \
  --num-layers 4 \
  --num-heads 16 \
  --d-ff 1344 \
  --batch-size 128 \
  --max-iters 20000 \
  --lr 0.001 \
  --min-lr 0.0001 \
  --warmup-iters 100 \
  --checkpoint-path checkpoints/tinystories_baseline.pt \
  --resume-from checkpoints/tinystories_baseline.pt \
  --log-dir experiments \
  --run-name tinystories_resume_20k \
  --bf16 \
  --fused-adamw \
  --compile-model \
  --device cuda:0
```

Important: `--max-iters` is the final iteration number, not "additional steps". If the checkpoint is at 7000 and you set `--max-iters 10000`, it will run iterations 7000 through 9999.

Ctrl-C does not damage `data/*.txt`, `artifacts/*.npy`, or tokenizer files. It can interrupt a checkpoint write if it happens at exactly the wrong time, so verify the checkpoint after stopping:

```sh
CUDA_VISIBLE_DEVICES="" UV_NO_SYNC=1 uv run python - <<'PY'
import torch
ckpt = torch.load("checkpoints/tinystories_baseline.pt", map_location="cpu")
print("iteration:", ckpt["iteration"])
PY
```

## 7. Generate text from a checkpoint

```sh
cd /work/7/uw07387/CS336/assignment1-basics
UV_NO_SYNC=1 uv run python -m cs336_basics.Generate \
  --checkpoint checkpoints/tinystories_baseline.pt \
  --vocab artifacts/tinystories_vocab_10k.json \
  --merges artifacts/tinystories_merges_10k.json \
  --prompt "Once upon a time" \
  --temperature 0.8 \
  --top-p 0.9 \
  --max-new-tokens 200 \
  --device cuda:0
```

For quick inspection, generation can run on CPU by changing `--device cpu`, but GPU is faster.

## 8. Tuning order

Start from this baseline:

```text
vocab_size=10000
context_length=256
d_model=512
num_layers=4
num_heads=16
d_ff=1344
batch_size=128
lr=1e-3
min_lr=1e-4
warmup_iters=100
weight_decay=0.01
grad_clip=1.0
```

Tune in this order:

1. First make the run stable: no CUDA OOM, loss decreases, checkpoint and metrics are written.
2. Increase `max_iters`; a short run only checks plumbing, not quality.
3. Tune learning rate. Try `3e-4`, `6e-4`, `1e-3`, and `2e-3`. If loss spikes or becomes NaN, lower it.
4. Tune batch size. Larger batches are smoother but use more VRAM. If OOM, halve `--batch-size`.
5. Tune context length. `256` is cheap; `512` improves longer dependencies but increases attention memory roughly quadratically.
6. Tune model size. Increase `d_model`, `num_layers`, and `d_ff` only after the smaller baseline behaves well.
7. Tune tokenizer vocabulary. TinyStories often works with `10000`; OWT usually wants `32000` or larger.

Reasonable next configurations:

TinyStories larger:

```text
vocab_size=10000
context_length=512
d_model=768
num_layers=6
num_heads=12
d_ff=2048
batch_size=16
lr=6e-4
```

OWT starter:

```text
vocab_size=32000
context_length=256
d_model=512
num_layers=6
num_heads=8
d_ff=2048
batch_size=16
lr=6e-4
```

If `d_model` is not divisible by `num_heads`, the attention implementation will fail. Keep `d_model / num_heads` even because RoPE works on pairs of dimensions.

## 9. What to inspect after every run

Each run writes:

```text
experiments/RUN_NAME/config.json
experiments/RUN_NAME/metrics.csv
experiments/RUN_NAME/loss_curves_steps.svg
experiments/RUN_NAME/loss_curves_time.svg
checkpoints/*.pt
```

Check the metrics:

```sh
sed -n '1,20p' experiments/tinystories_baseline/metrics.csv
tail -n 20 experiments/tinystories_baseline/metrics.csv
```

What to look for:

- `train_loss` should decrease early.
- `val_loss` should decrease, then eventually flatten.
- If train loss decreases but val loss gets worse, reduce model size, train less, increase weight decay, or use more data.
- If both losses are flat, increase LR a little or check that the data/tokenizer match the `vocab_size`.
- If loss is NaN, lower LR, keep `grad_clip=1.0`, and verify no corrupted data.

## 10. Current caveats in this repo

- The package and tests are usable.
- The typo `Normalizaiton.py` is intentional in imports; do not rename it unless you update all imports.
- `cs336_basics.Train` uses bf16 autocast on supported CUDA GPUs, TF32, fused AdamW, `torch.compile`, and fused SDPA by default. It does not implement gradient accumulation. If you hit OOM, reduce `batch_size`, `context_length`, model size, or disable compile for a debugging run.
- `--device gpu` is wrong for PyTorch here. Use `--device cuda:0`.
- The login node warning about CUDA is expected. Check CUDA from the on-demand desktop instead.
