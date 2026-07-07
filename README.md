# CS336 Spring 2025 Assignment 1: Basics

For a full description of the assignment, see the assignment handout at
[cs336_assignment1_basics.pdf](./cs336_assignment1_basics.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

### Environment
We manage our environments with `uv` to ensure reproducibility, portability, and ease of use.
Install `uv` [here](https://github.com/astral-sh/uv#installation) (recommended), or run `pip install uv`/`brew install uv`.
We recommend reading a bit about managing projects in `uv` [here](https://docs.astral.sh/uv/guides/projects/#managing-dependencies) (you will not regret it!).

You can now run any code in the repo using
```sh
uv run <python_file_path>
```
and the environment will be automatically solved and activated when necessary.

### Run unit tests


```sh
uv run pytest
```

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).

### Download data
Download the TinyStories data and a subsample of OpenWebText

``` sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

## TSUBAME training quick reference

Work from the on-demand desktop, not the login node:

```sh
cd /work/7/uw07387/CS336/assignment1-basics
```

Fast TinyStories iteration run:

```sh
./scripts/train_tinystories_fast_iter.sh
```

The training code enables the current speed path by default on CUDA:

- bf16 autocast on supported GPUs
- TF32 matmul/cuDNN
- PyTorch fused AdamW
- `torch.compile`
- fused scaled-dot-product attention

At startup it prints a line like:

```text
speed options: bf16=True, optimizer=AdamW, compiled=True, sdpa=True, tf32=True
```

To run a new experiment after one fast run finishes, use a new `RUN_NAME`.
Use a new `CHECKPOINT_PATH` if you want a separate checkpoint instead of
continuing from the existing baseline checkpoint.

Continue from the existing checkpoint with changed runtime parameters:

```sh
RUN_NAME=tinystories_fast_bs64_$(date +%Y%m%d_%H%M%S) \
BATCH_SIZE=64 \
MAX_ITERS=12000 \
LOG_EVERY=20 \
EVAL_EVERY=200 \
VAL_ITERS=5 \
./scripts/train_tinystories_fast_iter.sh
```

Start a separate new run from iteration 0:

```sh
RUN_NAME=tinystories_new_bs64_$(date +%Y%m%d_%H%M%S) \
CHECKPOINT_PATH=checkpoints/tinystories_new_bs64.pt \
BATCH_SIZE=64 \
MAX_ITERS=5000 \
./scripts/train_tinystories_fast_iter.sh
```

If a checkpoint already exists at `CHECKPOINT_PATH`, the script resumes it. If
the path does not exist, the script starts from scratch and writes a new
checkpoint there.

Safe parameters to change while resuming the same model:

```text
BATCH_SIZE
MAX_ITERS
LOG_EVERY
EVAL_EVERY
VAL_ITERS
CHECKPOINT_EVERY
RUN_NAME
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

To avoid `torch.compile` startup or memory overhead for a debugging run, pass
`--no-compile-model` in a manual `python -m cs336_basics.Train ...` command, or
remove `--compile-model` from the shell script temporarily.

More detailed TSUBAME notes live in
`cs336_basics/TSUBAME_TRAINING_GUIDE.md`.
