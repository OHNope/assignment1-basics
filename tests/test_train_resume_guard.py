import numpy as np
import torch

from cs336_basics.Train import TrainConfig, save_checkpoint, train
from cs336_basics.Transformer import Transformer


def test_resume_with_lower_max_iters_does_not_rewrite_checkpoint(tmp_path):
    train_data = tmp_path / "train.npy"
    np.save(train_data, np.arange(64, dtype=np.int64) % 16)

    model_config = {
        "vocab_size": 16,
        "context_length": 4,
        "d_model": 8,
        "num_layers": 1,
        "num_heads": 2,
        "d_ff": 16,
        "rope_theta": 10000.0,
    }
    model = Transformer(**model_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    checkpoint_path = tmp_path / "checkpoint.pt"
    save_checkpoint(model, optimizer, 3, checkpoint_path, model_config)

    config = TrainConfig(
        train_data=train_data,
        checkpoint_path=checkpoint_path,
        resume_from=checkpoint_path,
        max_iters=2,
        batch_size=2,
        device="cpu",
        use_bf16=False,
        use_fused_adamw=False,
        compile_model=False,
        **model_config,
    )

    train(config)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    assert checkpoint["iteration"] == 3
