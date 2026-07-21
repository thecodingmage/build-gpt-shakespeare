"""
Shared hyperparameters for gpt.py.

Two profiles: a small one to iterate on quickly on a laptop/CPU, and a
scaled-up one for an actual training run on the eehpc cluster GPU.
Video Section 21 covers the "scaling up" jump between these two regimes.
"""

import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

# --- local/laptop-scale config (fast iteration, sanity checks) ---
LOCAL = dict(
    batch_size=32,
    block_size=8,
    max_iters=5000,
    eval_interval=500,
    learning_rate=1e-3,
    eval_iters=200,
    n_embd=32,
    n_head=4,
    n_layer=4,
    dropout=0.0,
)

# --- cluster-scale config (matches the video's final "scaled up" model) ---
CLUSTER = dict(
    batch_size=64,
    block_size=256,
    max_iters=5000,
    eval_interval=500,
    learning_rate=3e-4,
    eval_iters=200,
    n_embd=384,
    n_head=6,
    n_layer=6,
    dropout=0.2,
)
