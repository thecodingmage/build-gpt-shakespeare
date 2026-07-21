"""
Full decoder-only GPT, built up piece by piece.
Corresponds to video Sections 8-21.

Rewrite this yourself section by section -- use model_config.py for
hyperparameters (swap LOCAL <-> CLUSTER depending on where you're running).

TODO, in the order the video builds it:
- [ ] Head(nn.Module): single self-attention head
      (key, query, value projections; scaled dot-product; causal mask via
      torch.tril + masked_fill; softmax; weighted aggregation of values)
- [ ] MultiHeadAttention(nn.Module): run several Heads in parallel, concat,
      project back down
- [ ] FeedForward(nn.Module): Linear -> ReLU -> Linear (Section 18)
- [ ] Block(nn.Module): MultiHeadAttention + FeedForward, each wrapped in a
      residual connection + LayerNorm (pre-norm formulation) (Sections 19-20)
- [ ] GPTLanguageModel(nn.Module):
      - token embedding table + positional embedding table
      - stack of Blocks (n_layer of them)
      - final LayerNorm + linear head to vocab_size logits
      - forward() with cross_entropy loss
      - generate() for autoregressive sampling, respecting block_size context
- [ ] Dropout in attention + feedforward + embeddings (Section 21)
- [ ] Training loop: AdamW, periodic train/val loss estimation, checkpoint
      saving to ../checkpoints/ (this is what should be gitignored and moved
      to/from the cluster over SFTP, not git)
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

from model_config import LOCAL, CLUSTER, device

torch.manual_seed(1337)

# pick a config depending on where this runs
cfg = CLUSTER if device == "cuda" else LOCAL

# TODO: everything below


if __name__ == "__main__":
    pass
