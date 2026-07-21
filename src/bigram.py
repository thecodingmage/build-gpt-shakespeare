"""
Bigram baseline language model.
Corresponds to video Sections 5-7 (Simplest Baseline -> Training -> Porting
to a Script).

This is the "does the training loop even work" sanity check before building
up to full self-attention in gpt.py. Rewrite this yourself from the video --
don't copy from ng-video-lecture, that defeats the point.

TODO:
- [ ] Load data/input.txt, build char-level vocab (stoi/itos)
- [ ] Train/val split (~90/10)
- [ ] get_batch(split) -> (xb, yb)
- [ ] BigramLanguageModel(nn.Module): a single nn.Embedding(vocab_size, vocab_size)
      as the "lookup table" of next-char logits
- [ ] forward() computes logits + cross_entropy loss
- [ ] generate() for autoregressive sampling
- [ ] training loop: AdamW, loss.backward(), track train/val loss
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters -- keep these here for this file only;
# gpt.py pulls shared ones from model_config.py instead
batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200

torch.manual_seed(1337)

# TODO: everything below


if __name__ == "__main__":
    pass
