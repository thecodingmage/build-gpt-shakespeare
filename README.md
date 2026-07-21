# build-gpt-shakespeare

Personal replication of Andrej Karpathy's ["Let's build GPT: from scratch, in
code, spelled out"](https://www.youtube.com/watch?v=kCc8FmEb1nY) — a
character-level, decoder-only Transformer trained on tiny Shakespeare.

Reference (do not copy directly — rewrite it yourself, that's the point):
[github.com/karpathy/ng-video-lecture](https://github.com/karpathy/ng-video-lecture)

## Project layout

```
build-gpt-shakespeare/
├── src/
│   ├── bigram.py       # Sections 5-7: simplest baseline model
│   ├── gpt.py          # Sections 8-21: full GPT, built up piece by piece
│   └── model_config.py # hyperparameters, kept separate so you can bump them
│                         # for the cluster (batch size, n_layer, n_head, etc.)
├── data/
│   └── input.txt       # tiny shakespeare — fetch with scripts/download_data.sh
├── scripts/
│   └── download_data.sh
├── checkpoints/         # trained model weights — gitignored, lives on cluster
├── .vscode/
│   └── sftp.json        # cluster connection config (fill in your details)
├── requirements.txt
└── .gitignore
```

## Progress checklist (mirrors the video's own sections)

- [ ] Data loading + character-level tokenizer
- [ ] Train/val split, batch loader (`get_batch`)
- [ ] Bigram baseline (`bigram.py`), trained end to end
- [ ] Self-attention math trick (weighted aggregation via matmul)
- [ ] Single self-attention head
- [ ] Multi-head attention
- [ ] Feedforward layer
- [ ] Residual connections
- [ ] LayerNorm
- [ ] Full Block (attention + FFN + residuals + LayerNorm), stacked
- [ ] Dropout, scaled up to ~10M params
- [ ] Training run on the cluster, sample generation from the trained model
- [ ] **Extension**: pick the next task to fine-tune/adapt this model for

## Two independent syncs — don't confuse them

**1. Local VS Code → GitHub** (source of truth for code)
```bash
git init
git add .
git commit -m "Initial project scaffold"
git branch -M main
git remote add origin git@github.com:<your-username>/build-gpt-shakespeare.git
git push -u origin main
```
After that, just use VS Code's Source Control panel (sidebar, or Ctrl+Shift+G):
stage → commit → sync. To make "commit" also auto-push, add to your VS Code
`settings.json`:
```json
"git.postCommitCommand": "push"
```
Commit at meaningful checkpoints (matching the list above), not on every
keystroke — a history like "bigram baseline working" / "added single
attention head" is one you'll actually want to read again later.

**2. Local VS Code → eehpc cluster via SFTP** (for running training — heavy
files never touch GitHub)
- Install the **SFTP** extension (Natizyskunk fork) in VS Code.
- Fill in `.vscode/sftp.json` with your cluster host/username/remote path.
- `Cmd/Ctrl+Shift+P` → "SFTP: Upload Project" to push code to the cluster,
  or enable `"uploadOnSave": true` to push on every save automatically.
- SSH into the cluster (integrated terminal or a separate one) to actually
  launch training jobs, then "SFTP: Download" to pull back checkpoints/logs.
- `data/`, `checkpoints/`, and any `.pt`/`.bin` files stay out of git
  (see `.gitignore`) — they move between local ↔ cluster over SFTP only.
