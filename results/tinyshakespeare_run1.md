# Training run — CLUSTER config, tinyshakespeare, n10, 5000 iters

**Model:** decoder-only GPT, char-level tokenization, tiny Shakespeare dataset
**Config:** CLUSTER (n_embd=384, n_head=6, n_layer=6, block_size=256, batch_size=64, dropout=0.2)
**Node:** n10 (eehpc cluster)
**Parameters:** 10,788,929 (~10.79M)

## Loss curve

| Step | Train loss | Val loss |
|------|-----------|----------|
| 0    | 4.2221    | 4.2306   |
| 500  | 1.7509    | 1.9086   |
| 1000 | 1.3909    | 1.5970   |
| 1500 | 1.2704    | 1.5250   |
| 2000 | 1.1856    | 1.5057   |
| 2500 | 1.1222    | 1.4999   |
| 3000 | 1.0684    | 1.4830   |
| 3500 | 1.0197    | 1.5090   |
| 4000 | 0.9570    | 1.5212   |
| 4500 | 0.9049    | 1.5446   |
| 4999 | 0.8540    | 1.5724   |

## Observations

- Val loss bottoms out at **step 3000** (1.4830), then rises while train loss keeps falling — overfitting past this point.
- Best-generalizing checkpoint is `ckpt_iter3000.pt`, not the final one.
- For future runs, consider reducing `max_iters` to ~3000 for this dataset/config, or adding stronger regularization if longer training is wanted.

## Sample output (step 4999, post-overfitting)

```
The woman diversue that he comes abroad,
When we in his voices, through permonlio invet.
Nay, thrice, gapes thus to revenge, give him out
That it our hately may reverence his finger
May that climate; and in haste, so I will ovethe
Twice I have frown on you.
COMINIUS:
You are leave this,
This honour should in most public
And Menenius beloved the tremble.
AUFIDIUS:
Why, that's a name and they: if your tent now,
Where he think the gods you in the climate;
Call your son's reasons from accustoming
```