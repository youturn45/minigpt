# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: .venv_2
#     language: python
#     name: python3
# ---

# %% [markdown]
# # MiniGPT — Decoder-Only Transformer from Scratch
#
# This notebook walks through training a small GPT-style language model on a
# plain text file. The model learns to predict the next token and, once trained,
# can generate new text in the style of the training data.
#
# All model classes live in `model.py`. This notebook focuses on:
# - The data pipeline
# - A step-by-step visual walkthrough of the transformer
# - The training loop
# - Inference / text generation

# %% [markdown]
# ## 1. Imports

# %%
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath("__file__")), "model"))

import torch
import torch.nn as nn
import sentencepiece as spm
from model import (
    PositionalEncoding,
    Head,
    MultiHeadAttention,
    FeedForward,
    Block,
    DecoderModel,
    get_batch,
    estimate_loss,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# %% [markdown]
# ## 2. Tokenizer
#
# We use SentencePiece BPE to convert raw text into integer token IDs.
# The tokenizer is trained once on `input.txt` and saved to `mymodel.model`.
# Re-run the training cell only if you change the input file or vocabulary size.
#
# **Hyperparameter:**
# | Name | Value | Description |
# |---|---|---|
# | `VOCAB_SIZE` | 10000 | number of unique BPE tokens the tokenizer can produce |

# %%
VOCAB_SIZE = 20000   # number of unique BPE tokens

# %%
# Train tokenizer (only needs to run once)
spm.SentencePieceTrainer.train(
    input="data/三国演义.txt",
    model_prefix="tokenizer/mymodel",
    vocab_size=VOCAB_SIZE,
    model_type="bpe",
)

# %%
# Load the trained tokenizer
sp = spm.SentencePieceProcessor()
sp.load("tokenizer/mymodel.model")

print(f"Vocabulary size : {sp.get_piece_size()}")
print(f"Example encoding: {sp.encode('张飞何许人也', out_type=str)}")

# %% [markdown]
# ## 3. Data Loading
#
# The full text is encoded into a flat tensor of token IDs, then split
# 90% for training and 10% for validation.
#
# **Hyperparameters:**
# | Name | Value | Description |
# |---|---|---|
# | `BATCH_SIZE` | 64 | number of sequences processed in parallel per training step |
# | `BLOCK_SIZE` | 60 | maximum context length in tokens (how far back the model can look) |

# %%
BATCH_SIZE = 64   # sequences per training step
BLOCK_SIZE = 60   # context window length (tokens)

# %%
with open("data/三国演义.txt", "r") as f:
    text = f.read()

data = torch.tensor(sp.encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data   = data[n:]

print(f"Total tokens : {len(data):,}")
print(f"Train tokens : {len(train_data):,}")
print(f"Val tokens   : {len(val_data):,}")

# %% [markdown]
# ## 4. Transformer Walkthrough — Step by Step
#
# Rather than jumping straight to the full model, this section manually passes
# a sample batch through each component so you can see exactly what shape the
# data is at every stage.
#
# **Notation:** `(B, T, C)` means *batch × sequence length × channels (embedding dim)*
#
# **Hyperparameters:**
# | Name | Value | Description |
# |---|---|---|
# | `EMBED_DIM` | 128 | size of each token's vector representation |
# | `N_HEAD` | 4 | number of parallel attention heads per transformer block |
# | `N_LAYER` | 4 | number of transformer blocks stacked on top of each other |
# | `DROPOUT` | 0.2 | fraction of activations randomly zeroed during training to prevent overfitting |

# %%
EMBED_DIM = 128   # token embedding / hidden size
N_HEAD    = 4     # attention heads per block
N_LAYER   = 4     # number of stacked transformer blocks
DROPOUT   = 0.2   # dropout rate

# %%
# Grab a small sample batch to use throughout the walkthrough
x_ids, _ = get_batch(train_data, val_data, "train", BLOCK_SIZE, BATCH_SIZE, DEVICE)

# Show the raw text behind the first sequence in the batch
sample_ids   = x_ids[0].tolist()
sample_text  = sp.decode(sample_ids)
sample_pieces = [sp.id_to_piece(id) for id in sample_ids]

print("=== Sample sequence (first item in the batch) ===\n")
print(f"Raw text    : {sample_text!r}\n")
print(f"Subword pieces ({len(sample_pieces)}) : {sample_pieces}\n")
print(f"Token IDs   : {sample_ids}\n")
print(f"Batch shape : {list(x_ids.shape)}  (batch_size={BATCH_SIZE}, seq_len={BLOCK_SIZE})")

# %% [markdown]
# ### Step 1 — Token Embedding
#
# Each integer token ID is looked up in a learned embedding table and replaced
# with a dense vector of size `EMBED_DIM`. The model will learn what these
# vectors mean during training.
#
# To make this concrete, we pick a few characters and show which tokens in the
# vocabulary are most and least similar to them using **cosine similarity**.
#
# > Note: the embedding is randomly initialised here — similarities will be
# > meaningful only after training. This just shows the mechanism.

# %%
embedding = nn.Embedding(VOCAB_SIZE, EMBED_DIM).to(DEVICE)

tok_emb = embedding(x_ids)

print(f"Input  →  token IDs  : {list(x_ids.shape)}")
print(f"Output →  embeddings : {list(tok_emb.shape)}  — each ID is now a {EMBED_DIM}-dimensional vector\n")

# %%
import torch.nn.functional as F

def show_similar_tokens(token: str, sp, embedding, top_n: int = 5):
    """Print the top_n most and least similar tokens to `token` by cosine similarity."""
    token_id = sp.piece_to_id(token)
    if token_id == 0:
        print(f"'{token}' not found in vocabulary.\n")
        return

    all_weights = embedding.weight  # (vocab_size, embed_dim)
    token_vec   = all_weights[token_id].unsqueeze(0)  # (1, embed_dim)

    sims = F.cosine_similarity(token_vec, all_weights)  # (vocab_size,)

    # exclude the token itself
    sims[token_id] = float("nan")
    valid = [(i, sims[i].item()) for i in range(len(sims)) if not sims[i].isnan()]
    valid.sort(key=lambda x: x[1], reverse=True)

    top  = [(sp.id_to_piece(i), f"{s:.2f}") for i, s in valid[:top_n]]
    bot  = [(sp.id_to_piece(i), f"{s:.2f}") for i, s in valid[-top_n:]]

    print(f"  '{token}'")
    print(f"    top {top_n} similar  : {top}")
    print(f"    top {top_n} dissimilar: {bot}\n")

probe_tokens = ["刘", "曹", "战", "兵", "城"]
print("=== Embedding similarity (randomly initialised — run again after training) ===\n")
for t in probe_tokens:
    show_similar_tokens(t, sp, embedding, top_n=5)

# %%

# %% [markdown]
# ### Step 2 — Positional Encoding
#
# The transformer processes all tokens in a sequence **simultaneously** — unlike
# reading left to right, it sees everything at once. That means by default it
# has no idea what order the tokens are in.
#
# These two sentences would look completely identical to it:
#
# ```
# 刘备 打 曹操   →  just a bag of 3 tokens
# 曹操 打 刘备   →  same 3 tokens, totally different meaning
# ```
#
# Without order, the model can't tell who hit who.
#
# ---
#
# **The solution: add a position signal to every embedding**
#
# Before the tokens go into the transformer, we add a unique pattern to each one
# based on its position. Token at position 0 gets one pattern added, position 1
# gets a slightly different pattern, and so on.
#
# ```
# position 0:  embedding + [0.0,  1.0,  0.0,  1.0, ...]
# position 1:  embedding + [0.84, 0.54, 0.01, 0.99, ...]
# position 2:  embedding + [0.91, -0.41, 0.02, 0.99, ...]
# ```
#
# The embedding still carries **what** the token is.
# The positional pattern carries **where** it is.
# They are simply added together — shape goes in unchanged, position baked in.

# %%
pos_enc = PositionalEncoding(d_model=EMBED_DIM, max_len=BLOCK_SIZE).to(DEVICE)

x = pos_enc(tok_emb)

# Show the actual positional vectors for the first few positions
pe_matrix = pos_enc.pe.squeeze(0)  # (block_size, embed_dim)
print("Positional encoding vectors for first 3 positions (first 8 dimensions shown):\n")
for pos in range(3):
    vec = pe_matrix[pos, :8].tolist()
    print(f"  position {pos}: [{', '.join(f'{v:.2f}' for v in vec)}, ...]")

print(f"\nInput  →  embeddings           : {list(tok_emb.shape)}")
print(f"Output →  embeddings + position : {list(x.shape)}  (shape unchanged, position baked in)")

# %% [markdown]
# **How similar are nearby positions?**
#
# Because the patterns use sine/cosine waves that change gradually, nearby
# positions have similar patterns and distant positions have very different ones.
# We can verify this with cosine similarity between position vectors:

# %%
def show_position_similarity(pos_enc, anchor: int, compare_positions: list):
    """Show cosine similarity between an anchor position and a list of other positions."""
    pe = pos_enc.pe.squeeze(0)  # (block_size, embed_dim)
    anchor_vec = pe[anchor].unsqueeze(0)
    print(f"Cosine similarity to position {anchor}:\n")
    for p in compare_positions:
        sim = F.cosine_similarity(anchor_vec, pe[p].unsqueeze(0)).item()
        bar = "█" * int(abs(sim) * 20)
        print(f"  vs position {p:>2}: {sim:+.3f}  {bar}")

show_position_similarity(pos_enc, anchor=0, compare_positions=[1, 2, 5, 10, 20, 40, 59])

# %% [markdown]
# ### Step 3 — Single Attention Head
#
# One attention head projects the input into Queries, Keys, and Values, then
# computes how much each token should attend to every other (past) token.
# The output per head has size `EMBED_DIM // N_HEAD`.

# %%
HEAD_SIZE = EMBED_DIM // N_HEAD
head = Head(EMBED_DIM, HEAD_SIZE, BLOCK_SIZE, DROPOUT).to(DEVICE)

head_out = head(x)

print(f"Input  →  embeddings  : {list(x.shape)}")
print(f"Output →  head output : {list(head_out.shape)}  (batch, seq_len, head_size)")
print(f"\nNote: head_size = embed_dim / n_heads = {EMBED_DIM} / {N_HEAD} = {HEAD_SIZE}")

# %% [markdown]
# ### Step 4 — Multi-Head Attention
#
# We run `N_HEAD` attention heads in parallel. Each head can focus on different
# patterns (e.g. syntax, semantics). Their outputs are concatenated and projected
# back to `EMBED_DIM` — the same size as the input.

# %%
mha = MultiHeadAttention(N_HEAD, HEAD_SIZE, EMBED_DIM, BLOCK_SIZE, DROPOUT).to(DEVICE)

mha_out = mha(x)

print(f"Input  →  embeddings          : {list(x.shape)}")
print(f"Output →  multi-head attention: {list(mha_out.shape)}  (batch, seq_len, embed_dim)")
print(f"\nNote: {N_HEAD} heads × head_size {HEAD_SIZE} = {N_HEAD * HEAD_SIZE}, projected back to {EMBED_DIM}")

# %% [markdown]
# ### Step 5 — Feed-Forward Network
#
# After attention, each token independently passes through a small two-layer MLP.
# It expands to `4 × EMBED_DIM` internally, then projects back. This gives the
# model capacity to process what it gathered from attention.

# %%
ffn = FeedForward(EMBED_DIM, DROPOUT).to(DEVICE)

ffn_out = ffn(x)

print(f"Input  →  embeddings  : {list(x.shape)}")
print(f"Hidden →  expanded    : (batch, seq_len, {4 * EMBED_DIM})  [4× internal expansion]")
print(f"Output →  ffn output  : {list(ffn_out.shape)}  (back to embed_dim)")

# %% [markdown]
# ### Step 6 — One Full Transformer Block
#
# A `Block` combines steps 4 and 5 with **residual connections** and **LayerNorm**.
# Residuals let gradients flow during training; LayerNorm stabilises activations.
#
# ```
# x  →  LayerNorm  →  MultiHeadAttention  →  + x  →  LayerNorm  →  FeedForward  →  + x
# ```

# %%
block = Block(EMBED_DIM, N_HEAD, BLOCK_SIZE, DROPOUT).to(DEVICE)

block_out = block(x)

print(f"Input  →  embeddings   : {list(x.shape)}")
print(f"Output →  block output : {list(block_out.shape)}  (shape preserved through the block)")

# %% [markdown]
# ### Step 7 — Stacked Transformer Blocks (the full transformer)
#
# The real power comes from stacking `N_LAYER` blocks. Each block refines the
# representation. Information can flow and be reinterpreted layer by layer.

# %%
blocks = nn.Sequential(
    *[Block(EMBED_DIM, N_HEAD, BLOCK_SIZE, DROPOUT) for _ in range(N_LAYER)]
).to(DEVICE)

stacked_out = blocks(x)

print(f"Input  →  embeddings      : {list(x.shape)}")
print(f"Output →  after {N_LAYER} blocks  : {list(stacked_out.shape)}  (shape still preserved)")
print(f"\nEach of the {N_LAYER} blocks independently refines every token's representation.")

# %% [markdown]
# ### Step 8 — Output Head: Logits → Probabilities
#
# Finally, a `LayerNorm` stabilises the output, then a linear layer projects
# each token's `EMBED_DIM` vector to `VOCAB_SIZE` scores (logits). Softmax
# turns these into a probability distribution — one probability per token in
# the vocabulary.

# %%
ln_f    = nn.LayerNorm(EMBED_DIM).to(DEVICE)
lm_head = nn.Linear(EMBED_DIM, VOCAB_SIZE).to(DEVICE)

normed  = ln_f(stacked_out)
logits  = lm_head(normed)
probs   = torch.softmax(logits, dim=-1)

print(f"After stacked blocks  : {list(stacked_out.shape)}")
print(f"After LayerNorm       : {list(normed.shape)}")
print(f"Logits (raw scores)   : {list(logits.shape)}  (batch, seq_len, vocab_size)")
print(f"Probabilities         : {list(probs.shape)}   (sums to 1.0 across vocab)")
print(f"\nFor each of the {BLOCK_SIZE} positions, the model outputs a score for all {VOCAB_SIZE:,} possible next tokens.")

# %% [markdown]
# ## 5. Full Model
#
# All the steps above are packaged inside `DecoderModel`. During training it
# also computes cross-entropy loss against the target tokens.

# %%
model = DecoderModel(
    vocab_size=VOCAB_SIZE,
    embed_dim=EMBED_DIM,
    block_size=BLOCK_SIZE,
    n_head=N_HEAD,
    n_layer=N_LAYER,
    dropout=DROPOUT,
).to(DEVICE)

n_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {n_params / 1e6:.2f}M")

# %% [markdown]
# ## 6. Training
#
# We use the AdamW optimizer and minimise cross-entropy loss between the model's
# predicted next token and the actual next token in the training data.
# Loss is printed every `EVAL_INTERVAL` steps on both train and validation sets.
#
# **Hyperparameters:**
# | Name | Value | Description |
# |---|---|---|
# | `MAX_ITERS` | 5000 | total number of gradient update steps |
# | `EVAL_INTERVAL` | 500 | print train/val loss every N steps |
# | `EVAL_ITERS` | 200 | number of batches averaged to estimate loss |
# | `LR` | 3e-4 | learning rate for AdamW optimizer |

# %%
MAX_ITERS     = 5000   # total training steps
EVAL_INTERVAL = 500    # how often to print loss
EVAL_ITERS    = 200    # batches averaged for each loss estimate
LR            = 3e-4   # learning rate

# %%
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

for step in range(MAX_ITERS):

    if step % EVAL_INTERVAL == 0 or step == MAX_ITERS - 1:
        losses = estimate_loss(
            model, train_data, val_data, BLOCK_SIZE, BATCH_SIZE, EVAL_ITERS, DEVICE
        )
        print(f"step {step:>5}: train loss {losses['train']:.4f}  val loss {losses['val']:.4f}")

    xb, yb = get_batch(train_data, val_data, "train", BLOCK_SIZE, BATCH_SIZE, DEVICE)
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# %% [markdown]
# ## 7. Learned Embeddings — What the Model Now Knows
#
# Earlier in Step 1, the cosine similarities were random noise because the
# embedding was uninitialised. Now that training is complete, the embedding
# table has been updated thousands of times by backpropagation.
#
# Tokens that appear in similar contexts in 三国演义 should now sit close
# together in the 128-dimensional space — without anyone telling the model
# what any character or word "means". It learned purely from patterns in text.
#
# We reuse the same `show_similar_tokens` function from Step 1 but now pass
# in `model.embedding` — the trained weights — instead of the random one.

# %%
print("=== Embedding similarity (TRAINED) ===\n")
for t in probe_tokens:
    show_similar_tokens(t, sp, model.embedding, top_n=5)

# %% [markdown]
# Compare these results to the random ones from Step 1. Tokens like `刘` should
# now cluster with characters and words that appear alongside it in the text
# (e.g. `备`, `玄德`), while its enemies or unrelated concepts should score low
# or negative.
#
# This is the core intuition behind embeddings: **meaning emerges from context**.

# %% [markdown]
# ## 7b. Positional Encoding — Position Similarity (after training)
#
# Unlike the embedding table, positional encoding is **fixed** — it is never
# updated during training. So the similarities between positions are the same
# before and after training.
#
# What this demo shows is the *structure* of the encoding: nearby positions
# are more alike than distant ones, which is what we want the model to feel.
# Position 1 should "feel close" to position 0, and position 59 should "feel
# far away".

# %%
print("=== Positional similarity from position 0 ===\n")
show_position_similarity(pos_enc, anchor=0, compare_positions=[1, 2, 5, 10, 20, 40, 59])

print("\n=== Positional similarity from position 30 (middle of sequence) ===\n")
show_position_similarity(pos_enc, anchor=30, compare_positions=[29, 28, 25, 20, 10, 1, 0])

# %% [markdown]
# ## 8. Inference
#
# Seed the model with a short prompt and let it generate new tokens one at a
# time. Each new token is sampled from the predicted probability distribution
# over the vocabulary, then appended to the context for the next step.

# %%
prompt = "SAM:"
context = torch.tensor(sp.encode(prompt), dtype=torch.long, device=DEVICE).unsqueeze(0)

print(f"Prompt  : {prompt!r}")
print(f"Context shape: {list(context.shape)}  →  generating 500 tokens...\n")

generated = model.generate(context, max_new_tokens=500)
print(sp.decode(generated[0].tolist()))
