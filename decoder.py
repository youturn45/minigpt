# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # MiniGPT — Decoder-Only Transformer Demo
#
# Loads the artefacts produced by `train.py` and walks through every stage of the
# transformer pipeline step by step, using **trained weights** throughout so that
# all similarity scores and attention patterns reflect real learned behaviour.
#
# **Run `train.py` first** to generate:
# - `tokenizer/bpe.model` — BPE tokenizer
# - `checkpoints/transformer_step*.pt` — trained model weights (latest checkpoint loaded automatically)
#
# **Pipeline overview:**
# ```
# raw text
#   │
#   ▼
# [Step 1]  Tokenisation      — BPE splits text into subword token IDs
#   │
#   ▼
# [Step 2]  Token Embedding   — each ID → dense vector (embed_dim)
#   │
#   ▼
# [Step 3]  Positional Enc.   — add position signal so the model knows token order
#   │
#   ▼
# [Step 4]  Attention Head    — one head: Q·K scores select which tokens to attend to
#   │
#   ▼
# [Step 5]  Multi-Head Attn   — N heads run in parallel, outputs concatenated
#   │
#   ▼
# [Step 6]  Feed-Forward      — per-token MLP processes what attention gathered
#   │
#   ▼
# [Step 7]  Full Block        — attention + FFN with LayerNorm and residuals
#   │
#   ▼
# [Step 8]  Stacked Blocks    — N_LAYER blocks refine representations layer by layer
#   │
#   ▼
# [Step 9]  Output Head       — LayerNorm → linear → softmax → P(next token)
#   │
#   ▼
# [Step 10] Inference         — autoregressively sample new tokens
# ```

# %%
import sys
import os
import glob
sys.path.append(os.path.join(os.path.dirname(os.path.abspath('__file__')), 'model'))

import torch
import torch.nn.functional as F
import sentencepiece as spm
from model import DecoderModel, get_batch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {DEVICE}')

# %% [markdown]
# ## Load Artefacts

# %%
# ── Tokenizer ────────────────────────────────────────────────
sp = spm.SentencePieceProcessor()
sp.load('tokenizer/bpe.model')
print(f'Tokenizer loaded  — vocab size: {sp.get_piece_size():,}')

# ── Model ─────────────────────────────────────────────────────
latest = sorted(glob.glob('checkpoints/transformer_step*.pt'))[-1]
print(f'Loading checkpoint: {latest}')
checkpoint = torch.load(latest, map_location=DEVICE)
hp = checkpoint['hyperparams']

model = DecoderModel(**hp).to(DEVICE)
model.load_state_dict(checkpoint['state_dict'])
model.eval()

print(f'Transformer loaded — {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters')
print(f'Hyperparameters    : {hp}')

# ── Unpack hyperparameters for use in the demos below ─────────
VOCAB_SIZE = hp['vocab_size']
EMBED_DIM  = hp['embed_dim']
BLOCK_SIZE = hp['block_size']
N_HEAD     = hp['n_head']
N_LAYER    = hp['n_layer']
DROPOUT    = hp['dropout']
HEAD_SIZE  = EMBED_DIM // N_HEAD

# %%
# Load corpus for train/val stats used later
with open('data/三国演义.txt', 'r') as f:
    text = f.read()

data = torch.tensor(sp.encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data   = data[n:]

BATCH_SIZE = 64

# Fixed canonical passage for the walkthrough: 温酒斩华雄
canonical_text = '''操曰：“将军出马，须要小心。”
云长曰：“如不胜，请斩某头。”
操教酾热酒一杯，与关公饮了上马。
关公曰：“酒且斟下，某去便来。”
出帐提刀，飞身上马。
众诸侯听得关外鼓声大震，喊声大举，如天摧地塌，岳撼山崩。
众皆失惊。
少顷，云长提华雄之头，掷于地上。
其酒尚温。'''

full_sample_ids = sp.encode(canonical_text)
if len(full_sample_ids) > BLOCK_SIZE:
    sample_ids = full_sample_ids[:BLOCK_SIZE]
    sample_text = sp.decode(sample_ids)
    truncated = True
else:
    sample_ids = full_sample_ids
    sample_text = canonical_text
    truncated = False
sample_pieces = [sp.id_to_piece(i) for i in sample_ids]
seq_len = len(sample_ids)
x_single = torch.tensor(sample_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
x_ids = x_single.repeat(BATCH_SIZE, 1)

print('=== Canonical walkthrough passage ===\n')
print(f'Full raw text : {canonical_text!r}\n')
if truncated:
    print(f'Using first {BLOCK_SIZE} tokens for model walkthrough because block_size={BLOCK_SIZE}.\n')
print(f'Walkthrough text: {sample_text!r}\n')
print(f'Subword pieces ({len(sample_pieces)}) : {sample_pieces}\n')
print(f'Token IDs    : {sample_ids}\n')
print(f'Batch shape  : {list(x_ids.shape)}  (batch_size={BATCH_SIZE}, seq_len={seq_len})')

# %% [markdown]
# ---
# ## Step 1 — Tokenisation
#
# Before the model sees any text it must convert raw characters into integers.
# BPE does this by learning common subword *pieces* from the training corpus:
# frequent sequences get their own token, rare ones are broken into smaller pieces.
#
# This lets the model handle any text with a fixed, compact vocabulary — common
# words like `刘备` may become a single token while rare names are split into
# smaller pieces the model has seen before.

# %%
examples = [
    '刘备仁义厚道',
    '曹操奸雄也',
    '诸葛亮足智多谋',
    '张飞勇猛善战',
]

print(f'{'Text':<16}  {'Pieces':<45}  IDs')
print('-' * 80)
for s in examples:
    pieces = sp.encode(s, out_type=str)
    ids    = sp.encode(s)
    print(f'{s:<16}  {str(pieces):<45}  {ids}')

# %% [markdown]
# ---
# ## Step 2 — Token Embedding
#
# Each integer token ID is looked up in a learned table (`vocab_size × embed_dim`)
# and replaced with a dense vector. The model adjusts these vectors during training
# so that tokens appearing in similar contexts end up close together in the
# `embed_dim`-dimensional space.
#
# We measure closeness with **cosine similarity**: 1.0 means identical direction,
# 0.0 means orthogonal, -1.0 means opposite.
#
# Because these are *trained* weights, tokens that appear in similar narrative
# contexts in 三国演义 should cluster together — without anyone labelling them.

# %%
with torch.no_grad():
    tok_emb = model.embedding(x_ids)   # (B, T, embed_dim)

print(f'Input  →  token IDs  : {list(x_ids.shape)}')
print(f'Output →  embeddings : {list(tok_emb.shape)}  — each ID is now a {EMBED_DIM}-dim vector')


# %%
def nearest_tokens_from_vector(vec: torch.Tensor, embedding, sp, top_n: int = 5, exclude_ids=None):
    """Return nearest tokens to an arbitrary vector in embedding space."""
    all_weights = embedding.weight.detach()
    sims = F.cosine_similarity(vec.unsqueeze(0), all_weights)
    exclude_ids = set(exclude_ids or [])
    for i in exclude_ids:
        if 0 <= i < len(sims):
            sims[i] = float('nan')
    valid = [(i, sims[i].item()) for i in range(len(sims)) if not sims[i].isnan()]
    valid.sort(key=lambda x: x[1], reverse=True)
    return [(sp.id_to_piece(i), f'{s:.2f}') for i, s in valid[:top_n]]


def show_similar_tokens(token: str, sp, embedding, top_n: int = 5):
    """Print the top_n most and least similar tokens to `token` by cosine similarity."""
    token_id = sp.piece_to_id(token)
    if token_id == 0:
        print(f"  '{token}' not found in vocabulary.\n")
        return

    all_weights = embedding.weight.detach()   # (vocab_size, embed_dim)
    token_vec   = all_weights[token_id].unsqueeze(0)

    sims = F.cosine_similarity(token_vec, all_weights)
    sims[token_id] = float('nan')
    valid = [(i, sims[i].item()) for i in range(len(sims)) if not sims[i].isnan()]
    valid.sort(key=lambda x: x[1], reverse=True)

    top = [(sp.id_to_piece(i), f'{s:.2f}') for i, s in valid[:top_n]]
    bot = [(sp.id_to_piece(i), f'{s:.2f}') for i, s in valid[-top_n:]]

    print(f"  '{token}'")
    print(f'    most similar    : {top}')
    print(f'    least similar   : {bot}\n')


probe_tokens = ['刘', '曹', '战', '兵', '城']
print('=== Trained embedding similarity ===\n')
print('Tokens that appear in similar contexts will cluster together.\n')
for t in probe_tokens:
    show_similar_tokens(t, sp, model.embedding, top_n=5)

# %% [markdown]
# ---
# ## Step 3 — Positional Encoding
#
# The transformer processes all tokens **simultaneously** — it has no built-in
# sense of order. Without position information these two sentences look identical:
#
# ```
# 刘备 打 曹操   →  bag of 3 tokens
# 曹操 打 刘备   →  same 3 tokens, completely different meaning
# ```
#
# The fix: **add a fixed sine/cosine pattern to each embedding based on its
# position**. Token at position 0 gets one pattern, position 1 a slightly
# different one, and so on. The embedding still carries *what* the token is;
# the pattern carries *where* it sits in the sequence.
#
# Positional encoding is **never updated by training** — it is computed once
# from a formula and stays fixed.

# %%
with torch.no_grad():
    x = model.position_encoding(tok_emb)   # adds position signal in-place

pe_matrix = model.position_encoding.pe.squeeze(0)   # (block_size, embed_dim)
print('Positional encoding vectors — first 3 positions, first 8 dimensions:\n')
for pos in range(3):
    vec = pe_matrix[pos, :8].tolist()
    print(f'  position {pos}: [{', '.join(f'{v:.3f}' for v in vec)}, ...]')

print(f'\nInput  →  embeddings           : {list(tok_emb.shape)}')
print(f'Output →  embeddings + position : {list(x.shape)}  (shape unchanged)')


# %%
def show_position_similarity(pe_module, anchor: int, positions: list):
    """Show cosine similarity between an anchor position and a list of others."""
    pe = pe_module.pe.squeeze(0)   # (block_size, embed_dim)
    anchor_vec = pe[anchor].unsqueeze(0)
    print(f'Cosine similarity to position {anchor}:\n')
    for p in positions:
        sim = F.cosine_similarity(anchor_vec, pe[p].unsqueeze(0)).item()
        bar = '█' * int(abs(sim) * 20)
        print(f'  vs position {p:>2}: {sim:+.3f}  {bar}')


print('Nearby positions share a similar pattern; distant positions differ.\n')
show_position_similarity(model.position_encoding, anchor=0,
                         positions=[1, 2, 5, 10, 20, 40, 59])
print()
show_position_similarity(model.position_encoding, anchor=30,
                         positions=[29, 28, 25, 20, 10, 1, 0])

# %% [markdown]
# ---
# ## Step 4 — Single Attention Head
#
# One attention head projects each token into three vectors:
# - **Query (Q)** — 'what information am I looking for?'
# - **Key (K)** — 'what information do I offer?'
# - **Value (V)** — 'what do I send if selected?'
#
# The dot product Q·Kᵀ scores how relevant each token is to every other token.
# A causal mask ensures position *t* can only attend to positions ≤ *t* — the
# model cannot cheat by looking at future tokens.
#
# After softmax the scores become weights that are used to take a weighted
# average of the Values, producing the head output.

# %%
# Access the first head of the first block
head = model.blocks[0].sa.heads[0]

with torch.no_grad():
    head_out = head(x)

print(f'Input  →  embeddings + position : {list(x.shape)}')
print(f'Output →  head output           : {list(head_out.shape)}  (batch, seq_len, head_size)')
print(f'\nhead_size = embed_dim / n_heads = {EMBED_DIM} / {N_HEAD} = {HEAD_SIZE}')
print(f'\nQ weight shape: {list(head.q.weight.shape)}  (head_size × embed_dim)')
print(f'K weight shape: {list(head.k.weight.shape)}')
print(f'V weight shape: {list(head.v.weight.shape)}')

# Recompute attention weights explicitly so we can inspect what one token attends to.
with torch.no_grad():
    B, T, C = x.shape
    q = head.q(x)
    k = head.k(x)
    wei = q @ k.transpose(-2, -1) / C**0.5
    wei = wei.masked_fill(head.tril[:T, :T] == 0, float('-inf'))
    wei = F.softmax(wei, dim=-1)

preferred_probe_tokens = ['温', '酒', '华雄', '云长', '关公']
probe_pos = None
for target in preferred_probe_tokens:
    if target in sample_pieces:
        probe_pos = sample_pieces.index(target)
        break
if probe_pos is None:
    probe_pos = min(12, T - 1)
probe_token = sample_pieces[probe_pos]
weights = wei[0, probe_pos].detach().cpu()
valid_positions = list(range(probe_pos + 1))
ranked = sorted(valid_positions, key=lambda i: weights[i].item(), reverse=True)[:5]

print('\n=== Single-head attention focus ===\n')
print(f"Probe token at position {probe_pos}: {probe_token!r}")
print('Top attended previous tokens:')
for i in ranked:
    tok = sample_pieces[i]
    print(f"  pos {i:>2}  token {tok!r:<12}  weight {weights[i].item():.3f}")

# %% [markdown]
# ---
# ## Step 5 — Multi-Head Attention
#
# Running a single head limits what the model can attend to at once. Running
# `N_HEAD` heads **in parallel** lets different heads specialise — one might
# focus on syntactic relationships, another on semantic ones.
#
# The outputs of all heads are concatenated (`N_HEAD × head_size = embed_dim`)
# and projected back through a linear layer so the shape is unchanged.

# %%
mha = model.blocks[0].sa

with torch.no_grad():
    mha_out = mha(x)

print(f'Input  →  embeddings + position : {list(x.shape)}')
print(f'Output →  multi-head attention  : {list(mha_out.shape)}  (shape preserved)')
print(f'\n{N_HEAD} heads × head_size {HEAD_SIZE} = {N_HEAD * HEAD_SIZE}, projected back to embed_dim {EMBED_DIM}')
print(f'\nProjection weight shape: {list(mha.proj.weight.shape)}')

print('\n=== How different heads focus on different context ===\n')
for h_idx, h in enumerate(mha.heads):
    with torch.no_grad():
        q = h.q(x)
        k = h.k(x)
        wei = q @ k.transpose(-2, -1) / x.shape[-1]**0.5
        wei = wei.masked_fill(h.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
    weights = wei[0, probe_pos].detach().cpu()
    ranked = sorted(valid_positions, key=lambda i: weights[i].item(), reverse=True)[:3]
    top_desc = [(sample_pieces[i], f'{weights[i].item():.3f}') for i in ranked]
    print(f'  head {h_idx}: {top_desc}')

# %% [markdown]
# ---
# ## Step 6 — Feed-Forward Network
#
# After attention each token has gathered information from its context.
# A small two-layer MLP then processes *each token independently*:
# expand to `4 × embed_dim`, apply ReLU, project back.
#
# The 4× expansion gives the model room to apply non-linear transformations
# — effectively letting it decide what to do with what it just read.

# %%
ffn = model.blocks[0].ffwd

with torch.no_grad():
    ffn_out = ffn(x)

print(f'Input  →  embeddings  : {list(x.shape)}')
print(f'Hidden →  expanded    : (batch, seq_len, {4 * EMBED_DIM})  [4× internal expansion]')
print(f'Output →  ffn output  : {list(ffn_out.shape)}  (projected back to embed_dim)')
print(f'\nLayer shapes:')
print(f'  Linear 1: {list(ffn.net[0].weight.shape)}  (4×embed_dim, embed_dim)')
print(f'  ReLU')
print(f'  Linear 2: {list(ffn.net[2].weight.shape)}  (embed_dim, 4×embed_dim)')

pre_ffn_vec = x[0, probe_pos].detach().cpu()
post_ffn_vec = ffn_out[0, probe_pos].detach().cpu()

print('\n=== Feed-forward semantic shift ===\n')
print(f"Probe token: {probe_token!r}")
print('Nearest tokens before FFN :', nearest_tokens_from_vector(pre_ffn_vec, model.embedding, sp, top_n=5, exclude_ids=[sample_ids[probe_pos]]))
print('Nearest tokens after FFN  :', nearest_tokens_from_vector(post_ffn_vec, model.embedding, sp, top_n=5, exclude_ids=[sample_ids[probe_pos]]))
print(f"Cosine(before, after)     : {F.cosine_similarity(pre_ffn_vec.unsqueeze(0), post_ffn_vec.unsqueeze(0)).item():.3f}")

# %% [markdown]
# ---
# ## Step 7 — One Full Transformer Block
#
# A `Block` combines multi-head attention and the feed-forward network, each
# wrapped with **LayerNorm** and a **residual connection**:
#
# ```
# x  →  LayerNorm  →  MultiHeadAttention  →  + x   (residual)
#    →  LayerNorm  →  FeedForward         →  + x   (residual)
# ```
#
# - **LayerNorm** (pre-norm style) stabilises the activations before each sub-layer.
# - **Residual connections** let gradients flow directly through the network,
#   making it possible to stack many layers without the signal vanishing.

# %%
block = model.blocks[0]

with torch.no_grad():
    block_out = block(x)

print(f'Input  →  embeddings   : {list(x.shape)}')
print(f'Output →  block output : {list(block_out.shape)}  (shape preserved through the entire block)')

# Show the residual connection in action: the output is close to the input
delta = (block_out - x).abs().mean().item()
print(f'\nMean absolute change from residual: {delta:.4f}')
print('(small = the block made targeted adjustments rather than replacing the representation)')

pre_block_vec = x[0, probe_pos].detach().cpu()
post_block_vec = block_out[0, probe_pos].detach().cpu()

print('\n=== One full block: semantic refinement ===\n')
print(f"Probe token: {probe_token!r}")
print('Nearest tokens before block:', nearest_tokens_from_vector(pre_block_vec, model.embedding, sp, top_n=5, exclude_ids=[sample_ids[probe_pos]]))
print('Nearest tokens after block :', nearest_tokens_from_vector(post_block_vec, model.embedding, sp, top_n=5, exclude_ids=[sample_ids[probe_pos]]))
print(f"Cosine(before, after)      : {F.cosine_similarity(pre_block_vec.unsqueeze(0), post_block_vec.unsqueeze(0)).item():.3f}")

# %% [markdown]
# ---
# ## Step 8 — Stacked Transformer Blocks
#
# The real depth of a transformer comes from stacking `N_LAYER` blocks.
# Each block can build on the representations produced by the one below it:
# - Early blocks tend to capture low-level patterns (character n-grams, punctuation)
# - Later blocks capture higher-level structure (who is doing what to whom)
#
# Because residual connections run through every block, the signal and gradients
# can still travel the full depth without degrading.

# %%
# Show how the representation evolves layer by layer
print(f'Tracking representation change through {N_LAYER} blocks:\n')
current = x
for i, block in enumerate(model.blocks):
    with torch.no_grad():
        out = block(current)
    delta = (out - current).abs().mean().item()
    print(f'  Block {i}  input: {list(current.shape)}  →  output: {list(out.shape)}  '
          f'  mean change: {delta:.4f}')
    current = out

stacked_out = current
print(f'\nFinal output after all {N_LAYER} blocks: {list(stacked_out.shape)}')

print('\n=== Layer-by-layer semantic drift ===\n')
current = x
for i, block in enumerate(model.blocks):
    with torch.no_grad():
        current = block(current)
    vec = current[0, probe_pos].detach().cpu()
    neighbors = nearest_tokens_from_vector(vec, model.embedding, sp, top_n=5, exclude_ids=[sample_ids[probe_pos]])
    print(f'  after block {i}: {neighbors}')

# %% [markdown]
# ---
# ## Step 9 — Output Head: Logits → Probabilities
#
# After the transformer stack, the model needs to produce a prediction for the
# next token at each position.
#
# 1. **LayerNorm** — final stabilisation of the stacked-block output.
# 2. **Linear** (`embed_dim → vocab_size`) — project each position's vector to a
#    score (logit) for every token in the vocabulary.
# 3. **Softmax** — convert raw scores into a probability distribution that sums to 1.
#
# At inference, only the last position's distribution is used to sample the next token.

# %%
with torch.no_grad():
    normed = model.ln_f(stacked_out)
    logits = model.lm_head(normed)
    probs  = torch.softmax(logits, dim=-1)

print(f'After stacked blocks  : {list(stacked_out.shape)}')
print(f'After LayerNorm       : {list(normed.shape)}')
print(f'Logits (raw scores)   : {list(logits.shape)}  (batch, seq_len, vocab_size)')
print(f'Probabilities         : {list(probs.shape)}   (sum to 1.0 across vocab dim)')

# Show the top predicted next tokens for a meaningful prefix from the canonical scene
prefix = '其酒尚'
prefix_ids = torch.tensor(sp.encode(prefix), dtype=torch.long, device=DEVICE).unsqueeze(0)
with torch.no_grad():
    prefix_logits, _ = model(prefix_ids)
    last_probs = torch.softmax(prefix_logits[:, -1, :], dim=-1)[0]

top_k = torch.topk(last_probs, 10)
print(f'\nTop 10 predicted next tokens after {prefix!r}:\n')
for prob, idx in zip(top_k.values.tolist(), top_k.indices.tolist()):
    piece = sp.id_to_piece(idx)
    bar   = '█' * int(prob * 200)
    print(f'  {piece:>8}  {prob:.4f}  {bar}')

# %% [markdown]
# ---
# ## Step 10 — Inference
#
# Text generation is **autoregressive**: the model predicts one token at a time,
# appends it to the context, then predicts the next one.
#
# ```
# prompt tokens  →  model  →  P(next token)
#                               │
#                               ▼  sample
# prompt + new token  →  model  →  P(next token)  →  ...
# ```
#
# Because the model was trained on 三国演义, a Chinese-language seed will produce
# text that resembles the style and characters of the novel.

# %%
prompts = ['操曰：“将军出马，须要小心。”', '关公曰：“酒且斟下，某去便来。”', '其酒尚']

for prompt in prompts:
    context = torch.tensor(sp.encode(prompt), dtype=torch.long, device=DEVICE).unsqueeze(0)

    with torch.no_grad():
        generated = model.generate(context, max_new_tokens=120)

    output = sp.decode(generated[0].tolist())
    print(f'Prompt: {prompt!r}')
    print(f'Output: {output!r}')
    print()
