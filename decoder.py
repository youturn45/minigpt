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
# The algorithm used here is **Byte-Pair Encoding (BPE)**.
#
# ### How BPE builds a vocabulary from a corpus
#
# **`vocab_size` is the total number of tokens in the final dictionary** —
# every character the tokenizer knows about, plus every merged token, counted together.
# It is the size of the lookup table that maps token → integer ID.
#
# Example with the letters A B C D E F G H and `vocab_size = 20`:
# ```
# Start : vocab = {A, B, C, D, E, F, G, H}  → 8 tokens, IDs 0-7
#
# Merge 1: AB is the most frequent pair → add AB to vocab
#          vocab = {A, B, C, D, E, F, G, H, AB}  → 9 tokens
#
# Merge 2: CD is next → add CD
#          vocab = {A, B, C, D, E, F, G, H, AB, CD}  → 10 tokens
# ...
# Merge 12: vocab reaches 20 tokens → stop
# ```
# The original characters are **never removed** — they stay in the vocab so that
# any rare combination the model hasn't seen can still fall back to individual chars.
# `vocab_size - num_unique_chars` tells you exactly how many merges will happen.
#
# **The algorithm:**
# ```
# vocab     ← all unique characters in the corpus          (fills slots 0..N-1)
# sequences ← every line split into individual characters
#
# while len(vocab) < vocab_size:
#     pairs ← count every adjacent (a, b) across all sequences
#     best  ← the pair with the highest count across the whole corpus
#     merge best everywhere → ab replaces every (a, b) in every sequence
#     vocab ← vocab ∪ {ab}                                 (fills one more slot)
# ```
#
# Counting pairs across the **whole corpus** means a pair that appears in many
# different lines beats one that appears many times in a single line. That is why
# `刘备` and `曹操` merge early — they appear throughout the entire novel.
#
# After training only the merge table is saved. At inference the tokenizer just
# replays those merges on new text — no corpus needed.
#
# **Why this is better than fixed word splitting:**
# - Common words → single tokens (fewer steps for the model to process)
# - Rare or unseen words → split into known smaller pieces (no unknown tokens ever)
# - `vocab_size` is an explicit knob: larger = fewer splits per word, smaller = more
#
# The cell below runs a **toy BPE simulation** on a small corpus so you can
# watch the vocabulary grow and the sequences shrink merge by merge.

# %%
from collections import Counter

def run_toy_bpe(corpus: list, vocab_size: int):
    """Simulate BPE training on a corpus until vocab reaches vocab_size."""
    # Initial vocab = every unique character across all lines
    initial_chars = sorted({ch for line in corpus for ch in line})
    vocab = set(initial_chars)
    sequences = [tuple(line) for line in corpus]
    n_merges = vocab_size - len(vocab)

    print(f'Corpus ({len(corpus)} lines):')
    for line in corpus:
        print(f'  {line!r}')
    print(f'\nInitial vocab  : {len(vocab)} unique characters')
    print(f'Target vocab   : {vocab_size}')
    print(f'Merges needed  : {n_merges}')
    print(f'\n{"Merge":<5}  {"Pair":<10}  {"Freq":<6}  {"New token":<10}  Vocab size')
    print('-' * 55)

    for merge_n in range(1, n_merges + 1):
        pairs = Counter()
        for seq in sequences:
            for a, b in zip(seq, seq[1:]):
                pairs[(a, b)] += 1
        if not pairs:
            break

        best  = max(pairs, key=pairs.__getitem__)
        merged = best[0] + best[1]
        vocab.add(merged)

        new_sequences = []
        for seq in sequences:
            new_seq, i = [], 0
            while i < len(seq):
                if i < len(seq) - 1 and seq[i] == best[0] and seq[i+1] == best[1]:
                    new_seq.append(merged)
                    i += 2
                else:
                    new_seq.append(seq[i])
                    i += 1
            new_sequences.append(tuple(new_seq))
        sequences = new_sequences

        print(f'{merge_n:<5}  {best[0]}+{best[1]:<6}  {pairs[best]:<6}  {merged!r:<10}  {len(vocab)}')

    print(f'\nFinal tokenisation of each line:')
    for orig, seq in zip(corpus, sequences):
        print(f'  {orig!r}')
        print(f'    → {list(seq)}')


toy_corpus = [
    '操曰：“将军出马，须要小心。”',
    '云长曰：“如不胜，请斩某头。”',
    '操教酾热酒一杯，与关公饮了上马。',
    '关公曰：“酒且斟下，某去便来。”',
    '出帐提刀，飞身上马。',
    '众诸侯听得关外鼓声大震，喊声大举，如天摧地塌，岳撼山崩。',
    '众皆失惊。',
    '少顷，云长提华雄之头，掷于地上。',
    '其酒尚温。'
]

unique_chars = {ch for line in toy_corpus for ch in line}
# unique chars fill slots 0..N-1 permanently; +30 adds exactly 30 merges on top.
run_toy_bpe(toy_corpus, vocab_size=len(unique_chars) + 10)

# %% [markdown]
# ### Why do we tokenize at all?
#
# Neural networks are mathematical functions — they can only operate on numbers,
# not on text. Tokenization is the bridge that converts language into a form the
# model can compute with.
#
# **The chain from raw text to model input:**
# ```
# "刘备胜"  →  tokenize  →  [刘备, 胜]  →  look up IDs  →  [312, 89]  →  embedding  →  vectors
# ```
#
# Each integer ID is then used to look up a row in the **embedding table** — a
# learned matrix where each token has its own vector. That is what the model
# actually sees and processes (Step 2 covers this in detail).
#
# **Why not feed in raw characters?**
#
# You could tokenize at the character level — every Chinese character gets its own ID.
# The problem is that the sequence becomes very long, and the model has to learn
# relationships between characters from scratch. `刘` and `备` are individually
# meaningless; the model only learns that `刘备` refers to a person after seeing
# the pair thousands of times.
#
# BPE solves this by pre-computing that `刘备` is a meaningful unit before the model
# ever sees it. The model gets a single token with a single embedding to learn from,
# rather than two separate tokens it has to learn to associate.
#
# **Why not use whole words?**
#
# Classical Chinese has no spaces, so "words" are ambiguous. More importantly, a
# word-level vocabulary would have hundreds of thousands of entries — most seen
# only a handful of times, making their embeddings unreliable. BPE finds the
# sweet spot: common patterns get their own token, rare ones share character-level
# pieces they have in common with frequent words.
#
# **What vocab_size controls in practice:**
#
# | vocab_size | effect |
# |---|---|
# | small (1 000) | almost every token is a single character — long sequences, slow training |
# | medium (10 000) | common words merged, rare ones split into 2–3 pieces |
# | large (64 000) | most common words and phrases are single tokens — our model's setting |
# | very large (100 000+) | diminishing returns; rare tokens have too few training examples |
#
# ### What the trained tokenizer does with real text
#
# The model's tokenizer was trained on the full 三国演义 corpus with a vocabulary
# of 64,000 pieces, so its merge table is much richer. Common names and phrases
# from the novel collapse into single tokens; unusual combinations are split.

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
# sense of order. Without position information, swapping the words produces
# the exact same internal representation:
#
# ```
# 关羽斩华雄   →  [关羽, 斩, 华雄]  — three vectors, no order
# 华雄斩关羽   →  [华雄, 斩, 关羽]  — same three vectors, shuffled
# ```
#
# The fix: **add a unique position signal to each token's vector** before
# it enters the transformer. The token still carries *what* it means;
# the position signal carries *where* it sits in the sentence.

# %%
with torch.no_grad():
    x = model.position_encoding(tok_emb)   # adds position signal in-place

# %%
# Demonstrate: same token at different positions → different vector
sent_a = '关羽斩华雄'
sent_b = '华雄斩关羽'

ids_a = sp.encode(sent_a)
ids_b = sp.encode(sent_b)
pieces_a = [sp.id_to_piece(i) for i in ids_a]
pieces_b = [sp.id_to_piece(i) for i in ids_b]

print(f'A: "{sent_a}"  →  {pieces_a}')
print(f'B: "{sent_b}"  →  {pieces_b}')
print()

t_a = torch.tensor(ids_a, dtype=torch.long, device=DEVICE).unsqueeze(0)
t_b = torch.tensor(ids_b, dtype=torch.long, device=DEVICE).unsqueeze(0)

with torch.no_grad():
    emb_a = model.embedding(t_a)
    emb_b = model.embedding(t_b)
    x_a   = model.position_encoding(emb_a)
    x_b   = model.position_encoding(emb_b)

# Find a token that appears in both sentences at genuinely different positions
target_piece = pos_a = pos_b = None
for i, tid in enumerate(ids_a):
    if tid in ids_b:
        j = ids_b.index(tid)
        if i != j:          # must be at a different position to show PE effect
            target_piece = sp.id_to_piece(tid)
            pos_a, pos_b  = i, j
            break

if target_piece is None:
    print('Could not find a shared token at different positions — try different example sentences.')
else:
    dist_before = (emb_a[0, pos_a] - emb_b[0, pos_b]).norm().item()
    dist_after  = (x_a[0, pos_a]  - x_b[0, pos_b]).norm().item()

    print(f'Shared token "{target_piece}" appears at different positions in each sentence:\n')
    print(f'  In A "{sent_a}": position {pos_a}  ({"subject — doing the slaying" if pos_a == 0 else "object — being slain"})')
    print(f'  In B "{sent_b}": position {pos_b}  ({"subject — doing the slaying" if pos_b == 0 else "object — being slain"})')
    print()
    print(f'  Vector distance BEFORE positional encoding: {dist_before:.4f}  ← identical, the model cannot tell them apart')
    print(f'  Vector distance AFTER  positional encoding: {dist_after:.4f}  ← now distinct, position is encoded in the vector')

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

def is_word(piece: str) -> bool:
    """Return True if the piece contains at least one Chinese character or letter."""
    return any('\u4e00' <= ch <= '\u9fff' or ch.isalpha() for ch in piece)

def show_head_attention(block_idx, head_idx):
    h = model.blocks[block_idx].sa.heads[head_idx]
    with torch.no_grad():
        q = h.q(x)
        k = h.k(x)
        w = q @ k.transpose(-2, -1) / x.shape[-1]**0.5
        w = w.masked_fill(h.tril[:T, :T] == 0, float('-inf'))
        w = F.softmax(w, dim=-1)
    print(f'Block {block_idx}, Head {head_idx}')
    print(f'{"Token":<12}  attends to')
    print('-' * 55)
    for pos in range(T):
        token = sample_pieces[pos]
        if not is_word(token):
            continue
        weights = w[0, pos].detach().cpu()
        ranked = sorted(range(pos + 1), key=lambda i: weights[i].item(), reverse=True)
        top_words = [sample_pieces[i] for i in ranked if is_word(sample_pieces[i])][:5]
        print(f'{token:<12}  →  {",  ".join(top_words)}')
    print()

print(f'Passage: {sample_text}\n')
for block_idx, head_idx in [(0, 0), (0, 1), (1, 0), (1, 1), (N_LAYER - 1, 0), (N_LAYER - 1, N_HEAD - 1)]:
    show_head_attention(block_idx, head_idx)

# keep probe_pos pointing at a meaningful token for later steps that reference it
preferred_probe_tokens = ['温', '酒', '华雄', '云长', '关公']
probe_pos = next((sample_pieces.index(t) for t in preferred_probe_tokens if t in sample_pieces), min(12, T - 1))
probe_token = sample_pieces[probe_pos]

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
