import math
import torch
import torch.nn as nn
from torch.nn import functional as F


# =============================================================================
# PRE-TRANSFORMER: Input Preparation
# Token embeddings + positional encoding feed into the transformer stack.
# =============================================================================

class PositionalEncoding(nn.Module):
    """Injects position information into token embeddings using sine/cosine signals.

    Since transformers have no built-in sense of order, this adds a unique
    positional pattern to each token's embedding so the model knows where
    in the sequence each token sits.
    """

    def __init__(self, d_model=64, max_len=30):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """x: (batch_size, seq_len, d_model)"""
        return x + self.pe[:, : x.size(1), :]


# =============================================================================
# TRANSFORMER — Part 1: Attention Mechanism
# The core of the transformer. Tokens look at each other and decide
# what information to pass along. "Head" is one attention head;
# "MultiHeadAttention" runs several in parallel.
# =============================================================================

class Head(nn.Module):
    """Single causal self-attention head.

    Each token computes a Query (what am I looking for?), a Key (what do I
    offer?), and a Value (what do I send if selected?). The lower-triangular
    mask enforces causality — a token can only attend to itself and past tokens,
    never future ones.
    """

    def __init__(self, embed_dim, head_size, block_size, dropout):
        super().__init__()
        self.q = nn.Linear(embed_dim, head_size, bias=False)
        self.k = nn.Linear(embed_dim, head_size, bias=False)
        self.v = nn.Linear(embed_dim, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        wei = q @ k.transpose(-2, -1) / C**0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        return wei @ v


class MultiHeadAttention(nn.Module):
    """Runs multiple attention heads in parallel and combines their outputs.

    Different heads can learn to attend to different aspects of the context
    (e.g. syntax, semantics, co-reference). Their outputs are concatenated
    and projected back to the embedding dimension.
    """

    def __init__(self, n_head, head_size, embed_dim, block_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(embed_dim, head_size, block_size, dropout) for _ in range(n_head)]
        )
        self.proj = nn.Linear(n_head * head_size, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


# =============================================================================
# TRANSFORMER — Part 2: Feed-Forward Network
# After attention, each token independently processes what it gathered
# through a small two-layer MLP.
# =============================================================================

class FeedForward(nn.Module):
    """Position-wise feed-forward network applied after attention.

    Expands to 4x the embedding dimension, applies ReLU, then projects back.
    This gives each token a chance to process what it learned from attention.
    """

    def __init__(self, embed_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# =============================================================================
# TRANSFORMER — Part 3: One Full Transformer Block
# Attention + FeedForward wrapped with LayerNorm and residual connections.
# The model stacks N of these blocks on top of each other.
# =============================================================================

class Block(nn.Module):
    """One full transformer decoder layer: attention then feed-forward, both with residuals.

    LayerNorm is applied before each sub-layer (pre-norm style). The residual
    connections let gradients flow directly through the network during training.
    """

    def __init__(self, embed_dim, n_head, block_size, dropout):
        super().__init__()
        head_size = embed_dim // n_head
        self.sa = MultiHeadAttention(n_head, head_size, embed_dim, block_size, dropout)
        self.ffwd = FeedForward(embed_dim, dropout)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


# =============================================================================
# POST-TRANSFORMER: Output Head
# After the transformer stack, a final LayerNorm + linear layer maps
# each token's representation to a probability distribution over the vocabulary.
# =============================================================================

class DecoderModel(nn.Module):
    """Decoder-only transformer language model (GPT-style).

    Full pipeline: token embedding → positional encoding → N transformer
    blocks → layer norm → linear projection to vocabulary logits.

    During training, returns cross-entropy loss against the target tokens.
    During inference, uses `generate()` to autoregressively sample new tokens.
    """

    def __init__(self, vocab_size, embed_dim, block_size, n_head, n_layer, dropout):
        super().__init__()
        self.block_size = block_size
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_encoding = PositionalEncoding(d_model=embed_dim, max_len=block_size)
        self.blocks = nn.Sequential(
            *[Block(embed_dim, n_head, block_size, dropout) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, idx, targets=None):
        tok_emb = self.embedding(idx)           # (B, T, embed_dim)
        x = self.position_encoding(tok_emb)     # (B, T, embed_dim)
        x = self.blocks(x)                      # (B, T, embed_dim)
        x = self.ln_f(x)                        # (B, T, embed_dim)
        logits = self.lm_head(x)                # (B, T, vocab_size)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))

        return logits, loss

    def generate(self, idx, max_new_tokens):
        """Autoregressively sample new tokens given a starting context."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


def get_batch(train_data, val_data, split, block_size, batch_size, device):
    """Sample a random batch of (input, target) sequences from train or val data."""
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, block_size, batch_size, eval_iters, device):
    """Evaluate average loss over multiple batches on train and val splits."""
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(train_data, val_data, split, block_size, batch_size, device)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
