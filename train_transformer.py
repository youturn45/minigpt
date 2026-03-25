import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model'))

import torch
import sentencepiece as spm
from model import DecoderModel, get_batch, estimate_loss

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')
print(f'Using device: {DEVICE}')

# ── Hyperparameters ──────────────────────────────────────────
VOCAB_SIZE    = 20_000
BATCH_SIZE    = 64
BLOCK_SIZE    = 60
EMBED_DIM     = 128
N_HEAD        = 4
N_LAYER       = 4
DROPOUT       = 0.2
MAX_ITERS     = 50_000
EVAL_INTERVAL = 500
EVAL_ITERS    = 200
LR            = 3e-4

# ── Load tokenizer ───────────────────────────────────────────
sp = spm.SentencePieceProcessor()
sp.load('tokenizer/bpe.model')

# ── Load and tokenize data ───────────────────────────────────
with open('data/三国演义.txt', 'r') as f:
    text = f.read()

data = torch.tensor(sp.encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data   = data[n:]

print(f'Total tokens : {len(data):,}')
print(f'Train tokens : {len(train_data):,}')
print(f'Val tokens   : {len(val_data):,}')

# ── Build model ──────────────────────────────────────────────
model = DecoderModel(
    vocab_size=VOCAB_SIZE,
    embed_dim=EMBED_DIM,
    block_size=BLOCK_SIZE,
    n_head=N_HEAD,
    n_layer=N_LAYER,
    dropout=DROPOUT,
).to(DEVICE)

n_params = sum(p.numel() for p in model.parameters())
print(f'Model parameters: {n_params / 1e6:.2f}M')

# ── Train ────────────────────────────────────────────────────
os.makedirs('checkpoints', exist_ok=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
save_interval = MAX_ITERS // 10

def save_checkpoint(step):
    checkpoint = {
        'step': step,
        'state_dict': model.state_dict(),
        'hyperparams': {
            'vocab_size': VOCAB_SIZE,
            'embed_dim':  EMBED_DIM,
            'block_size': BLOCK_SIZE,
            'n_head':     N_HEAD,
            'n_layer':    N_LAYER,
            'dropout':    DROPOUT,
        },
    }
    path = f'checkpoints/transformer_step{step:05d}.pt'
    torch.save(checkpoint, path)
    print(f'Saved → {path}')

for step in range(MAX_ITERS):
    if step % EVAL_INTERVAL == 0 or step == MAX_ITERS - 1:
        losses = estimate_loss(
            model, train_data, val_data, BLOCK_SIZE, BATCH_SIZE, EVAL_ITERS, DEVICE
        )
        print(f'step {step:>5}: train loss {losses["train"]:.4f}  val loss {losses["val"]:.4f}')

    xb, yb = get_batch(train_data, val_data, 'train', BLOCK_SIZE, BATCH_SIZE, DEVICE)
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if (step + 1) % save_interval == 0:
        save_checkpoint(step + 1)
