import os
import sentencepiece as spm

VOCAB_SIZE = 64_000

os.makedirs('tokenizer', exist_ok=True)

spm.SentencePieceTrainer.train(
    input='data/三国演义.txt',
    model_prefix='tokenizer/bpe',
    vocab_size=VOCAB_SIZE,
    model_type='bpe',
    character_coverage=1.0,
)
print('Saved → tokenizer/bpe.model')
print('Saved → tokenizer/bpe.vocab')
