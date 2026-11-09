import torch
import torch.nn as nn
from model.attention import MultiHeadAttention

class TinyPoT(nn.Module):
    def __init__(self, vocab_size, config):
        super().__init__()
        
        self.token_embed = nn.Embedding(vocab_size, config.embedding_dim)
        self.pos_embed = nn.Embedding(config.block_size, config.embedding_dim)

        self.blocks = nn.Sequential(*[
            Block(config.embedding_dim, config.num_heads)
            for _ in range(config.num_layers)
        ])

        self.ln_f = nn.LayerNorm(config.embedding_dim)
        self.lm_head = nn.Linear(config.embedding_dim, vocab_size)
    
    def forward(self, idx):
        B, T = idx.shape

        tok = self.token_embed(idx)
        pos = self.pos_embed(torch.arange(T))

        x = tok + pos
        x = self.blocks(x)
        x = self.ln_f(x)

        logits = self.lm_head(x)
        return logits

class Block(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()

        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads)

        self.ln2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim)
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x
