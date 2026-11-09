import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, head_dim):
        super().__init__()
        self.key = nn.Linear(head_dim, head_dim, bias=False)
        self.query = nn.Linear(head_dim, head_dim, bias=False)
        self.value = nn.Linear(head_dim, head_dim, bias=False)
    
    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        scores = q @ k.transpose(-2, -1) / (C ** 0.5)

        mask = torch.tril(torch.ones(T, T, device=x.device))
        scores = scores.masked_fill(mask == 0, float("-inf"))

        weights = F.softmax(scores, dim=-1)
        out = weights @ v

        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.heads = nn.ModuleList([
            SelfAttention(self.head_dim) for _ in range(num_heads)
        ])

        self.proj = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x):
        B, T, C = x.shape
        x = x.view(B, T, self.num_heads, self.head_dim)
        x = x.transpose(1, 2)
        out = torch.cat(
            [h(x[:, i]) for i, h in enumerate(self.heads)], 
            dim=-1
        )
        return self.proj(out)
