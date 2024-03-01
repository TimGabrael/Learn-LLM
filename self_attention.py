import torch
import torch.nn as nn

d_in = 256
d_out = d_in
block_size = 4

class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, block_size, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(block_size, block_size), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        
        attn_scores = queries @ keys.transpose(1, 2)
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=1)

        context_vec = attn_weights @ values
        return context_vec

class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, block_size, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList(
                [CausalAttention(d_in, d_out, block_size, dropout, qkv_bias)
                 for _ in range(num_heads)]
        )
    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)


# More efficient rewrite of MultiHeadAttentionWrapper, that combines every head in each matrix and outputs all context vectors simultanious
# Just like the Wrapper
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, block_size, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by n_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads # Reduce the projection dim to match desired output dim

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(block_size, block_size), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim) 
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        mask_unsqueezed = mask_bool.unsqueeze(0)
        attn_scores.masked_fill_(mask_unsqueezed, -torch.inf)
        
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2) 
        
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)

        # optional projection, often used in MultiheadAttention Algorithms
        context_vec = self.out_proj(context_vec)
        return context_vec

# multi_head_attention = MultiHeadAttentionWrapper(d_in, d_out, block_size, 0.0, num_heads=2)
# multi_head_attention = MultiHeadAttention(d_in, d_out, block_size, 0.0, num_heads=2)
# context_vecs = multi_head_attention(batch)

# Note: There is also the torch.nn.MultiheadAttention class in PyTorch

