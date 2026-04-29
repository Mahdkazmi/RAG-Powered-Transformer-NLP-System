# ============================================================
# Part A: Encoder-Only Transformer (from scratch)
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ════════════════════════════════════════════════════════════
# A.1  Scaled Dot-Product Attention
# ════════════════════════════════════════════════════════════
class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, Q, K, V, mask=None):
        """Scaled dot-product attention."""
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        output = torch.matmul(attn_weights, V)
        return output, attn_weights

# ════════════════════════════════════════════════════════════
# A.2  Multi-Head Attention
# ════════════════════════════════════════════════════════════
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        
        self.attention = ScaledDotProductAttention(dropout)
        self.dropout = nn.Dropout(dropout)
    
    def split_heads(self, x):
        """(batch, seq, d_model) → (batch, heads, seq, d_k)"""
        B, S, _ = x.size()
        x = x.view(B, S, self.n_heads, self.d_k)
        return x.transpose(1, 2)
    
    def forward(self, x, mask=None):
        B = x.size(0)
        
        Q = self.split_heads(self.W_q(x))
        K = self.split_heads(self.W_k(x))
        V = self.split_heads(self.W_v(x))
        
        attn_out, _ = self.attention(Q, K, V, mask)
        
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, -1, self.d_model)
        
        return self.W_o(attn_out)

# ════════════════════════════════════════════════════════════
# A.3  Position-wise Feed-Forward Network
# ════════════════════════════════════════════════════════════
class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return self.net(x)

# ════════════════════════════════════════════════════════════
# A.4  Single Encoder Block
# ════════════════════════════════════════════════════════════
class EncoderBlock(nn.Module):
    """Transformer encoder layer with pre-norm."""
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        x = x + self.dropout(self.attn(self.norm1(x), mask))
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x

# ════════════════════════════════════════════════════════════
# A.5  Positional Encoding (sinusoidal, fixed)
# ════════════════════════════════════════════════════════════
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# ════════════════════════════════════════════════════════════
# A.6  Full Encoder Model
# ════════════════════════════════════════════════════════════
class ReviewEncoder(nn.Module):
    """Encoder-only Transformer with multi-task heads."""
    
    def __init__(self, vocab_size: int, d_model: int, n_heads: int,
                 n_layers: int, d_ff: int, max_len: int,
                 n_sentiment: int = 3, n_length: int = 3,
                 dropout: float = 0.1, pad_idx: int = 0):
        super().__init__()
        
        self.d_model = d_model
        self.pad_idx = pad_idx
        self.cls_token_id = vocab_size
        
        self.embedding = nn.Embedding(vocab_size + 1, d_model, padding_idx=pad_idx)
        self.pos_enc = PositionalEncoding(d_model, max_len + 1, dropout)
        
        self.layers = nn.ModuleList([
            EncoderBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        
        self.sentiment_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_sentiment)
        )
        
        self.length_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_length)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def make_pad_mask(self, input_ids):
        """Create padding mask."""
        B, S = input_ids.size()
        pad_mask = (input_ids == self.pad_idx)
        cls_col = torch.zeros(B, 1, dtype=torch.bool, device=input_ids.device)
        pad_mask = torch.cat([cls_col, pad_mask], dim=1)
        return pad_mask.unsqueeze(1).unsqueeze(2)
    
    def forward(self, input_ids):
        B, S = input_ids.size()
        
        cls_tokens = torch.full((B, 1), self.cls_token_id,
                                dtype=torch.long, device=input_ids.device)
        x_ids = torch.cat([cls_tokens, input_ids], dim=1)
        
        mask = self.make_pad_mask(input_ids)
        
        x = self.embedding(x_ids) * math.sqrt(self.d_model)
        x = self.pos_enc(x)
        
        for layer in self.layers:
            x = layer(x, mask)
        
        x = self.norm(x)
        
        cls_repr = x[:, 0, :]
        sentiment_logits = self.sentiment_head(cls_repr)
        length_logits = self.length_head(cls_repr)
        
        return sentiment_logits, length_logits, cls_repr
