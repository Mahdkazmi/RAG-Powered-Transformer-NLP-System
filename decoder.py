# ============================================================
# Part C: Decoder-Only Transformer + RAG Pipeline
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from encoder import PositionalEncoding, MultiHeadAttention, FeedForward

# ════════════════════════════════════════════════════════════
# C.1  Reference Explanation Templates
# ════════════════════════════════════════════════════════════
SENTIMENT_TEMPLATES = {
    "positive": [
        "This review is positive because the customer expresses clear satisfaction with the product.",
        "The review carries a positive sentiment as the user highlights strong performance and value.",
        "A positive tone is evident as the reviewer recommends the product and praises its quality.",
        "This is a positive review reflecting the customer's overall satisfaction and approval.",
        "The review is positive with the customer expressing happiness about their purchase.",
    ],
    "neutral": [
        "This review is neutral as the customer notes both strengths and weaknesses of the product.",
        "The review carries a neutral sentiment reflecting mixed feelings about the product.",
        "A neutral tone is present as the reviewer acknowledges some positives but also raises concerns.",
        "This review is neutral since the customer neither strongly endorses nor rejects the product.",
        "The sentiment is neutral as the reviewer presents a balanced assessment of the product.",
    ],
    "negative": [
        "This review is negative because the customer expresses dissatisfaction with the product.",
        "The review carries a negative sentiment as the user encountered problems or unmet expectations.",
        "A negative tone is evident as the reviewer warns others and expresses disappointment.",
        "This is a negative review reflecting the customer's frustration with the product quality.",
        "The review is negative with the customer expressing regret about their purchase.",
    ],
}

LENGTH_CONTEXT = {
    0: "The review is brief and concise.",
    1: "The review is moderately detailed.",
    2: "The review is lengthy and provides extensive detail.",
}

# ════════════════════════════════════════════════════════════
# C.6  Decoder Block (GPT-style, causal)
# ════════════════════════════════════════════════════════════
class DecoderBlock(nn.Module):
    """Decoder-only Transformer block with causal masking."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, causal_mask=None):
        x = x + self.dropout(self.attn(self.norm1(x), causal_mask))
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x

# ════════════════════════════════════════════════════════════
# C.7  Full Decoder Model
# ════════════════════════════════════════════════════════════
class ReviewDecoder(nn.Module):
    """Decoder-only Transformer for explanation generation."""
    
    def __init__(self, vocab_size: int, d_model: int, n_heads: int,
                 n_layers: int, d_ff: int, max_len: int,
                 dropout: float = 0.1, pad_idx: int = 0):
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx
        
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_enc = PositionalEncoding(d_model, max_len, dropout)
        
        self.layers = nn.ModuleList([
            DecoderBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    @staticmethod
    def make_causal_mask(seq_len: int, device) -> torch.Tensor:
        """Create causal mask to prevent attending to future tokens."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        return mask.unsqueeze(0).unsqueeze(0)
    
    def forward(self, input_ids):
        B, S = input_ids.size()
        c_mask = self.make_causal_mask(S, input_ids.device)
        
        x = self.embedding(input_ids) * math.sqrt(self.d_model)
        x = self.pos_enc(x)
        
        for layer in self.layers:
            x = layer(x, c_mask)
        
        x = self.norm(x)
        logits = self.lm_head(x)
        
        return logits

# ════════════════════════════════════════════════════════════
# C.10  RAG Generation Pipeline
# ════════════════════════════════════════════════════════════
class RAGGenerator:
    """RAG pipeline: encode → retrieve → generate."""
    
    def __init__(self, encoder, decoder, vocab, retriever, device):
        self.encoder = encoder
        self.decoder = decoder
        self.vocab = vocab
        self.retriever = retriever
        self.device = device
        
        self.idx2sentiment = {0: "negative", 1: "neutral", 2: "positive"}
    
    def make_explanation(self, sentiment: str, length_label: int, seed: int = 0) -> str:
        """Generate template explanation."""
        templates = SENTIMENT_TEMPLATES[sentiment]
        template = templates[seed % len(templates)]
        length_ctx = LENGTH_CONTEXT[length_label]
        return f"{template} {length_ctx}"
    
    def build_decoder_input(self, review_text: str, sentiment: str,
                           length_label: int, retrieved: list[dict],
                           max_review_chars: int = 200) -> str:
        """Build RAG prompt combining review + sentiment + length + context."""
        review_snippet = review_text[:max_review_chars].strip()
        
        retrieved_context = " | ".join([
            f"{r['sentiment']}({r['rating']}★): {r['text'][:80]}"
            for r in retrieved[:3]
        ])
        
        prompt = (
            f"[REVIEW] {review_snippet} "
            f"[SENTIMENT] {sentiment} "
            f"[LENGTH] {LENGTH_CONTEXT[length_label]} "
            f"[CONTEXT] {retrieved_context} "
            f"[EXPLAIN]"
        )
        return prompt
    
    def generate_explanation(self, review_text: str, k: int = 5,
                           max_new_tokens: int = 40,
                           temperature: float = 0.7) -> dict:
        """Full RAG pipeline: encode → retrieve → generate."""
        self.decoder.eval()
        self.encoder.eval()
        
        # Placeholder - would integrate full pipeline in practice
        return {
            "review": review_text[:150],
            "pred_sentiment": "positive",
            "pred_length": "The review is moderately detailed.",
            "retrieved_top1": "Sample retrieved review text",
            "explanation": "Generated explanation would go here.",
        }
