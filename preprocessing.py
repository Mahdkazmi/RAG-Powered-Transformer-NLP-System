# ============================================================
# Step 1-2: Dataset Loading & Preprocessing Pipeline
# ============================================================

import gzip
import json
import re
import random
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter

# ── Configuration ──────────────────────────────────────────
DATASET_DIR = Path(r"C:\Users\PC\Desktop\Semester 6\NLP\Assignment_3\Dataset")
SAMPLES_PER_CATEGORY = 13_000
RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
MAX_SEQ_LEN = 256
MIN_FREQ = 3

# Special tokens
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"
SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN]

# Sentiment label mapping
SENTIMENT_MAP = {"negative": 0, "neutral": 1, "positive": 2}

def map_sentiment(rating: int) -> str:
    """Map review rating to sentiment label."""
    if rating in (1, 2):
        return "negative"
    elif rating == 3:
        return "neutral"
    else:
        return "positive"

def load_category(filepath: Path, category: str, n_samples: int, seed: int) -> list[dict]:
    """Load and sample reviews from a .json.gz file."""
    records = []
    print(f"  Reading {filepath.name} ...", end=" ")
    
    with gzip.open(filepath, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            text = obj.get("reviewText", "").strip()
            rating = obj.get("overall", None)
            
            if not text or rating is None:
                continue
            
            rating = int(rating)
            if rating not in (1, 2, 3, 4, 5):
                continue
            
            records.append({
                "text": text,
                "rating": rating,
                "sentiment": map_sentiment(rating),
                "category": category,
            })
    
    print(f"found {len(records):,} valid reviews.", end=" ")
    
    if len(records) > n_samples:
        random.seed(seed)
        records = random.sample(records, n_samples)
        print(f"Sampled {n_samples:,}.")
    else:
        print(f"Using all {len(records):,} (fewer than requested).")
    
    return records

def load_datasets(dataset_dir: Path):
    """Load all category datasets and return train/val/test splits."""
    files = {
        "electronics": dataset_dir / "electronics.json.gz",
        "sports": dataset_dir / "sports.json.gz",
        "cellphones": dataset_dir / "cellphones.json.gz",
    }
    
    all_records = []
    for category, filepath in files.items():
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        records = load_category(filepath, category, SAMPLES_PER_CATEGORY, RANDOM_SEED)
        all_records.extend(records)
    
    df = pd.DataFrame(all_records)
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    
    total = len(df)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)
    
    train_df = df.iloc[:train_end].reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].reset_index(drop=True)
    test_df = df.iloc[val_end:].reset_index(drop=True)
    
    return train_df, val_df, test_df

# ── Text Cleaning ───────────────────────────────────────────
def clean_text(text: str) -> str:
    """Clean review text."""
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def tokenize(text: str) -> list[str]:
    """Simple whitespace tokenization."""
    return text.split()

def clean_and_tokenize(text: str) -> list[str]:
    """Clean and tokenize review text."""
    return tokenize(clean_text(text))

# ── Vocabulary ──────────────────────────────────────────────
class Vocabulary:
    """Build vocabulary from training data."""
    
    def __init__(self, min_freq: int = MIN_FREQ):
        self.min_freq = min_freq
        self.token2idx = {}
        self.idx2token = {}
        self.token_freq = Counter()
        self._built = False
    
    def build(self, tokenized_texts: list[list[str]]):
        """Build vocabulary from tokenized texts."""
        for tokens in tokenized_texts:
            self.token_freq.update(tokens)
        
        self.token2idx = {tok: idx for idx, tok in enumerate(SPECIAL_TOKENS)}
        
        idx = len(SPECIAL_TOKENS)
        for token, freq in self.token_freq.most_common():
            if freq >= self.min_freq:
                self.token2idx[token] = idx
                idx += 1
        
        self.idx2token = {idx: tok for tok, idx in self.token2idx.items()}
        self._built = True
        
        print(f"  Vocabulary built: {len(self.token2idx):,} tokens "
              f"(min_freq={self.min_freq}, "
              f"discarded {sum(1 for f in self.token_freq.values() if f < self.min_freq):,} rare tokens)")
    
    def encode(self, tokens: list[str]) -> list[int]:
        """Convert tokens to indices."""
        unk = self.token2idx[UNK_TOKEN]
        return [self.token2idx.get(tok, unk) for tok in tokens]
    
    def __len__(self):
        return len(self.token2idx)
    
    @property
    def pad_idx(self):
        return self.token2idx[PAD_TOKEN]
    
    @property
    def unk_idx(self):
        return self.token2idx[UNK_TOKEN]
    
    @property
    def sos_idx(self):
        return self.token2idx[SOS_TOKEN]
    
    @property
    def eos_idx(self):
        return self.token2idx[EOS_TOKEN]

def encode_and_pad(tokens: list[str], vocab: Vocabulary, max_len: int) -> np.ndarray:
    """Encode tokens and pad/truncate to max_len."""
    indices = vocab.encode(tokens)
    
    if len(indices) > max_len:
        indices = indices[:max_len]
    
    pad_length = max_len - len(indices)
    indices = indices + [vocab.pad_idx] * pad_length
    
    return np.array(indices, dtype=np.int32)

def preprocess_df(df: pd.DataFrame, vocab: Vocabulary = None,
                  is_train: bool = False, max_len: int = MAX_SEQ_LEN):
    """Preprocess DataFrame: clean, tokenize, encode, pad."""
    print(f"  Cleaning and tokenizing {len(df):,} reviews...", end=" ")
    df = df.copy()
    
    df["tokens"] = df["text"].apply(clean_and_tokenize)
    
    if is_train:
        vocab = Vocabulary(min_freq=MIN_FREQ)
        vocab.build(df["tokens"].tolist())
    
    df["input_ids"] = df["tokens"].apply(
        lambda toks: encode_and_pad(toks, vocab, max_len)
    )
    
    df["seq_len"] = df["tokens"].apply(len)
    df["sentiment_label"] = df["sentiment"].map(SENTIMENT_MAP)
    
    print("done.")
    return df, vocab
