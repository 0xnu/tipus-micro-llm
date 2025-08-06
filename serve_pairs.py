#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@name: serve_pairs.py
@author: Finbarrs Oketunji
@contact: f@finbarrs.eu
@time: Tuesday August 05 19:12:25 2025
@desc: Tipus Q&A API
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn as nn
from torch.nn import functional as F
import json
from pathlib import Path
import re
from typing import List, Dict

# -----------------------------
# Tokeniser (Word-level with special tokens)
# -----------------------------
class SimpleTokeniser:
    def __init__(self, word_to_id: Dict[str, int], id_to_word: Dict[int, str], special_tokens: List[str]):
        self.word_to_id = word_to_id
        self.id_to_word = id_to_word
        self.special_tokens = special_tokens
        self.vocab_size = len(word_to_id)
        self.unk_token = "<UNK>"
        self.sep_token = "<SEP>"
        self.eos_token = "<EOS>"
    
    def encode(self, text: str) -> List[int]:
        """Convert text to token IDs"""
        words = self._tokenise_text(text)
        return [self.word_to_id.get(word, self.word_to_id[self.unk_token]) for word in words]
    
    def decode(self, token_ids: List[int]) -> str:
        """Convert token IDs back to text"""
        words = []
        for id in token_ids:
            if id in self.id_to_word:
                word = self.id_to_word[id]
                if word not in self.special_tokens:
                    words.append(word)
        return " ".join(words)
    
    def _tokenise_text(self, text: str) -> List[str]:
        """Simple word tokenisation"""
        text = text.lower()
        words = re.findall(r'\b\w+\b|[.!?]', text)
        return words

# -----------------------------
# Positional Encoding
# -----------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                             -(torch.log(torch.tensor(10000.0)) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# -----------------------------
# Multi-Head Attention
# -----------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()
        
        # Linear transformations and split into heads
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        
        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)
            scores.masked_fill_(mask == 0, -1e9)
        
        # Causal mask for autoregressive generation
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).bool()
        scores.masked_fill_(~causal_mask, -1e9)
        
        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, V)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # Final linear transformation
        output = self.W_o(context)
        return output

# -----------------------------
# Transformer Block
# -----------------------------
class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(0.1)
        )
        
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_output = self.attention(self.norm1(x), mask)
        x = x + attn_output
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(self.norm2(x))
        x = x + ff_output
        
        return x

# -----------------------------
# Q&A Transformer Model
# -----------------------------
class QATransformer(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.d_model = 768
        self.embedding = nn.Embedding(vocab_size, self.d_model)
        self.pos_encoding = PositionalEncoding(self.d_model, max_len=256)  # Ensure max_len is defined
        self.dropout = nn.Dropout(0.1)
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(self.d_model, n_heads=8) for _ in range(8)
        ])
        
        self.ln_f = nn.LayerNorm(self.d_model)
        self.fc_out = nn.Linear(self.d_model, vocab_size)
        
    def forward(self, input_ids, attention_mask=None, targets=None):
        batch_size, seq_len = input_ids.shape
        
        # Embedding and positional encoding
        x = self.embedding(input_ids)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x, attention_mask)
        
        # Final layer norm and output projection
        x = self.ln_f(x)
        logits = self.fc_out(x)
        
        loss = None
        if targets is not None:
            # Ensure logits and targets have compatible shapes
            assert logits.shape[:2] == targets.shape, f"Shape mismatch: logits {logits.shape[:2]} vs targets {targets.shape}"
            
            # Calculate loss only on non-padded tokens
            pad_id = 0  # Padding token ID
            loss_fct = nn.CrossEntropyLoss(ignore_index=pad_id)
            loss = loss_fct(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        
        return logits, loss
    
    @torch.no_grad()
    def generate(self, input_ids, tokeniser, max_new_tokens=50, temperature=1.0):
        """Generate answer given a question"""
        self.eval()
        
        for _ in range(max_new_tokens):
            # Get predictions for last token
            logits, _ = self(input_ids)
            logits = logits[:, -1, :] / temperature
            
            # Sample from distribution
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # Stop if EOS token is generated
            if next_token.item() == tokeniser.word_to_id[tokeniser.eos_token]:
                break
        
        return input_ids

# -----------------------------
# Inference Class
# -----------------------------
class QAInference:
    def __init__(self, model_dir: str = "./model/", device: str = "cpu"):
        self.device = device
        model_path = Path(model_dir)
        
        # Find latest files
        model_files = sorted(model_path.glob("model_20250805_194450.pt"))
        if not model_files:
            raise ValueError(f"No model files found in {model_dir}")
        
        latest_model = model_files[-1]
        latest_model.stem.split('_')[1]
        
        # Load config
        with open(model_path / "config_20250805_194450.json", 'r') as f:
            config = json.load(f)
        
        # Load tokeniser
        with open(model_path / "tokeniser_20250805_194450.json", 'r') as f:
            tokeniser_data = json.load(f)
        
        self.tokeniser = SimpleTokeniser(
            word_to_id=tokeniser_data["word_to_id"],
            id_to_word={int(k): v for k, v in tokeniser_data["id_to_word"].items()},
            special_tokens=tokeniser_data["special_tokens"]
        )
        
        # Load model
        self.model = QATransformer(config["vocab_size"]).to(device)
        self.model.load_state_dict(torch.load(latest_model, map_location=device))
        self.model.eval()
    
    def answer_question(self, question: str, max_length: int = 50, temperature: float = 0.7) -> str:
        """Generate answer for a given question"""
        # Prepare input
        input_text = question + " " + self.tokeniser.sep_token
        input_ids = self.tokeniser.encode(input_text)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        
        # Generate answer
        with torch.no_grad():
            output_ids = self.model.generate(input_tensor, self.tokeniser, max_length, temperature)
        
        # Decode output
        output_tokens = output_ids[0].tolist()
        
        # Find separator token position
        sep_id = self.tokeniser.word_to_id[self.tokeniser.sep_token]
        if sep_id in output_tokens:
            sep_idx = output_tokens.index(sep_id)
            answer_tokens = output_tokens[sep_idx + 1:]
        else:
            answer_tokens = output_tokens[len(input_ids):]
        
        # Remove special tokens and decode
        answer = self.tokeniser.decode(answer_tokens)
        return answer.strip()

# -----------------------------
# FastAPI Application
# -----------------------------
app = FastAPI()

class QuestionRequest(BaseModel):
    question: str
    max_length: int = 120
    temperature: float = 0.7

@app.get("/")
async def main_entry():
    return {"message": "Welcome to the Q&A model API. Use the /generate endpoint to ask questions."}

@app.post("/generate")
async def generate_answer(request: QuestionRequest):
    try:
        inference = QAInference()
        answer = inference.answer_question(request.question, request.max_length, request.temperature)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=2025)