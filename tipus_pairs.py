#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@name: tipus_pairs.py
@author: Finbarrs Oketunji
@contact: f@finbarrs.eu
@time: Tuesday August 05 19:12:25 2025
@desc: Token-based language model for Q&A pairs
"""

from __future__ import annotations
import json
import datetime as dt
from pathlib import Path
from typing import List, Tuple
import re

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

# -----------------------------
# Hyper-parameters
# -----------------------------
max_seq_length = 256  # Maximum sequence length for Q&A pairs
batch_size = 32
n_layer = 8  # Increased layers for better understanding
n_head = 8
n_embd = 768  # Larger embedding dimension
dropout = 0.1
max_iters = 10_000
eval_interval = 500
learning_rate = 1e-4
eval_iters = 50
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(1337)

# -------------------------------------------
# Tokeniser (Word-level with special tokens)
# -------------------------------------------
class SimpleTokeniser:
    def __init__(self, vocab_size_limit=10000):
        self.vocab_size_limit = vocab_size_limit
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.sep_token = "<SEP>"
        self.eos_token = "<EOS>"
        self.special_tokens = [self.pad_token, self.unk_token, self.sep_token, self.eos_token]
        self.word_to_id = {}
        self.id_to_word = {}
        self.vocab_size = 0
        
    def build_vocab(self, texts: List[str]):
        """Build vocabulary from list of texts"""
        word_freq = {}
        
        # Count word frequencies
        for text in texts:
            words = self._tokenise_text(text)
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sort by frequency and limit vocab size
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        vocab_words = [word for word, _ in sorted_words[:self.vocab_size_limit - len(self.special_tokens)]]
        
        # Build word-to-id mappings
        for i, token in enumerate(self.special_tokens):
            self.word_to_id[token] = i
            self.id_to_word[i] = token
        
        for i, word in enumerate(vocab_words, len(self.special_tokens)):
            self.word_to_id[word] = i
            self.id_to_word[i] = word
        
        self.vocab_size = len(self.word_to_id)
    
    def _tokenise_text(self, text: str) -> List[str]:
        """Simple word tokenisation"""
        text = text.lower()
        # Split on whitespace and punctuation
        words = re.findall(r'\b\w+\b|[.!?]', text)
        return words
    
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

# ----------------------
# Dataset for Q&A pairs
# ----------------------
class QADataset(Dataset):
    def __init__(self, qa_pairs: List[Tuple[str, str]], tokeniser: SimpleTokeniser, max_length: int):
        self.qa_pairs = qa_pairs
        self.tokeniser = tokeniser
        self.max_length = max_length
        
    def __len__(self):
        return len(self.qa_pairs)
    
    def __getitem__(self, idx):
        question, answer = self.qa_pairs[idx]
        
        # Encode question and answer with separator
        q_tokens = self.tokeniser.encode(question)
        a_tokens = self.tokeniser.encode(answer)
        
        # Create input: [Question] <SEP> [Answer] <EOS>
        sep_id = self.tokeniser.word_to_id[self.tokeniser.sep_token]
        eos_id = self.tokeniser.word_to_id[self.tokeniser.eos_token]
        pad_id = self.tokeniser.word_to_id[self.tokeniser.pad_token]
        
        input_ids = q_tokens + [sep_id] + a_tokens + [eos_id]
        
        # Truncate if too long (leave room for shifting)
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
        
        # Pad if too short
        padding_length = self.max_length - len(input_ids)
        if padding_length > 0:
            input_ids = input_ids + [pad_id] * padding_length
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = [1 if id != pad_id else 0 for id in input_ids]
        
        # For training: input is all tokens except last, targets are all tokens except first
        input_tensor = torch.tensor(input_ids[:-1], dtype=torch.long)
        target_tensor = torch.tensor(input_ids[1:], dtype=torch.long)
        attention_tensor = torch.tensor(attention_mask[:-1], dtype=torch.long)
        
        return {
            'input_ids': input_tensor,
            'targets': target_tensor,
            'attention_mask': attention_tensor
        }

# Load/Create Q&A data
def load_qa_data() -> List[Tuple[str, str]]:
    """Load or create Q&A pairs for training"""
    qa_file = Path("./data/qa_pairs.json")
    
    if qa_file.exists():
        # Load existing Q&A pairs
        with open(qa_file, 'r', encoding='utf-8') as f:
            qa_pairs = json.load(f)
    else:
        # Create sample Q&A pairs if file doesn't exist
        qa_pairs = [
            ("What is the capital of France?", "The capital of France is Paris."),
            ("Who wrote Romeo and Juliet?", "William Shakespeare wrote Romeo and Juliet."),
            ("What is the largest planet in our solar system?", "Jupiter is the largest planet in our solar system."),
            ("When was the Declaration of Independence signed?", "The Declaration of Independence was signed in 1776."),
            ("What is photosynthesis?", "Photosynthesis is the process by which plants convert light energy into chemical energy."),
            ("Who painted the Mona Lisa?", "Leonardo da Vinci painted the Mona Lisa."),
            ("What is the speed of light?", "The speed of light is approximately 299,792,458 metres per second."),
            ("What is machine learning?", "Machine learning is a type of artificial intelligence that enables computers to learn from data."),
            ("What is the smallest unit of matter?", "The atom is the smallest unit of matter that retains the properties of an element."),
            ("Who discovered gravity?", "Sir Isaac Newton discovered the law of universal gravitation.")
        ]
        
        # Save for future use
        qa_file.parent.mkdir(exist_ok=True)
        with open(qa_file, 'w', encoding='utf-8') as f:
            json.dump(qa_pairs, f, ensure_ascii=False, indent=2)
    
    return qa_pairs

# -------------------
# Model Architecture
# -------------------
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
        self.dropout = nn.Dropout(dropout)
        
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
            nn.Dropout(dropout)
        )
        
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_output = self.attention(self.norm1(x), mask)
        x = x + attn_output
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(self.norm2(x))
        x = x + ff_output
        
        return x

class QATransformer(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.d_model = n_embd
        self.embedding = nn.Embedding(vocab_size, self.d_model)
        self.pos_encoding = PositionalEncoding(self.d_model, max_seq_length)
        self.dropout = nn.Dropout(dropout)
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(self.d_model, n_head) for _ in range(n_layer)
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

# ---------
# Training
# ---------
def train_model():
    # Load data
    qa_pairs = load_qa_data()
    
    # Split into train and validation
    n_train = int(0.9 * len(qa_pairs))
    train_pairs = qa_pairs[:n_train]
    val_pairs = qa_pairs[n_train:]
    
    # Build tokeniser
    tokeniser = SimpleTokeniser()
    all_texts = [q + " " + a for q, a in qa_pairs]
    tokeniser.build_vocab(all_texts)
    
    # Create datasets and dataloaders
    train_dataset = QADataset(train_pairs, tokeniser, max_seq_length)
    val_dataset = QADataset(val_pairs, tokeniser, max_seq_length)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialise model
    model = QATransformer(tokeniser.vocab_size).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=max_iters)
    
    # Training loop
    global_step = 0
    model.train()
    
    for epoch in range(max_iters // len(train_loader) + 1):
        for batch in train_loader:
            if global_step >= max_iters:
                break
                
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            targets = batch['targets'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Forward pass
            logits, loss = model(input_ids, attention_mask, targets)
            
            # Backward pass
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            scheduler.step()
            
            # Evaluation
            if global_step % eval_interval == 0:
                model.eval()
                val_losses = []
                with torch.no_grad():
                    for val_batch in val_loader:
                        val_input = val_batch['input_ids'].to(device)
                        val_targets = val_batch['targets'].to(device)
                        val_mask = val_batch['attention_mask'].to(device)
                        _, val_loss = model(val_input, val_mask, val_targets)
                        val_losses.append(val_loss.item())
                
                avg_val_loss = sum(val_losses) / len(val_losses) if val_losses else float('inf')
                print(f"Step {global_step}: train loss = {loss.item():.4f}, val loss = {avg_val_loss:.4f}")
                model.train()
            
            global_step += 1
    
    # Save model and tokeniser
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path("models/")
    save_dir.mkdir(exist_ok=True)
    
    # Save model weights
    model_file = save_dir / f"model_{timestamp}.pt"
    torch.save(model.state_dict(), model_file)
    
    # Save tokeniser
    tokeniser_data = {
        "word_to_id": tokeniser.word_to_id,
        "id_to_word": {str(k): v for k, v in tokeniser.id_to_word.items()},
        "vocab_size": tokeniser.vocab_size,
        "special_tokens": tokeniser.special_tokens
    }
    with open(save_dir / f"tokeniser_{timestamp}.json", 'w') as f:
        json.dump(tokeniser_data, f, ensure_ascii=False, indent=2)
    
    # Save config
    config = {
        "max_seq_length": max_seq_length,
        "n_layer": n_layer,
        "n_head": n_head,
        "n_embd": n_embd,
        "dropout": dropout,
        "vocab_size": tokeniser.vocab_size
    }
    with open(save_dir / f"config_{timestamp}.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Model saved to {model_file}")
    return model, tokeniser

# ----------
# Inference
# ----------
class QAInference:
    def __init__(self, model_dir: str = "models/", device: str = "cpu"):
        self.device = device
        model_path = Path(model_dir)
        
        # Find latest files
        model_files = sorted(model_path.glob("model_*.pt"))
        if not model_files:
            raise ValueError(f"No model files found in {model_dir}")
        
        latest_model = model_files[-1]
        timestamp = latest_model.stem.split('_')[1]
        
        # Load config
        with open(model_path / f"config_{timestamp}.json", 'r') as f:
            config = json.load(f)
        
        # Load tokeniser
        with open(model_path / f"tokeniser_{timestamp}.json", 'r') as f:
            tokeniser_data = json.load(f)
        
        self.tokeniser = SimpleTokeniser()
        self.tokeniser.word_to_id = tokeniser_data["word_to_id"]
        self.tokeniser.id_to_word = {int(k): v for k, v in tokeniser_data["id_to_word"].items()}
        self.tokeniser.vocab_size = tokeniser_data["vocab_size"]
        self.tokeniser.special_tokens = tokeniser_data["special_tokens"]
        
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

# -------------
# Main Function
# -------------
if __name__ == "__main__":
    model, tokeniser = train_model()
    inference = QAInference()
    question = "What is the capital of France?"
    answer = inference.answer_question(question)
    print(f"Question: {question}\nAnswer: {answer}")
