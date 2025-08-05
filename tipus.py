#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@name: tipus.py
@author: Finbarrs Oketunji
@contact: f@finbarrs.eu
@time: Sunday August 03 14:04:25 2025
@desc: A minimal character-level language model
"""

from __future__ import annotations
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F

# -----------------------------
# Hyper-parameters
# -----------------------------
block_size   = 128
batch_size   = 64
n_layer      = 6
n_head       = 8
n_embd       = 512
dropout      = 0.1
max_iters    = 5_000
eval_interval= 250
learning_rate= 3e-4
eval_iters   = 100
device       = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(1337)

# -----------------------------
# Data / tokenisation
# -----------------------------
text = Path("./data/corpus.txt").read_text(encoding="utf-8")
chars = sorted(set(text))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s: str) -> list[int]:
    return [stoi[c] for c in s]

def decode(token_ids: list[int]) -> str:
    return "".join(itos[i] for i in token_ids)

# -----------------------------
# Train / val split
# -----------------------------
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]

def get_batch(split: str):
    data = train_data if split == "train" else val_data
    upper = len(data) - block_size - 1
    if upper <= 0:
        raise ValueError(f"Dataset too small ({len(data)} < {block_size + 1})")
    ix = torch.randint(0, upper, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)

# -----------------------------
# Model
# -----------------------------
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * (C ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class CharLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block() for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        if targets is None:
            return logits, None
        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

# -----------------------------
# Training loop
# -----------------------------
model = CharLM().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

for iter in range(max_iters):
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train {losses['train']:.4f}, val {losses['val']:.4f}")
    X, Y = get_batch("train")
    logits, loss = model(X, Y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# -----------------------------
# Save checkpoint
# -----------------------------
save_dir = Path("model")
save_dir.mkdir(exist_ok=True)
torch.save(model.state_dict(), save_dir / "model.pt")
(save_dir / "meta.json").write_text(json.dumps({"stoi": stoi, "itos": itos,
                                                "vocab_size": vocab_size,
                                                "block_size": block_size}))
(save_dir / "config.json").write_text(json.dumps({"n_layer": n_layer,
                                                  "n_head": n_head,
                                                  "n_embd": n_embd,
                                                  "dropout": dropout}))
print("Saved checkpoint to", save_dir.resolve())

# -----------------------------
# Quick inference sanity-check
# -----------------------------
class CharLMInference:
    def __init__(self, ckpt_dir: str = "model", device: str = "cpu"):
        ckpt = Path(ckpt_dir)
        meta = json.loads((ckpt / "meta.json").read_text())
        json.loads((ckpt / "config.json").read_text())
        self.stoi, self.itos = meta["stoi"], {int(k): v for k, v in meta["itos"].items()}
        self.block_size = meta["block_size"]
        self.device = device
        self.model = CharLM().to(device)
        self.model.load_state_dict(torch.load(ckpt / "model.pt", map_location=device))
        self.model.eval()

    def encode(self, s: str) -> list[int]:
        return [self.stoi.get(c, 0) for c in s]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(self.itos.get(i, "") for i in token_ids)

    @torch.no_grad()
    def generate(self, prompt: str = "", max_new_tokens: int = 120):
        idx = torch.tensor(self.encode(prompt), dtype=torch.long, device=self.device).unsqueeze(0)
        for _ in range(max_new_tokens):
            idx = idx[:, -self.block_size :]
            logits = self.model(idx)[0][:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return self.decode(idx[0].tolist())

# Quick Test
if __name__ == "__main__":
    generator = CharLMInference()
    print(generator.generate("Creativity is ", 60))