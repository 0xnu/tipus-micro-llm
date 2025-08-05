#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@name: tipus.py
@author: Finbarrs Oketunji
@contact: f@finbarrs.eu
@time: Sunday August 03 16:00:25 2025
@desc: Tipus Micro-LLM API
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ------------------------------------------------------------
# Hyper-params (must match training script)
# ------------------------------------------------------------
BLOCK_SIZE   = 128
N_EMBD       = 512
N_HEAD       = 8
N_LAYER      = 6
DROPOUT      = 0.1
# ------------------------------------------------------------

# ---------- Transformer components ----------
class Head(nn.Module):
    def __init__(self, head_size: int) -> None:
        super().__init__()
        self.key   = nn.Linear(N_EMBD, head_size, bias=False)
        self.query = nn.Linear(N_EMBD, head_size, bias=False)
        self.value = nn.Linear(N_EMBD, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * (C**-0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        v = self.value(x)
        return wei @ v

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head: int) -> None:
        super().__init__()
        head_size = N_EMBD // n_head
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_head)])
        self.proj = nn.Linear(N_EMBD, N_EMBD)
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.proj(out)

class FeedForward(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_EMBD, 4 * N_EMBD),
            nn.ReLU(),
            nn.Linear(4 * N_EMBD, N_EMBD),
        )
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.sa   = MultiHeadAttention(N_HEAD)
        self.ffwd = FeedForward()
        self.ln1  = nn.LayerNorm(N_EMBD)
        self.ln2  = nn.LayerNorm(N_EMBD)
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class CharLM(nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.block_size = BLOCK_SIZE
        self.tok_emb = nn.Embedding(vocab_size, N_EMBD)
        self.pos_emb = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.blocks  = nn.Sequential(*[Block() for _ in range(N_LAYER)])
        self.ln_f    = nn.LayerNorm(N_EMBD)
        self.lm_head = nn.Linear(N_EMBD, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.tok_emb(idx)
        pos_emb = self.pos_emb(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))
        return logits, loss

# ------------------------------------------------------------
# Inference wrapper
# ------------------------------------------------------------
class CharLMInference:
    def __init__(self, ckpt_dir: str = "model", device: str = "cpu") -> None:
        ckpt_path = Path(ckpt_dir)

        # tokenizer
        meta = json.loads((ckpt_path / "meta_20250805_112037.json").read_text())
        self.stoi = meta["stoi"]
        self.itos = {int(k): v for k, v in meta["itos"].items()}
        self.vocab_size = meta["vocab_size"]
        self.block_size = meta["block_size"]

        # config
        cfg = json.loads((ckpt_path / "config_20250805_112037.json").read_text())

        # build & load model
        self.device = torch.device(device)
        self.model = CharLM(vocab_size=self.vocab_size).to(self.device)
        state = torch.load(ckpt_path / "model_20250805_112037.pt", map_location=self.device)
        # map old keys if necessary
        key_map = {
            "tok_emb.weight": "tok_emb.weight",
            "pos_emb.weight": "pos_emb.weight",
            "head.weight": "lm_head.weight",
            "head.bias": "lm_head.bias"
        }
        new_state = {}
        for k, v in state.items():
            new_key = key_map.get(k, k)
            new_state[new_key] = v
        self.model.load_state_dict(new_state, strict=False)
        self.model.eval()

    # --------------------------------------------------------
    def encode(self, s: str) -> list[int]:
        return [self.stoi.get(c, 0) for c in s]

    def decode(self, idx: list[int]) -> str:
        return "".join(self.itos.get(i, "") for i in idx)

    # --------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        prompt: str = "Creativity is ",
        max_new_tokens: int = 26,
        temperature: float = 0.8,
        top_k: int | None = None,
    ) -> str:
        idx = torch.tensor(
            self.encode(prompt), dtype=torch.long, device=self.device
        ).unsqueeze(0)
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size :]
            logits = self.model(idx_cond)[0][:, -1, :]
            if temperature != 1.0:
                logits = logits / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)
        return self.decode(idx[0].tolist())

# ------------------------------------------------------------
# FastAPI application
# ------------------------------------------------------------
app = FastAPI(title="Tipus Micro-LLM API", version="1.0")

gen = CharLMInference("./model")

class Payload(BaseModel):
    prompt: str = Field(default="Creativity is ", description="Seed text")
    max_new_tokens: int = Field(default=26, ge=1, le=1024)
    temperature: float = Field(default=0.8, gt=0.0, le=2.0)
    top_k: int | None = Field(default=None, ge=1)

@app.get("/")
async def root():
    return {"message": "Tipus Micro-LLM API is running. POST to /generate"}

@app.post("/generate")
async def generate_endpoint(p: Payload):
    try:
        text = gen.generate(
            prompt=p.prompt,
            max_new_tokens=p.max_new_tokens,
            temperature=p.temperature,
            top_k=p.top_k
        )
        # trim trailing artefacts
        text = text[:-2] if text else text
        return {"text": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))