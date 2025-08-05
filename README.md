## Tipus Micro-LLM

[![Lint](https://github.com/0xnu/tipus-micro-llm/actions/workflows/lint.yaml/badge.svg)](https://github.com/0xnu/tipus-micro-llm/actions/workflows/lint.yaml)
[![Release](https://img.shields.io/github/release/0xnu/tipus-micro-llm.svg)](https://github.com/0xnu/tipus-micro-llm/releases/latest)
[![License](https://img.shields.io/badge/License-Modified_MIT-f5de53?&color=f5de53)](/LICENSE)

A minimal character-level language model implemented in pure PyTorch, featuring:

- Transformer decoder architecture with causal masking
- FastAPI-based REST API for text generation
- Training on next-character prediction

> [!IMPORTANT]
> Give me GPUs, and I'll train open-source LLMs with internet access for shits and giggles. 😁 😎

### Features

- Character-level tokenization
- Multi-head self-attention
- Layer normalization and dropout
- Positional embeddings
- Temperature and top-k sampling

### Model Architecture

- Block size: 128 tokens
- 6 transformer layers
- 8 attention heads
- 512 embedding dimensions
- Dropout rate: 0.1

### Install Dependencies

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install following

```python
## Prerequisites
python3 -m venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
python3 -m pip install --upgrade pip
```

### Training

```sh
python3 -m tipus

## Training time on Apple M4 Macbook Pro 16GB (Memory) and 1TB (Storage) is 178 minutes or 2.97 hours
```

The model will:

1. Load training data from `data/corpus.txt`
2. Train for 5000 iterations
3. Save model checkpoints to `model/` directory

### API Usage

Start the FastAPI server:

```sh
uvicorn serve:app --host 0.0.0.0 --port 2025
```

### Generate Text

```sh
curl -X POST http://localhost:2025/generate \
     -H "Content-Type: application/json" \
     -d '{"prompt":"Creativity is ", "max_new_tokens":26, "temperature":0.8, "top_k": 1}'
```

### API Parameters

- `prompt`: Initial text to continue
- `max_new_tokens`: Maximum length of generated text
- `temperature`: Controls randomness (lower = more deterministic)
- `top_k`: Limits vocabulary to top-k most likely tokens

### License

This project is licensed under the [Modified MIT License](./LICENSE).

### Copyright

(c) 2025 [Finbarrs Oketunji](https://finbarrs.eu). All Rights Reserved.
