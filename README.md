## Tipus Micro-LLM

[![Lint](https://github.com/0xnu/tipus-micro-llm/actions/workflows/lint.yaml/badge.svg)](https://github.com/0xnu/tipus-micro-llm/actions/workflows/lint.yaml)
[![Release](https://img.shields.io/github/release/0xnu/tipus-micro-llm.svg)](https://github.com/0xnu/tipus-micro-llm/releases/latest)
[![License](https://img.shields.io/badge/License-Modified_MIT-f5de53?&color=f5de53)](/LICENSE)

Character-level and token-based language models implemented in pure PyTorch, featuring:

- Character-level language model with:
  - 128 token block size
  - 6 transformer layers
  - 8 attention heads
  - 512 embedding dimensions
- Token-based language model with:
  - 256 token block size
  - 8 transformer layers
  - 8 attention heads
  - 768 embedding dimensions
- Transformer decoder architecture with causal masking
- Training on next-character prediction (character-level)
- Training on next-token prediction (token-based)
- FastAPI-based REST API for text generation

> [!IMPORTANT]
> Give me GPUs, and I'll train open-source LLMs with internet access for shits and giggles. 😁 😎

### Model Architecture (Character-level)

- Block size: 128 tokens
- 6 transformer layers
- 8 attention heads
- 512 embedding dimensions
- Dropout rate: 0.1
- Batch size: 64
- Maximum iterations: 5,000
- Learning rate: 3e-4

### Model Architecture (Token-based)

- Block size: 256 tokens
- 8 transformer layers
- 8 attention heads
- 768 embedding dimensions
- Dropout rate: 0.1
- Batch size: 32
- Maximum iterations: 10,000
- Learning rate: 1e-4

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
python3 -m tipus ## Character-level language model
python3 -m tipus_pairs ## Token-based language model
```

#### Training Comparison (Character-level)

Training time for [corpus.txt](./data/corpus.txt) with 5000 iterations:

| **Device**                     | **Training Time**         | **Equivalent in Hours**  | **Iterations**  |
|--------------------------------|---------------------------|--------------------------|-----------------|
| Apple M4 MacBook Pro (16GB RAM, 1TB Storage) | 178 minutes               | 2.97 hours |   5000          |
| NVIDIA Tesla P100 GPU                        | 20 minutes 😂             | 0.33 hours |   5000          |

The model will:

1. Load training data from `data/corpus.txt`
2. Train for 5000 iterations
3. Save model checkpoints to `model/` directory

#### Training Comparison (Token-based)

Training time for [qa_pairs.json](./data/qa_pairs.json) with 10,000 iterations:

| **Device**                     | **Training Time**         | **Equivalent in Hours**  | **Iterations**   |
|--------------------------------|---------------------------|--------------------------|------------------|
| Apple M4 MacBook Pro (16GB RAM, 1TB Storage) | 680 minutes               | 11.33 hours|   10000          |
| NVIDIA Tesla P100 GPU                        | 25 minutes 😂             | 0.42 hours |   10000          |

The model will:

1. Load training data from `data/qa_pairs.json`
2. Train for 10,000 iterations
3. Save model checkpoints to `model/` directory

### API Usage

Start the FastAPI server:

```sh
uvicorn serve:app --host 0.0.0.0 --port 2025 ## Character-level language model
uvicorn serve_pairs:app --host 0.0.0.0 --port 2025 ## Token-based language model
```

### Generate Text (Character-level)

```sh
curl -X POST http://localhost:2025/generate \
     -H "Content-Type: application/json" \
     -d '{"prompt":"Creativity is ", "max_new_tokens":26, "temperature":0.8, "top_k": 1}'
```

#### API Parameters (Character-level)

- `prompt`: Initial text to continue
- `max_new_tokens`: Maximum length of generated text
- `temperature`: Controls randomness (lower = more deterministic)
- `top_k`: Limits vocabulary to top-k most likely tokens

### Generate Text (Token-based)

```sh
## Token-based
curl -X POST http://localhost:2025/generate \
     -H "Content-Type: application/json" \
     -d '{"question":"What is the capital of France?", "max_length":120, "temperature":0.7}'
```

### API Parameters (Token-based)

- `question`: Your question (e.g., "What is the capital of France?")
- `max_length`: Maximum length of generated text
- `temperature`: Controls randomness (lower = more deterministic)

### License

This project is licensed under the [Modified MIT License](./LICENSE).

### Copyright

(c) 2025 [Finbarrs Oketunji](https://finbarrs.eu). All Rights Reserved.