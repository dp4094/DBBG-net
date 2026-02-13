# DBBG-net: Chinese Named Entity Recognition

> A **PyTorch** implementation of a **GlobalPointer**-based NER model that supports both **nested** and **non-nested** entity recognition.

## 📖 Overview

### 🌟 Model Highlights

1. **Multiplicative Attention** — more efficient than additive attention.
2. **RoPE positional encoding** — rotary positional embeddings that enable relative-position effects via absolute positions.
3. **BiGRU enhancement** — bidirectional GRU for stronger sequence feature extraction.
4. **BBFE boundary feature extraction** — a lightweight CNN module to extract entity-boundary features.
5. **Regularization toolbox** — supports EMA, label smoothing, gradient clipping, and more.

## 📊 Results

| Dataset | Dev F1 | #Entity Types | Notes |
|---|---:|---:|---|
| CLUENER2020 | **0.8186** | 10 | Fine-grained Chinese NER |
| Weibo | **0.7407** | 4 | Chinese Weibo NER |

### Supported Entity Types

- **CLUENER2020**: address, book, company, game, government, movie, person, organization, position, scenic spot  
- **Weibo**: address, GPE (gpe.nom), person, organization

## 🚀 Quick Start

### Requirements

- Python **3.11+**
- CUDA (recommended for GPU training)
- GPU VRAM: **≥ 8GB** (e.g., RTX 4060 can run it)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/dp4094/DBBG-net.git
cd GlobalPointer_pytorch
```

2. **Create a virtual environment** (recommended: conda)
```bash
conda create -n test python=3.11.7
conda activate test
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download a pretrained encoder**

Download one of the following models into `pretrained_models/`:

- `bert-base-chinese` — general-purpose
- `Erlangshen-DeBERTa-v2-320M-Chinese` — typically better performance

## 🏋️ Training

1. **Edit config** in `config.py`:
```python
common = {
    "exp_name": "cluener",  # dataset name: cluener, weibo
    "encoder": "DeBERTa",   # or "BERT"
    "bert_path": "./pretrained_models/Erlangshen-DeBERTa-v2-320M-Chinese",
    "run_type": "train",
}
```

2. **Run training**
```bash
python train.py
```

Logs and checkpoints will be saved to:
`outputs/{exp_name}/{timestamp}/`

## 🧪 Evaluation

1. **Edit config** in `config.py`:
```python
common = {
    "run_type": "eval",
}

eval_config = {
    "model_state_dir": "./outputs/cluener/2025-01-01_12.00.00",  # checkpoint dir
}
```

2. **Run evaluation**
```bash
python evaluate.py
```

Predictions will be saved to:
`results/{exp_name}/`

## 📁 Project Structure

```text
GlobalPointer_pytorch/
├── train.py                 # training script
├── evaluate.py              # evaluation & inference
├── config.py                # main configs
├── config_ner_datasets.py   # dataset config examples
├── requirements.txt         # dependencies
├── README.md                # documentation
├── LICENSE                  # license
│
├── models/                  # model definitions
│   └── GlobalPointer.py     # GlobalPointer model
│
├── common/                  # utilities
│   ├── utils.py             # general utilities
│   ├── augment.py           # data augmentation
│   └── ema.py               # exponential moving average
│
├── data_utils/              # data processing
│   └── data_use.py
│
├── datasets/                # datasets
│   ├── cluener/             # CLUENER2020
│   │   ├── train.json
│   │   ├── dev.json
│   │   ├── test.json
│   │   └── ent2id.json
│   └── weibo/               # Weibo
│       ├── train.json
│       ├── dev.json
│       ├── test.json
│       └── ent2id.json
│
├── pretrained_models/       # pretrained encoders (download yourself)
│   └── bert-base-chinese/
│
├── outputs/                 # training outputs
│   └── {exp_name}/
│       └── {timestamp}/
│           ├── model_state_dict_*.pt
│           └── training_log_*.txt
│
└── results/                 # evaluation outputs
    └── {exp_name}/
        └── predict_result.json
```

## ⚙️ Configuration

You can tune the following in `config.py`:

### Basic

- `exp_name`: experiment/dataset name (`cluener`, `weibo`)
- `encoder`: encoder type (`BERT`, `DeBERTa`)
- `bert_path`: path to pretrained model
- `run_type`: run mode (`train`, `eval`)

### Training Hyperparameters

- `lr`: learning rate (default `5e-6`)
- `batch_size`: batch size (default `12`)
- `epochs`: number of epochs (default `20`)
- `max_seq_len`: max sequence length (default `256`)
- `gru_hidden_size`: GRU hidden size (default `320`)
- `dropout`: dropout rate (default `0.2`)

### Advanced Features

- `use_data_augment`: enable augmentation
- `use_ema`: enable EMA
- `use_mixed_precision`: enable mixed precision training
- `use_label_smoothing`: enable label smoothing

## 📝 Data Format

### Training data (JSON Lines)

One JSON object per line:

```json
{
  "text": "浙商银行企业信贷部叶老桂博士则从另一个角度对五道门槛进行了解读。",
  "label": {
    "name": {
      "叶老桂": [[9, 12]]
    },
    "company": {
      "浙商银行": [[0, 4]]
    },
    "position": {
      "博士": [[12, 14]]
    }
  }
}
```

### Entity mapping file (`ent2id.json`)

```json
{
  "address": 0,
  "book": 1,
  "company": 2,
  "name": 3
}
```

## 🔧 Advanced Features

### 1) Data Augmentation

Enable in `config.py`:

```python
"use_data_augment": True,
"augment_prob": 0.3,
"augment_methods": ["synonym", "replace", "swap"]
```

### 2) Exponential Moving Average (EMA)

Helps generalization:

```python
"use_ema": True,
"ema_decay": 0.999,
"ema_update_interval": 5
```

### 3) Mixed Precision Training

Speeds up training and reduces memory usage:

```python
"use_mixed_precision": True
```

### 4) Model Ensemble

Configure in `evaluate.py`:

```python
"use_ensemble": True,
"ensemble_models": 5  # ensemble the best 5 checkpoints
```

### 5) Per-Entity-Type Thresholds

Set different decode thresholds by entity type:

```python
"entity_thresholds": {
    "address": 0.01,   # lower threshold for address
    "name": 0.04,      # higher threshold for person name
    "default": 0.02
}
```

## 🎯 Tips

### Dataset-specific tuning

- **Small datasets** (e.g., Weibo): increase epochs, enable augmentation.
- **Larger datasets** (e.g., CLUENER): slightly lower LR, increase batch size if VRAM allows.

### VRAM optimization

If you run out of memory:

- reduce `batch_size`
- reduce `max_seq_len`
- reduce `gru_hidden_size`
- disable augmentation

### Improving performance

- use a DeBERTa encoder (often better than BERT)
- enable EMA
- use ensembling
- tune per-type thresholds
