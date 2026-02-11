DBBG-net中文命名实体识别

> 基于PyTorch实现的GlobalPointer命名实体识别模型，支持嵌套和非嵌套实体识别


## 📖 项目简介



### 🌟 模型特点

1. **乘性Attention机制** - 相比加性Attention更高效
2. **RoPE位置编码** - 旋转式位置编码，通过绝对位置实现相对位置编码
3. **BiGRU增强** - 使用双向GRU提取序列特征
4. **BBFE边界特征提取** - 轻量级CNN提取实体边界特征
5. **多种正则化技术** - 支持EMA、标签平滑、梯度裁剪等

## 📊 性能表现

| 数据集 | 验证集F1 | 实体类型数 | 说明 |
|--------|---------|-----------|------|
| CLUENER2020 | **0.8186** | 10 | 细粒度命名实体识别 |
| Weibo | **0.7407** | 4 | 微博命名实体识别 |


### 支持的实体类型

**CLUENER2020**: 地址、书籍、公司、游戏、政府、电影、人名、组织、职位、景点

**Weibo**: 地址、地理政治实体(gpe.nom)、人名、组织

## 🚀 快速开始

### 环境要求

- Python 3.11+
- CUDA（推荐，用于GPU训练）
- 显存：至少8GB（RTX 4060可运行）

### 安装步骤

1. **克隆项目**
```bash
git clone https://github.com/dp4094/DBBG-net.git
cd GlobalPointer_pytorch
```

2. **创建虚拟环境**（推荐使用conda）
```bash
conda create -n test python=3.11.7
conda activate test
```

3. **安装依赖**
```bash
pip install -r requirements.txt
```

4. **下载预训练模型**

下载以下模型之一到 `pretrained_models/` 目录：

- [bert-base-chinese](https://huggingface.co/bert-base-chinese) - 适合通用场景
- [Erlangshen-DeBERTa-v2-320M-Chinese](https://huggingface.co/IDEA-CCNL/Erlangshen-DeBERTa-v2-320M-Chinese) - 性能更好

### 训练模型

1. **修改配置文件** `config.py`：
```python
common = {
    "exp_name": "cluener",  # 数据集名称：cluener, weibo
    "encoder": "DeBERTa",  # 或 "DeBERTa"
    "bert_path": "./pretrained_models/Erlangshen-DeBERTa-v2-320M-Chinese",
    "run_type": "train",
}
```

2. **开始训练**：
```bash
python train.py
```

训练日志和模型检查点将保存在 `outputs/{exp_name}/{timestamp}/` 目录。

### 评估模型

1. **修改配置文件** `config.py`：
```python
common = {
    "run_type": "eval",
}

eval_config = {
    "model_state_dir": "./outputs/cluener/2025-01-01_12.00.00",  # 模型路径
}
```

2. **运行评估**：
```bash
python evaluate.py
```

预测结果将保存在 `results/{exp_name}/` 目录。

## 📁 项目结构

```
GlobalPointer_pytorch/
├── train.py                 # 训练脚本
├── evaluate.py              # 评估和预测脚本
├── config.py                # 配置文件
├── config_ner_datasets.py   # 数据集配置示例
├── requirements.txt         # 依赖列表
├── README.md                # 项目文档
├── LICENSE                  # 开源许可证
│
├── models/                  # 模型定义
│   └── GlobalPointer.py     # GlobalPointer模型
│
├── common/                  # 工具模块
│   ├── utils.py             # 通用工具函数
│   ├── augment.py           # 数据增强
│   └── ema.py               # 指数移动平均
│
├── data_utils/              # 数据处理
│   └── data_use.py
│
├── datasets/                # 数据集目录
│   ├── cluener/             # CLUENER2020数据集
│   │   ├── train.json
│   │   ├── dev.json
│   │   ├── test.json
│   │   └── ent2id.json
│   └── weibo/               # Weibo数据集
│       ├── train.json
│       ├── dev.json
│       ├── test.json
│       └── ent2id.json
│
├── pretrained_models/       # 预训练模型（需自行下载）
│   └── bert-base-chinese/
│
├── outputs/                 # 训练输出
│   └── {exp_name}/
│       └── {timestamp}/
│           ├── model_state_dict_*.pt
│           └── training_log_*.txt
│
└── results/                 # 评估结果
    └── {exp_name}/
        └── predict_result.json
```

## ⚙️ 配置说明

### 主要配置参数

在 `config.py` 中可以调整以下参数：

#### 基础配置
- `exp_name`: 实验名称/数据集名称（cluener, weibo）
- `encoder`: 编码器类型（BERT, DeBERTa）
- `bert_path`: 预训练模型路径
- `run_type`: 运行类型（train, eval）

#### 训练超参数
- `lr`: 学习率（默认5e-6）
- `batch_size`: 批次大小（默认12）
- `epochs`: 训练轮数（默认20）
- `max_seq_len`: 最大序列长度（默认256）
- `gru_hidden_size`: GRU隐藏层大小（默认320）
- `dropout`: Dropout比例（默认0.2）

#### 高级功能
- `use_data_augment`: 是否使用数据增强
- `use_ema`: 是否使用指数移动平均
- `use_mixed_precision`: 是否使用混合精度训练
- `use_label_smoothing`: 是否使用标签平滑

## 📝 数据格式

### 训练数据格式

每行一个JSON对象：
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

### 实体映射文件 (ent2id.json)

```json
{
  "address": 0,
  "book": 1,
  "company": 2,
  "name": 3
}
```

## 🔧 高级功能

### 1. 数据增强

在 `config.py` 中启用：
```python
"use_data_augment": True,
"augment_prob": 0.3,
"augment_methods": ["synonym", "replace", "swap"]
```

### 2. 指数移动平均 (EMA)

提高模型泛化能力：
```python
"use_ema": True,
"ema_decay": 0.999,
"ema_update_interval": 5
```

### 3. 混合精度训练

加速训练并减少显存占用：
```python
"use_mixed_precision": True
```

### 4. 模型集成

在 `evaluate.py` 中配置：
```python
"use_ensemble": True,
"ensemble_models": 5  # 集成最佳的5个模型
```

### 5. 实体类型特定阈值

为不同实体类型设置不同的解码阈值：
```python
"entity_thresholds": {
    "address": 0.01,  # 地址使用较低阈值
    "name": 0.04,     # 人名使用较高阈值
    "default": 0.02
}
```

## 🎯 使用技巧

### 1. 针对不同数据集调优

- **小数据集**（如Weibo）：增加epochs，使用数据增强
- **大数据集**（如CLUENER）：适当降低学习率，增加batch_size

### 2. 显存优化

如果显存不足，可以：
- 减小 `batch_size`
- 减小 `max_seq_len`
- 减小 `gru_hidden_size`
- 禁用数据增强

### 3. 提高性能

- 使用DeBERTa编码器（比BERT性能更好）
- 启用EMA
- 使用模型集成
- 调整实体类型特定的阈值




