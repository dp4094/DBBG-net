# 快速开始指南

本指南将帮助您快速开始使用 GlobalPointer 进行中文命名实体识别。

## 前提条件

确保您已经完成了[安装步骤](installation.md)：

- ✅ Python 环境已配置
- ✅ 依赖包已安装
- ✅ 预训练模型已下载
- ✅ 数据集已准备好

## 验证环境

在开始训练前，建议先运行测试脚本验证环境：

```bash
# 测试配置
python tests/test_config.py

# 测试训练流程（快速测试）
python tests/test_train_flow.py
```

如果所有测试通过，说明环境配置正确。

## 训练模型

### 方法1：使用 train.py（推荐）

#### 训练 Weibo 数据集

```bash
# 1. 修改 config.py
# 将 common["exp_name"] 设置为 "weibo"
# 将 common["run_type"] 设置为 "train"

# 2. 运行训练
python train.py
```

#### 训练 CLUENER 数据集

```bash
# 1. 修改 config.py
# 将 common["exp_name"] 设置为 "cluener"
# 将 common["run_type"] 设置为 "train"

# 2. 运行训练
python train.py
```

### 方法2：使用 main.py（命令行接口）

#### 训练 Weibo 数据集

```bash
python main.py train --dataset weibo
```

#### 训练 CLUENER 数据集

```bash
python main.py train --dataset cluener
```

#### 自定义训练参数

```bash
# 自定义训练轮数和批次大小
python main.py train --dataset weibo --epochs 30 --batch-size 16

# 自定义学习率
python main.py train --dataset cluener --lr 2e-5
```

## 训练配置

### 关键参数说明

在 `config.py` 中，您可以调整以下关键参数：

```python
# 基础配置
common = {
    "exp_name": "weibo",      # 数据集名称：weibo 或 cluener
    "encoder": "DeBERTa",     # 编码器：BERT 或 DeBERTa
    "run_type": "train",      # 运行模式：train 或 eval
    "f1_2_save": 0.72,        # 保存模型的F1阈值
}

# 训练超参数
hyper_parameters = {
    "lr": 5e-6,               # 学习率（DeBERTa建议5e-6，BERT建议2e-5）
    "batch_size": 12,         # 批次大小（根据显存调整）
    "epochs": 20,             # 训练轮数
    "max_seq_len": 256,       # 最大序列长度
    "gru_hidden_size": 320,   # GRU隐藏层大小
    "dropout": 0.2,           # Dropout概率
    
    # 数据增强
    "use_data_augment": True, # 是否使用数据增强
    "augment_prob": 0.3,      # 数据增强概率
    
    # EMA
    "use_ema": True,          # 是否使用指数移动平均
    "ema_decay": 0.999,       # EMA衰减率
    
    # 混合精度训练
    "use_mixed_precision": True,  # 是否使用FP16训练
}
```

### 针对不同显卡的配置建议

#### 6GB 显存（如 RTX 2060）

```python
hyper_parameters = {
    "batch_size": 8,          # 减小批次大小
    "max_seq_len": 256,       # 保持序列长度
    "use_mixed_precision": True,  # 必须开启混合精度
}
```

#### 8GB 显存（如 RTX 3060）

```python
hyper_parameters = {
    "batch_size": 12,         # 默认配置
    "max_seq_len": 256,
    "use_mixed_precision": True,
}
```

#### 12GB+ 显存（如 RTX 3080）

```python
hyper_parameters = {
    "batch_size": 16,         # 可以增大批次
    "max_seq_len": 512,       # 可以增加序列长度
    "use_mixed_precision": True,
}
```

## 监控训练过程

### 查看训练日志

训练日志会保存在：

```
outputs/{dataset_name}/{timestamp}/training_log_{timestamp}.txt
```

### 实时监控（使用 WandB）

如果想使用 WandB 进行可视化监控：

```bash
# 1. 安装 wandb
pip install wandb

# 2. 登录 wandb
wandb login

# 3. 修改 config.py
common["logger"] = "wandb"

# 4. 运行训练
python train.py
```

## 训练输出

训练完成后，模型会保存在：

```
outputs/{dataset_name}/{timestamp}/
├── model_state_dict_0.pt    # 第一个达到阈值的模型
├── model_state_dict_1.pt    # 第二个达到阈值的模型
└── training_log_*.txt       # 训练日志
```

## 评估模型

### 使用 evaluate.py

```bash
# 1. 修改 config.py 中的 eval_config
eval_config = {
    "model_state_dir": "./outputs/weibo/2025-02-09_16.30.00",  # 模型目录
    "predict_data": "test.json",
}

# 2. 修改 common 配置
common = {
    "exp_name": "weibo",
    "run_type": "eval",  # 改为 eval 模式
}

# 3. 运行评估
python evaluate.py
```

### 使用 main.py

```bash
python main.py eval --dataset weibo --model-dir ./outputs/weibo/2025-02-09_16.30.00
```

## 预测新文本

创建一个简单的预测脚本：

```python
import torch
from transformers import AutoTokenizer
from models.GlobalPointer import GlobalPointer
import json

# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("./pretrained_models/Erlangshen-DeBERTa-v2-320M-Chinese")

# 加载实体映射
with open("./datasets/weibo/ent2id.json") as f:
    ent2id = json.load(f)

# 创建模型并加载权重
model = GlobalPointer(encoder, len(ent2id), 64)
model.load_state_dict(torch.load("./outputs/weibo/xxx/model_state_dict_0.pt"))
model = model.to(device)
model.eval()

# 预测
text = "华为公司在深圳成立"
inputs = tokenizer(text, return_tensors="pt").to(device)
with torch.no_grad():
    logits = model(**inputs)

# 解码实体
# ... 解码逻辑
```

## 常见问题

### 1. 显存不足

**错误**: `RuntimeError: CUDA out of memory`

**解决方案**:
- 减小 `batch_size`
- 减小 `max_seq_len`
- 开启 `use_mixed_precision`
- 关闭 `use_data_augment`

### 2. 训练速度慢

**解决方案**:
- 开启 `use_mixed_precision`
- 增大 `batch_size`（如果显存允许）
- 减少 `augment_prob`
- 使用更快的GPU

### 3. F1分数低

**解决方案**:
- 增加训练轮数 `epochs`
- 开启数据增强 `use_data_augment`
- 调整学习率 `lr`
- 使用 EMA `use_ema`
- 调整解码阈值

### 4. 训练不收敛

**解决方案**:
- 降低学习率
- 增加 `warmup_ratio`
- 检查数据格式是否正确
- 减小 `label_smoothing_factor`

## 性能基准

### Weibo 数据集

- **验证集 F1**: ~0.74
- **训练时间**: ~2小时（RTX 4060, 20 epochs）
- **推荐配置**: DeBERTa, lr=5e-6, batch_size=12

### CLUENER 数据集

- **验证集 F1**: ~0.80
- **训练时间**: ~3小时（RTX 4060, 20 epochs）
- **推荐配置**: DeBERTa, lr=5e-6, batch_size=12

## 下一步

- 查看 [使用说明](usage.md) 了解更多高级功能
- 查看 [API文档](api.md) 了解代码结构
- 查看 [性能报告](performance.md) 了解详细的性能分析

## 获取帮助

如果遇到问题：

1. 查看 [常见问题](../README.md#常见问题)
2. 运行测试脚本诊断问题
3. 提交 [Issue](https://github.com/your-username/GlobalPointer_pytorch/issues)
