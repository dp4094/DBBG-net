# 训练工作流程说明

本文档详细说明 GlobalPointer 项目的训练工作流程。

## 训练流程概述

执行 `python train.py` 时，程序会按以下顺序执行：

```
1. 加载配置和数据
   ↓
2. 创建模型和优化器
   ↓
3. 训练循环（每个 epoch）
   ├─ 在训练集上训练
   ├─ 在验证集（dev.json）上验证
   └─ 保存最佳模型（如果 F1 > f1_2_save）
   ↓
4. 训练完成
   ↓
5. 自动在测试集上评估（可选）
   └─ 加载最佳模型
   └─ 在测试集（test.json）上评估
   └─ 输出最终 F1 分数
```

## 详细说明

### 1. 训练阶段

在每个 epoch 中：

- **训练集（train.json）**: 用于训练模型参数
- **验证集（dev.json）**: 用于验证模型性能，选择最佳模型

```python
for epoch in range(epochs):
    # 训练
    train(model, train_dataloader)
    
    # 验证
    valid_f1 = valid(model, valid_dataloader)
    
    # 保存最佳模型
    if valid_f1 > max_f1 and valid_f1 > f1_2_save:
        save_model(model)
```

### 2. 模型保存策略

模型保存条件：

- 验证集 F1 分数 > 历史最佳 F1
- 验证集 F1 分数 > `f1_2_save` 阈值（默认 0.72）

保存位置：

```
outputs/{dataset_name}/{timestamp}/
└── model_state_dict_0.pt  # 第一个达到阈值的模型
└── model_state_dict_1.pt  # 第二个达到阈值的模型
└── ...
```

### 3. 自动测试功能（新增）

**默认行为**：训练完成后自动在测试集上评估

配置选项：

```python
# config.py
common = {
    "auto_test_after_train": True,  # 是否自动测试（默认 True）
}
```

**执行流程**：

1. 训练完成后，查找保存的模型
2. 加载最后保存的模型（通常是最佳模型）
3. 在测试集（test.json）上评估
4. 输出最终 F1 分数和损失

**输出示例**：

```
============================================================
训练完成！开始在测试集上评估最佳模型...
============================================================
加载最佳模型: ./outputs/weibo/2025-02-09_16.30.00/model_state_dict_0.pt
在测试集上评估...
******************************************
总体评估指标 - Precision: 0.7234, Recall: 0.7456, F1: 0.7343
******************************************

============================================================
测试集最终结果 - F1: 0.7343, Loss: 0.234567
============================================================
```

### 4. 关闭自动测试

如果只想训练而不测试，可以修改配置：

```python
# config.py
common = {
    "auto_test_after_train": False,  # 关闭自动测试
}
```

或者使用命令行（如果实现了）：

```bash
python train.py --no-auto-test
```

## 数据集使用说明

### 数据集划分

- **train.json**: 训练集，用于训练模型
- **dev.json**: 验证集，用于模型选择和超参数调优
- **test.json**: 测试集，用于最终评估

### 为什么需要三个数据集？

1. **训练集（train.json）**
   - 用途：训练模型参数
   - 特点：模型会"记住"这些数据

2. **验证集（dev.json）**
   - 用途：选择最佳模型，调整超参数
   - 特点：用于模型选择，但不用于训练

3. **测试集（test.json）**
   - 用途：评估模型的真实性能
   - 特点：只在最后使用一次，避免过拟合

### 数据集大小建议

典型划分比例：

- 训练集：70-80%
- 验证集：10-15%
- 测试集：10-15%

## 完整训练示例

### 示例1：训练 Weibo 数据集

```bash
# 1. 确认配置
# config.py:
#   exp_name = "weibo"
#   run_type = "train"
#   auto_test_after_train = True

# 2. 运行训练
python train.py

# 输出：
# - 训练 20 个 epoch
# - 每个 epoch 在验证集上验证
# - 保存最佳模型
# - 训练完成后在测试集上评估
```

### 示例2：训练 CLUENER 数据集

```bash
# 1. 修改配置
# config.py:
#   exp_name = "cluener"
#   run_type = "train"

# 2. 运行训练
python train.py
```

### 示例3：只训练不测试

```bash
# 1. 修改配置
# config.py:
#   auto_test_after_train = False

# 2. 运行训练
python train.py

# 训练完成后不会自动测试
```

## 手动评估测试集

如果关闭了自动测试，可以手动评估：

### 方法1：使用 evaluate.py

```bash
# 1. 修改 config.py
# eval_config:
#   model_state_dir = "./outputs/weibo/2025-02-09_16.30.00"
# common:
#   run_type = "eval"

# 2. 运行评估
python evaluate.py
```

### 方法2：使用 train.py 的 eval 模式

```bash
# 1. 修改 config.py
# common:
#   run_type = "eval"

# 2. 运行评估
python train.py
```

## 训练监控

### 查看训练日志

```bash
# 实时查看日志
tail -f outputs/{dataset_name}/{timestamp}/training_log_*.txt

# Windows
Get-Content outputs\{dataset_name}\{timestamp}\training_log_*.txt -Wait
```

### 使用 WandB 监控

```bash
# 1. 安装 wandb
pip install wandb

# 2. 登录
wandb login

# 3. 修改配置
# config.py:
#   logger = "wandb"

# 4. 运行训练
python train.py

# 5. 在浏览器中查看
# https://wandb.ai/your-username/GlobalPointer_weibo
```

## 常见问题

### Q1: 训练完成后没有在测试集上评估？

**原因**：
- `auto_test_after_train = False`
- 或者没有保存的模型（F1 未达到阈值）

**解决**：
- 检查配置：`auto_test_after_train = True`
- 降低保存阈值：`f1_2_save = 0.5`

### Q2: 为什么没有保存模型？

**原因**：验证集 F1 分数未达到 `f1_2_save` 阈值

**解决**：
- 降低阈值：`f1_2_save = 0.5` 或 `0.6`
- 增加训练轮数
- 调整超参数

### Q3: 测试集 F1 比验证集低很多？

**原因**：可能过拟合到验证集

**解决**：
- 使用数据增强
- 增加 dropout
- 使用 EMA
- 减少训练轮数

### Q4: 如何只在测试集上评估而不训练？

**方法**：

```python
# config.py
common = {
    "run_type": "eval",  # 改为 eval 模式
}

eval_config = {
    "model_state_dir": "./outputs/weibo/xxx",  # 指定模型目录
}
```

然后运行：

```bash
python train.py
# 或
python evaluate.py
```

## 最佳实践

1. **训练时**：
   - 使用验证集选择最佳模型
   - 不要频繁查看测试集结果
   - 保存多个检查点

2. **评估时**：
   - 只在最后使用测试集
   - 报告测试集结果时要诚实
   - 不要根据测试集调整模型

3. **超参数调优**：
   - 只使用训练集和验证集
   - 不要使用测试集调参
   - 使用交叉验证（如果数据充足）

## 总结

- ✅ 训练时使用训练集和验证集
- ✅ 验证集用于选择最佳模型
- ✅ 训练完成后自动在测试集上评估（默认）
- ✅ 可以通过配置关闭自动测试
- ✅ 测试集只在最后使用一次

这样的工作流程确保了模型评估的公正性和可靠性。
