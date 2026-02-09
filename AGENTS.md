# GlobalPointer项目开发规范

## 项目结构说明

### 核心文件
- `train.py` - 训练入口脚本，包含完整的训练流程
- `evaluate.py` - 评估和预测脚本，支持模型集成
- `config.py` - 配置管理，包含所有超参数和数据集特定配置
- `config_ner_datasets.py` - 数据集配置示例，方便切换不同数据集

### 模型模块
- `models/GlobalPointer.py` - GlobalPointer模型定义
  - `MyDataset` - 数据集类
  - `DataMaker` - 数据预处理和批次生成
  - `GlobalPointer` - 主模型类（包含BiGRU和BBFE模块）
  - `MetricsCalculator` - 评估指标计算

### 工具模块
- `common/utils.py` - 通用工具函数（预处理、损失函数等）
- `common/augment.py` - 数据增强模块
- `common/ema.py` - 指数移动平均（EMA）实现

### 数据目录
- `datasets/` - 数据集存放目录
  - `cluener/` - CLUENER2020数据集
  - `weibo/` - Weibo NER数据集
  - `converted_ner/` - 转换后的其他数据集（可选）
- `pretrained_models/` - 预训练模型（BERT/DeBERTa）
- `outputs/` - 训练输出和模型检查点
- `results/` - 评估结果

## 开发命令

### 环境配置

```bash
# 创建conda环境
conda create -n CLUENER2020 python=3.6
conda activate CLUENER2020

# 安装依赖
pip install -r requirements.txt

# 下载预训练模型
# 手动下载bert-base-chinese或DeBERTa模型到pretrained_models/目录
```

### 训练模型

```bash
# 1. 修改config.py中的配置
# - exp_name: 数据集名称（cluener, weibo）
# - bert_path: 预训练模型路径
# - run_type: "train"

# 2. 开始训练
python train.py

# 训练输出将保存在 outputs/{exp_name}/{timestamp}/
```

### 评估模型

```bash
# 1. 修改config.py中的配置
# - run_type: "eval"
# - model_state_dir: 模型检查点目录

# 2. 运行评估
python evaluate.py

# 预测结果将保存在 results/{exp_name}/
```

### 数据集切换

```bash
# 方法1：直接修改config.py
# 修改 common["exp_name"] = "cluener" 或 "weibo"

# 方法2：使用config_ner_datasets.py（如果有其他数据集）
# 在代码中导入并使用：
# from config_ner_datasets import switch_to_weibo
# train_config = switch_to_weibo()
```

## 代码规范

### 命名约定

- **类名**：PascalCase（如 `GlobalPointer`, `DataMaker`）
- **函数名**：snake_case（如 `train_step`, `load_data`）
- **常量**：UPPER_CASE（如 `BASE_CONFIG`, `MAX_SEQ_LEN`）
- **私有变量**：_leading_underscore（如 `_hidden_state`）

### 文档字符串

使用Google风格的docstring：

```python
def function_name(param1, param2):
    """函数简短描述。
    
    详细描述（可选）。
    
    Args:
        param1 (type): 参数1说明
        param2 (type): 参数2说明
    
    Returns:
        type: 返回值说明
    
    Raises:
        ExceptionType: 异常说明（如果有）
    """
    pass
```

### 代码注释

- 为复杂逻辑添加注释
- 注释应该解释"为什么"而不是"是什么"
- 使用中文注释（项目主要面向中文用户）

### 配置管理

- 所有超参数集中在 `config.py` 中管理
- 为每个参数添加注释说明其用途
- 支持通过环境变量覆盖配置
- 数据集特定配置使用 `get_dataset_specific_configs()` 函数

## 数据集规范

### 目录结构

```
datasets/
└── {dataset_name}/
    ├── train.json       # 训练集
    ├── dev.json         # 验证集
    ├── test.json        # 测试集
    └── ent2id.json      # 实体类型映射
```

### 数据格式

**训练/验证/测试数据**（JSON Lines格式）：
```json
{
  "text": "文本内容",
  "label": {
    "实体类型": {
      "实体文本": [[起始位置, 结束位置]]
    }
  }
}
```

**实体映射文件**（JSON格式）：
```json
{
  "实体类型1": 0,
  "实体类型2": 1,
  "实体类型3": 2
}
```

### 数据集要求

- 文本编码：UTF-8
- 位置索引：字符级别（不是token级别）
- 实体位置：左闭右开区间 [start, end)

## Git规范

### 提交信息格式

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Type类型**：
- `feat`: 新功能
- `fix`: 修复bug
- `docs`: 文档更新
- `style`: 代码格式（不影响功能）
- `refactor`: 代码重构
- `perf`: 性能优化
- `test`: 测试相关
- `chore`: 构建过程或辅助工具的变动

**示例**：
```
feat(model): 添加BBFE边界特征提取模块

- 实现轻量级1D-CNN提取头尾边界特征
- 添加门控融合机制
- 在CLUENER数据集上F1提升0.5个百分点

Closes #123
```

### 分支策略

- `main` - 稳定版本，用于发布
- `develop` - 开发版本，日常开发
- `feature/*` - 功能分支，开发新功能
- `fix/*` - 修复分支，修复bug
- `release/*` - 发布分支，准备发布

### Pull Request规范

1. **标题**：简洁描述变更内容
2. **描述**：详细说明变更原因和实现方式
3. **测试**：说明如何测试，提供测试结果
4. **截图**：如果涉及UI或输出变化，提供截图
5. **关联Issue**：使用 `Closes #123` 关联相关Issue

## 测试规范

### 单元测试

- 使用 `pytest` 框架
- 测试文件命名：`test_*.py`
- 测试函数命名：`test_*`
- 放置位置：`tests/` 目录

### 测试覆盖

- 核心功能必须有测试
- 目标覆盖率：>80%
- 运行测试：`pytest tests/`

### 手动测试

在提交前进行以下手动测试：

1. **训练测试**：
   ```bash
   python train.py  # 至少训练1个epoch
   ```

2. **评估测试**：
   ```bash
   python evaluate.py  # 使用训练好的模型
   ```

3. **数据加载测试**：
   - 验证数据格式正确
   - 检查实体位置准确

## 性能优化建议

### 训练加速

1. **使用混合精度训练**：
   ```python
   "use_mixed_precision": True
   ```

2. **调整batch_size**：
   - 根据显存大小调整
   - 建议：6GB显存用batch_size=8-12

3. **使用数据增强**（小数据集）：
   ```python
   "use_data_augment": True
   ```

### 模型优化

1. **使用EMA**：
   ```python
   "use_ema": True,
   "ema_decay": 0.999
   ```

2. **调整GRU隐藏层大小**：
   - 默认320，可根据数据集调整
   - 小数据集可减小到256

3. **实体类型特定阈值**：
   - 为不同实体类型设置不同解码阈值
   - 提高精确率或召回率

## 常见问题

### 1. 显存不足

**解决方案**：
- 减小 `batch_size`
- 减小 `max_seq_len`
- 减小 `gru_hidden_size`
- 禁用数据增强

### 2. 训练不收敛

**解决方案**：
- 降低学习率
- 增加warmup_ratio
- 检查数据格式是否正确
- 检查实体位置是否准确

### 3. F1分数低

**解决方案**：
- 增加训练轮数
- 使用数据增强
- 调整解码阈值
- 使用模型集成

### 4. 预测速度慢

**解决方案**：
- 使用ONNX导出模型
- 减小max_seq_len
- 使用批量预测

## 贡献指南

### 如何贡献

1. Fork本项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交变更 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建Pull Request

### 代码审查

- 所有PR需要至少一人审查
- 确保测试通过
- 确保代码符合规范
- 更新相关文档

## 发布流程

### 版本号规范

使用语义化版本号：`MAJOR.MINOR.PATCH`

- `MAJOR`: 不兼容的API变更
- `MINOR`: 向后兼容的功能新增
- `PATCH`: 向后兼容的bug修复

### 发布步骤

1. 更新版本号
2. 更新CHANGELOG.md
3. 创建Git标签
4. 推送到GitHub
5. 创建GitHub Release
6. 发布到PyPI（可选）

## 联系方式

- Issue: [GitHub Issues](https://github.com/your-username/GlobalPointer_pytorch/issues)
- Email: your-email@example.com
- 微信群：[加入讨论群]

---

最后更新：2025年
