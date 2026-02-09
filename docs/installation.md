# 安装指南

本文档详细说明如何安装和配置 GlobalPointer 中文命名实体识别项目。

## 目录

- [系统要求](#系统要求)
- [环境配置](#环境配置)
- [依赖安装](#依赖安装)
- [预训练模型下载](#预训练模型下载)
- [数据集准备](#数据集准备)
- [验证安装](#验证安装)
- [常见问题](#常见问题)

## 系统要求

### 硬件要求

- **CPU**: 任意现代CPU
- **内存**: 至少 8GB RAM（推荐 16GB+）
- **GPU**: 
  - 训练：推荐 NVIDIA GPU（6GB+ 显存）
  - 推理：可选，CPU也可运行

### 软件要求

- **操作系统**: Windows / Linux / macOS
- **Python**: 3.6 或更高版本（推荐 3.8+）
- **CUDA**: 如果使用GPU，需要 CUDA 11.0+（推荐 11.8）
- **Git**: 用于克隆代码仓库

## 环境配置

### 方法1：使用 Conda（推荐）

Conda 可以更好地管理Python环境和依赖。

```bash
# 1. 创建conda环境
conda create -n globalpointer python=3.8

# 2. 激活环境
conda activate globalpointer

# 3. 安装PyTorch（根据CUDA版本选择）
# CUDA 11.8
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# CUDA 12.1
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# CPU版本
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

### 方法2：使用 venv

如果不使用Conda，可以使用Python自带的venv。

```bash
# 1. 创建虚拟环境
python -m venv venv

# 2. 激活环境
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate

# 3. 升级pip
python -m pip install --upgrade pip
```

## 依赖安装

### 基础安装（仅核心依赖）

```bash
# 克隆项目
git clone https://github.com/your-username/GlobalPointer_pytorch.git
cd GlobalPointer_pytorch

# 安装核心依赖
pip install -r requirements.txt
```

### 完整安装（包含所有可选依赖）

```bash
# 安装所有依赖（用于开发）
pip install -r requirements-dev.txt
```

### 自定义安装

如果只需要特定功能，可以手动安装：

```bash
# 核心依赖
pip install torch>=1.8.1 transformers>=4.1.1 numpy>=1.19.0 tqdm>=4.54.1 jieba>=0.42.1

# 训练可视化
pip install wandb>=0.10.26 matplotlib>=3.3.0

# ONNX导出
pip install onnx>=1.9.0 onnxruntime>=1.8.0

# Web服务
pip install flask>=2.0.0 fastapi>=0.68.0 uvicorn>=0.15.0

# 开发工具
pip install pytest>=6.2.0 black>=21.0 flake8>=3.9.0
```

## 预训练模型下载

项目支持以下预训练模型：

### BERT-base-Chinese

```bash
# 方法1：使用Hugging Face CLI
pip install huggingface_hub
huggingface-cli download bert-base-chinese --local-dir ./pretrained_models/bert-base-chinese

# 方法2：手动下载
# 访问 https://huggingface.co/bert-base-chinese
# 下载以下文件到 pretrained_models/bert-base-chinese/：
#   - config.json
#   - pytorch_model.bin
#   - vocab.txt
```

### Erlangshen-DeBERTa-v2-320M-Chinese

```bash
# 方法1：使用Hugging Face CLI
huggingface-cli download IDEA-CCNL/Erlangshen-DeBERTa-v2-320M-Chinese \
    --local-dir ./pretrained_models/Erlangshen-DeBERTa-v2-320M-Chinese

# 方法2：手动下载
# 访问 https://huggingface.co/IDEA-CCNL/Erlangshen-DeBERTa-v2-320M-Chinese
# 下载以下文件到 pretrained_models/Erlangshen-DeBERTa-v2-320M-Chinese/：
#   - config.json
#   - pytorch_model.bin
#   - vocab.txt
#   - tokenizer_config.json
#   - special_tokens_map.json
```

### 模型文件结构

下载完成后，目录结构应该如下：

```
pretrained_models/
├── bert-base-chinese/
│   ├── config.json
│   ├── pytorch_model.bin
│   └── vocab.txt
└── Erlangshen-DeBERTa-v2-320M-Chinese/
    ├── config.json
    ├── pytorch_model.bin
    ├── vocab.txt
    ├── tokenizer_config.json
    └── special_tokens_map.json
```

## 数据集准备

### CLUENER2020 数据集

CLUENER2020 是中文细粒度命名实体识别数据集。

```bash
# 数据集已包含在项目中
# 位置: datasets/cluener/
# 包含: train.json, dev.json, test.json, ent2id.json
```

### Weibo NER 数据集

微博命名实体识别数据集。

```bash
# 数据集已包含在项目中
# 位置: datasets/weibo/
# 包含: train.json, dev.json, test.json, ent2id.json
```

### 自定义数据集

如果要使用自定义数据集，请参考 [数据集说明文档](dataset.md)。

## 验证安装

运行测试脚本验证安装是否成功：

```bash
# 运行环境测试
python tests/test_environment.py
```

预期输出：

```
============================================================
🧪 GlobalPointer环境测试
============================================================

包导入测试:
  ✓ torch 导入成功
  ✓ transformers 导入成功
  ✓ numpy 导入成功
  ✓ tqdm 导入成功

配置文件测试:
  ✓ 配置文件加载成功
  ✓ 数据集配置: weibo

数据加载测试:
  ✓ 实体映射加载成功
  ✓ 训练数据加载成功

模型创建测试:
  ✓ 模型创建成功
  ✓ 模型前向传播正常

============================================================
🎉 所有测试通过！环境配置正确，可以开始训练。
============================================================
```

## 常见问题

### 1. PyTorch安装失败

**问题**: `pip install torch` 下载速度慢或失败

**解决方案**:
```bash
# 使用清华镜像源
pip install torch -i https://pypi.tuna.tsinghua.edu.cn/simple

# 或使用conda安装
conda install pytorch -c pytorch
```

### 2. CUDA版本不匹配

**问题**: `RuntimeError: CUDA error: no kernel image is available`

**解决方案**:
```bash
# 检查CUDA版本
nvidia-smi

# 安装对应版本的PyTorch
# 访问 https://pytorch.org/ 选择合适的版本
```

### 3. 显存不足

**问题**: `RuntimeError: CUDA out of memory`

**解决方案**:
- 减小 `batch_size`（在 config.py 中）
- 减小 `max_seq_len`
- 使用梯度累积
- 使用混合精度训练（已默认开启）

### 4. Transformers下载模型失败

**问题**: 无法从Hugging Face下载模型

**解决方案**:
```bash
# 方法1：使用镜像站
export HF_ENDPOINT=https://hf-mirror.com
pip install transformers

# 方法2：手动下载后放到指定目录
# 参考"预训练模型下载"部分
```

### 5. jieba分词安装失败

**问题**: Windows上安装jieba失败

**解决方案**:
```bash
# 使用预编译的wheel文件
pip install jieba --no-build-isolation

# 或使用conda安装
conda install -c conda-forge jieba
```

### 6. 权限错误

**问题**: `PermissionError: [Errno 13] Permission denied`

**解决方案**:
```bash
# Linux/macOS: 使用sudo（不推荐）或虚拟环境
# Windows: 以管理员身份运行命令提示符

# 推荐：使用虚拟环境避免权限问题
```

## 下一步

安装完成后，您可以：

1. 查看 [使用说明](usage.md) 了解如何训练和评估模型
2. 查看 [API文档](api.md) 了解代码结构
3. 查看 [性能报告](performance.md) 了解模型性能

## 获取帮助

如果遇到其他问题：

1. 查看 [常见问题](../README.md#常见问题)
2. 提交 [Issue](https://github.com/your-username/GlobalPointer_pytorch/issues)
3. 参考 [项目文档](../README.md)
