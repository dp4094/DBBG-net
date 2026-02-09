"""
GlobalPointer NER 项目配置文件

本文件包含训练、评估和模型的所有配置参数。
支持通过环境变量覆盖部分配置。

环境变量：
    DATA_HOME: 数据集根目录路径
    MODEL_HOME: 预训练模型目录路径
    OUTPUT_DIR: 训练输出目录路径
    RESULTS_DIR: 评估结果目录路径
"""

import os
import time
from typing import Dict, Any, Optional


# ============================================================================
# 基础配置
# ============================================================================

common = {
    # 实验名称，对应数据集名称（cluener, weibo, msra, peoplesdaily）
    "exp_name": "weibo",
    
    # 编码器类型（BERT, DeBERTa）
    "encoder": "DeBERTa",
    
    # 数据集根目录，可通过环境变量DATA_HOME覆盖
    "data_home": os.getenv("DATA_HOME", "./datasets"),
    
    # 预训练模型路径，可通过环境变量MODEL_HOME覆盖
    # 支持的模型：bert-base-chinese, Erlangshen-DeBERTa-v2-320M-Chinese
    "bert_path": os.getenv(
        "MODEL_HOME",
        "./pretrained_models/Erlangshen-DeBERTa-v2-320M-Chinese"
    ),
    
    # 运行类型（train: 训练, eval: 评估）
    "run_type": "train",
    
    # 保存模型的F1分数阈值，只有超过此阈值的模型才会被保存
    "f1_2_save": 0.72,
    
    # 日志记录器类型（wandb: 使用Weights & Biases, default: 控制台输出）
    "logger": "default"
}


# ============================================================================
# Weights & Biases 配置
# ============================================================================
# 只有在 logger="wandb" 时生效，用于可视化训练过程

wandb_config = {
    # 运行名称，默认使用当前时间戳
    "run_name": time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime()),
    
    # 日志记录间隔（每N个batch记录一次）
    "log_interval": 10
}


# ============================================================================
# 训练配置
# ============================================================================

train_config = {
    # -------------------- 数据文件配置 --------------------
    # 训练数据文件名（相对于 data_home/exp_name/ 目录）
    "train_data": "train.json",
    
    # 验证数据文件名
    "valid_data": "dev.json",
    
    # 测试数据文件名
    "test_data": "test.json",
    
    # 实体类型映射文件名
    "ent2id": "ent2id.json",
    
    # -------------------- 输出配置 --------------------
    # 模型保存路径，可通过环境变量OUTPUT_DIR覆盖
    "path_to_save_model": os.getenv("OUTPUT_DIR", "./outputs"),
    
    # -------------------- 超参数配置 --------------------
    "hyper_parameters": {
        # ===== 优化器参数 =====
        # 学习率，DeBERTa建议使用较小的学习率（5e-6），BERT可使用2e-5
        "lr": 5e-6,
        
        # 权重衰减系数，用于L2正则化
        "weight_decay": 0.01,
        
        # ===== 训练参数 =====
        # 批次大小，根据显存大小调整（6GB显存建议8-12）
        "batch_size": 12,
        
        # 训练轮数
        "epochs": 20,
        
        # 随机种子，用于结果复现
        "seed": 2333,
        
        # 最大序列长度，超过此长度的文本会被截断
        "max_seq_len": 256,
        
        # ===== 学习率调度器 =====
        # 调度器类型（CAWR: 余弦退火重启, StepLR: 步进衰减）
        "scheduler": "CAWR",
        
        # 预热比例，前N%的训练步数用于学习率预热
        "warmup_ratio": 0.15,
        
        # ===== 模型结构参数 =====
        # GRU隐藏层大小
        "gru_hidden_size": 320,
        
        # Dropout概率，用于防止过拟合
        "dropout": 0.2,
        
        # ===== 数据增强配置 =====
        # 是否使用数据增强（小数据集建议开启）
        "use_data_augment": True,
        
        # 数据增强概率，每个样本有N%的概率被增强
        "augment_prob": 0.3,
        
        # 数据增强方法列表（synonym: 同义词替换, replace: 随机替换, swap: 位置交换）
        "augment_methods": ["synonym", "replace", "swap"],
        
        # ===== 正则化配置 =====
        # 是否使用LayerNorm
        "use_layernorm": True,
        
        # 是否使用梯度裁剪
        "use_gradient_clip": True,
        
        # 梯度裁剪阈值
        "gradient_clip_value": 1.0,
        
        # 是否使用标签平滑
        "use_label_smoothing": True,
        
        # 标签平滑系数（0.0-1.0），较小的值可减少过拟合
        "label_smoothing_factor": 0.03,
        
        # ===== EMA配置 =====
        # 是否使用指数移动平均（EMA），可提高模型稳定性
        "use_ema": True,
        
        # EMA衰减率，越接近1.0，历史权重影响越大
        "ema_decay": 0.999,
        
        # EMA更新间隔（每N个batch更新一次）
        "ema_update_interval": 5,
        
        # ===== 混合精度训练 =====
        # 是否使用混合精度训练（FP16），可加速训练并减少显存占用
        "use_mixed_precision": True,
        
        # ===== 实体类型特定配置 =====
        # 实体类型特定的标签平滑因子，将在运行时根据数据集动态设置
        "entity_specific_smoothing": None
    }
}


# ============================================================================
# 评估配置
# ============================================================================

eval_config = {
    # -------------------- 模型配置 --------------------
    # 模型检查点目录路径（需要根据实际训练输出修改）
    "model_state_dir": "./outputs/weibo",  # 默认使用数据集名称的输出目录
    
    # WandB运行ID（如果使用wandb）
    "run_id": "",
    
    # 使用倒数第N个模型检查点（1表示最后一个）
    "last_k_model": 1,
    
    # -------------------- 数据配置 --------------------
    # 预测数据文件名
    "predict_data": "test.json",
    
    # 实体类型映射文件名
    "ent2id": "ent2id.json",
    
    # -------------------- 输出配置 --------------------
    # 结果保存目录，可通过环境变量RESULTS_DIR覆盖
    "save_res_dir": os.getenv("RESULTS_DIR", "./results"),
    
    # -------------------- 超参数配置 --------------------
    "hyper_parameters": {
        # 批次大小
        "batch_size": 12,
        
        # 最大序列长度
        "max_seq_len": 256,
        
        # GRU隐藏层大小（需与训练时一致）
        "gru_hidden_size": 320,
        
        # Dropout概率（评估时通常设为较小值）
        "dropout": 0.1
    },
    
    # -------------------- 模型集成配置 --------------------
    # 是否使用模型集成（多个模型投票，可提高性能）
    "use_ensemble": False,
    
    # 集成模型数量
    "ensemble_models": 5,
    
    # -------------------- 解码配置 --------------------
    # 默认解码阈值，低于此阈值的实体会被过滤
    "decode_threshold": 0.02,
    
    # 是否启用阈值搜索（在验证集上搜索最优阈值）
    "threshold_search": True,
    
    # 是否使用实体后处理规则
    "use_post_processing": True,
    
    # ===== 实体类型特定配置 =====
    # 实体类型特定的解码阈值，将在运行时根据数据集动态设置
    "entity_thresholds": None
}


# ============================================================================
# 学习率调度器配置
# ============================================================================

# CosineAnnealingWarmRestarts 调度器参数
cawr_scheduler = {
    # 周期倍增因子，每次重启后周期变为原来的T_mult倍
    "T_mult": 2,
    
    # 重启周期（epoch数），每N个epoch重启一次
    "rewarm_epoch_num": 2,
}

# StepLR 调度器参数
step_scheduler = {
    # 学习率衰减率，每次衰减后学习率变为原来的decay_rate倍
    "decay_rate": 0.95,
    
    # 衰减步数，每N个step衰减一次
    "decay_steps": 100,
}


# ============================================================================
# 数据集特定配置
# ============================================================================

def get_dataset_specific_configs(dataset_name: str) -> Dict[str, Any]:
    """
    根据数据集名称返回特定的实体参数配置
    
    不同数据集的实体类型分布和难度不同，需要针对性地调整参数。
    
    Args:
        dataset_name: 数据集名称 ('cluener', 'msra', 'peoplesdaily', 'weibo')
    
    Returns:
        包含entity_specific_smoothing和entity_thresholds的配置字典
    """
    configs = {
        'cluener': {
            # 实体类型特定的标签平滑系数
            'entity_specific_smoothing': {
                "address": 0.01,      # 地址：使用更小的平滑系数，提高精确率
                "scene": 0.01,        # 景点：使用更小的平滑系数
                "organization": 0.02, # 组织机构
                "company": 0.02,      # 公司
                "book": 0.03,         # 书籍
                "game": 0.03,         # 游戏
                "government": 0.03,   # 政府
                "movie": 0.03,        # 电影
                "name": 0.03,         # 人名
                "position": 0.03,     # 职位
                "default": 0.03       # 默认值
            },
            # 实体类型特定的解码阈值
            'entity_thresholds': {
                "address": 0.01,      # 地址：使用较低阈值，提高召回率
                "scene": 0.01,        # 景点：使用较低阈值
                "organization": 0.03, # 组织机构：使用较高阈值，提高精确率
                "company": 0.03,      # 公司：使用较高阈值
                "name": 0.04,         # 人名：使用更高阈值，减少错误识别
                "book": 0.02,         # 书籍
                "game": 0.02,         # 游戏
                "government": 0.02,   # 政府
                "movie": 0.02,        # 电影
                "position": 0.02,     # 职位
                "default": 0.02       # 默认阈值
            }
        },
        'msra': {
            'entity_specific_smoothing': {
                "address": 0.015,     # 地址实体，适中的平滑系数
                "name": 0.02,         # 人名实体，稍高的平滑系数
                "organization": 0.025,# 组织机构，较高的平滑系数
                "default": 0.03       # 默认值
            },
            'entity_thresholds': {
                "address": 0.015,     # 地址使用适中阈值
                "name": 0.035,        # 人名使用较高阈值，提高精确率
                "organization": 0.025,# 组织机构使用适中阈值
                "default": 0.02       # 默认阈值
            }
        },
        'peoplesdaily': {
            'entity_specific_smoothing': {
                "name": 0.02,         # 人名
                "location": 0.015,    # 地点
                "organization": 0.025,# 组织
                "default": 0.03
            },
            'entity_thresholds': {
                "name": 0.03,
                "location": 0.02,
                "organization": 0.025,
                "default": 0.02
            }
        },
        'weibo': {
            'entity_specific_smoothing': {
                "address": 0.01,      # 降低标签平滑
                "gpe.nom": 0.01,      # 地缘政治实体
                "name": 0.015,        # 降低name的平滑系数
                "organization": 0.02, # 降低organization平滑系数
                "default": 0.02       # 降低默认值
            },
            'entity_thresholds': {
                "address": 0.015,     # 降低阈值提高召回率
                "gpe.nom": 0.015,     # 地缘政治实体阈值
                "name": 0.02,         # 大幅降低name阈值
                "organization": 0.02, # 降低organization阈值
                "default": 0.015      # 降低默认阈值
            }
        }
    }
    
    # 如果数据集不在配置中，使用cluener的配置作为默认值
    return configs.get(dataset_name, configs['cluener'])


# ============================================================================
# 配置验证和合并
# ============================================================================

def validate_config() -> None:
    """
    验证配置参数的有效性
    
    Raises:
        ValueError: 如果配置参数无效
    """
    # 验证exp_name
    valid_datasets = ['cluener', 'weibo', 'msra', 'peoplesdaily']
    if common["exp_name"] not in valid_datasets:
        raise ValueError(
            f"Invalid exp_name: {common['exp_name']}. "
            f"Must be one of {valid_datasets}"
        )
    
    # 验证run_type
    if common["run_type"] not in ['train', 'eval']:
        raise ValueError(
            f"Invalid run_type: {common['run_type']}. "
            f"Must be 'train' or 'eval'"
        )
    
    # 验证encoder
    if common["encoder"] not in ['BERT', 'DeBERTa']:
        raise ValueError(
            f"Invalid encoder: {common['encoder']}. "
            f"Must be 'BERT' or 'DeBERTa'"
        )
    
    # 验证logger
    if common["logger"] not in ['wandb', 'default']:
        raise ValueError(
            f"Invalid logger: {common['logger']}. "
            f"Must be 'wandb' or 'default'"
        )
    
    # 验证路径存在性
    if not os.path.exists(common["data_home"]):
        print(f"Warning: data_home does not exist: {common['data_home']}")
    
    if not os.path.exists(common["bert_path"]):
        print(f"Warning: bert_path does not exist: {common['bert_path']}")


# 根据当前数据集配置更新参数
dataset_configs = get_dataset_specific_configs(common["exp_name"])
train_config["hyper_parameters"]["entity_specific_smoothing"] = \
    dataset_configs['entity_specific_smoothing']
eval_config["entity_thresholds"] = dataset_configs['entity_thresholds']

# 数据集特定的数据文件配置
if common["exp_name"] == "msra":
    # MSRA数据集没有dev.json，使用test.json作为验证集
    train_config["valid_data"] = "test.json"
    train_config["test_data"] = "test.json"

# 合并调度器配置到训练配置
train_config["hyper_parameters"].update(**cawr_scheduler, **step_scheduler)

# 合并所有配置
train_config = {**train_config, **common, **wandb_config}
eval_config = {**eval_config, **common}

# 验证配置
validate_config()


# ============================================================================
# 配置打印函数
# ============================================================================

def print_config(config_type: str = "train") -> None:
    """
    打印当前配置信息
    
    Args:
        config_type: 配置类型 ('train' 或 'eval')
    """
    config = train_config if config_type == "train" else eval_config
    
    print("\n" + "="*60)
    print(f"{config_type.upper()} Configuration")
    print("="*60)
    
    for key, value in config.items():
        if isinstance(value, dict):
            print(f"\n{key}:")
            for sub_key, sub_value in value.items():
                print(f"  {sub_key}: {sub_value}")
        else:
            print(f"{key}: {value}")
    
    print("="*60 + "\n")


# ============================================================================
# 使用示例
# ============================================================================

if __name__ == "__main__":
    # 打印训练配置
    print_config("train")
    
    # 打印评估配置
    print_config("eval")
