
import time

common = {
    "exp_name": "weibo",  # 设置为cluener以匹配数据集
    "encoder": "DeBERTa",
    "data_home": "./datasets",  # 改回原始路径
    "bert_path": "./pretrained_models/Erlangshen-DeBERTa-v2-320M-Chinese",  # bert-base-chinese or other plm from https://huggingface.co/models
    "run_type": "eval",  # train, eval
    "f1_2_save": 0.72,  # 降低保存阈值，以便保存更多模型用于集成
    "logger": "default"  # wandb or default，default意味着只输出日志到控制台
}


# wandb的配置，只有在logger=wandb时生效。用于可视化训练过程
wandb_config = {
    "run_name": time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime()),
    "log_interval": 10
}

train_config = {
    "train_data": "train.json",
    "valid_data": "dev.json",  # 使用dev.json作为验证集
    "test_data": "test.json",   # 使用相同的test.json作为测试集
    "ent2id": "ent2id.json",
    "path_to_save_model": "./outputs",  # 在logger不是wandb时生效
    "hyper_parameters": {
        "lr": 5e-6,  # 降低初始学习率，更稳定地收敛
        "weight_decay": 0.01,  # 适度减小权重衰减，避免过度正则化
        "batch_size": 12,  # 进一步减小批量大小以适应更长序列
        "epochs": 20,  # 增加训练轮次，确保充分训练
        "seed": 2333,
        "max_seq_len": 256,  # 调整序列长度以适应数据
        "scheduler": "CAWR",  # 使用余弦退火重启调度器
        "warmup_ratio": 0.15,  # 增加预热比例，帮助模型稳定训练
        "gru_hidden_size": 320,  # 恢复为320以匹配保存的模型参数
        "dropout": 0.2,  # 略微增加dropout，提高泛化能力
        

        # 数据增强配置
        "use_data_augment": True,  # 禁用数据增强以减少内存使用
        "augment_prob": 0.3,  # 增加数据增强概率
        "augment_methods": ["synonym", "replace", "swap"],  # 增加一种增强方法
        
        # 额外正则化配置
        "use_layernorm": True,  # 保持LayerNorm
        "use_gradient_clip": True,  # 保持梯度裁剪
        "gradient_clip_value": 1.0,  # 保持梯度裁剪值
        "use_label_smoothing": True,  # 使用标签平滑
        "label_smoothing_factor": 0.03,  # 降低标签平滑系数，减少过度平滑
        
        # EMA配置
        "use_ema": True,  # 使用EMA
        "ema_decay": 0.999,  # 保持EMA衰减率
        "ema_update_interval": 5,  # 减少EMA更新间隔，使EMA更新更频繁
        
        # 混合精度训练配置
        "use_mixed_precision": True,  # 使用混合精度训练
        
        # 实体类型特定的标签平滑因子 - 根据数据集动态配置
        "entity_specific_smoothing": None  # 将在运行时根据数据集类型动态设置
    }
}

eval_config = {
    "model_state_dir": "./outputs/2025-09-29_14.29.34",  # 使用有模型文件的目录
    "run_id": "",
    "last_k_model": 1,  # 取倒数第几个model_state
    "predict_data": "test.json",
    "ent2id": "ent2id.json",
    "save_res_dir": "./results",
    "hyper_parameters": {
        "batch_size": 12,
        "max_seq_len": 256,
        "gru_hidden_size": 320,
        "dropout": 0.1
    },
    "use_ensemble": False,  # 使用模型集成（测试集评估时设为False）
    "ensemble_models": 5,  # 保持集成模型数量
    "decode_threshold": 0.02,  # 提高默认解码阈值，减少噪声
    "threshold_search": True,  # 启用阈值搜索
    "use_post_processing": True,  # 使用实体后处理规则
    
    # 实体类型特定的解码阈值 - 根据数据集动态配置
    "entity_thresholds": None  # 将在运行时根据数据集类型动态设置
}

cawr_scheduler = {
    # CosineAnnealingWarmRestarts
    "T_mult": 2,  # 修改为整数值，满足调度器要求
    "rewarm_epoch_num": 2,  # 减小rewarm_epoch_num，更频繁地重启
}
step_scheduler = {
    # StepLR
    "decay_rate": 0.95,  # 增大衰减率，使学习率下降更明显
    "decay_steps": 100,  # 减小衰减步数，使学习率更频繁地下降
}

# ---------------------------------------------
# 数据集特定的实体参数配置
def get_dataset_specific_configs(dataset_name):
    """
    根据数据集名称返回特定的实体参数配置
    
    Args:
        dataset_name (str): 数据集名称 ('cluener', 'msra', 'peoplesdaily', 'weibo')
    
    Returns:
        dict: 包含entity_specific_smoothing和entity_thresholds的配置字典
    """
    configs = {
        'cluener': {
            'entity_specific_smoothing': {
                "address": 0.01,  # 地址使用更小的平滑系数，提高精确率
                "scene": 0.01,    # 景点使用更小的平滑系数
                "organization": 0.02,  # 组织机构
                "company": 0.02,  # 公司
                "book": 0.03,     # 书籍
                "game": 0.03,     # 游戏
                "government": 0.03,  # 政府
                "movie": 0.03,    # 电影
                "name": 0.03,     # 人名
                "position": 0.03, # 职位
                "default": 0.03   # 默认值
            },
            'entity_thresholds': {
                "address": 0.01,     # 地址使用较低阈值，提高召回率
                "scene": 0.01,       # 景点使用较低阈值
                "organization": 0.03,  # 组织机构使用较高阈值，提高精确率
                "company": 0.03,     # 公司使用较高阈值
                "name": 0.04,        # 人名使用更高阈值，减少错误识别
                "book": 0.02,        # 书籍
                "game": 0.02,        # 游戏
                "government": 0.02,  # 政府
                "movie": 0.02,       # 电影
                "position": 0.02,    # 职位
                "default": 0.02      # 默认阈值
            }
        },
        'msra': {
            'entity_specific_smoothing': {
                "address": 0.015,    # 地址实体，适中的平滑系数
                "name": 0.02,        # 人名实体，稍高的平滑系数
                "organization": 0.025, # 组织机构，较高的平滑系数
                "default": 0.03      # 默认值
            },
            'entity_thresholds': {
                "address": 0.015,    # 地址使用适中阈值
                "name": 0.035,       # 人名使用较高阈值，提高精确率
                "organization": 0.025, # 组织机构使用适中阈值
                "default": 0.02      # 默认阈值
            }
        },
        'peoplesdaily': {
            'entity_specific_smoothing': {
                "name": 0.02,        # 人名
                "location": 0.015,   # 地点
                "organization": 0.025, # 组织
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
                "address": 0.01,     # 降低标签平滑
                "gpe.nom": 0.01,     # 新增gpe.nom配置
                "name": 0.015,       # 降低name的平滑系数
                "organization": 0.02, # 降低organization平滑系数
                "default": 0.02      # 降低默认值
            },
            'entity_thresholds': {
                "address": 0.015,    # 降低阈值提高召回率
                "gpe.nom": 0.015,    # 新增gpe.nom阈值
                "name": 0.02,        # 大幅降低name阈值
                "organization": 0.02, # 降低organization阈值
                "default": 0.015     # 降低默认阈值
            }
        }
    }
    
    return configs.get(dataset_name, configs['cluener'])  # 默认使用cluener配置

# 根据当前数据集配置更新参数
dataset_configs = get_dataset_specific_configs(common["exp_name"])
train_config["hyper_parameters"]["entity_specific_smoothing"] = dataset_configs['entity_specific_smoothing']
eval_config["entity_thresholds"] = dataset_configs['entity_thresholds']

# 数据集特定的数据文件配置
if common["exp_name"] == "msra":
    train_config["valid_data"] = "test.json"  # MSRA数据集没有dev.json，使用test.json作为验证集
    train_config["test_data"] = "test.json"

train_config["hyper_parameters"].update(**cawr_scheduler, **step_scheduler)
train_config = {**train_config, **common, **wandb_config}
eval_config = {**eval_config, **common}
