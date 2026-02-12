"""
GlobalPointer NER 训练脚本
Date: 2021-05-31 19:50:58
LastEditors: GodK
"""

import os
import config as config_module
import sys
import torch
import json
import logging
from transformers import AutoTokenizer, DebertaV2Model
from common.utils import Preprocessor, multilabel_categorical_crossentropy, multilabel_categorical_crossentropy_with_smoothing
from models.GlobalPointer import DataMaker, MyDataset, GlobalPointer, MetricsCalculator
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import glob
# from evaluate import load_model  # 延迟导入，避免在模块加载时执行evaluate的初始化代码
import time
# 兼容新版本的PyTorch混合精度训练
from torch.amp import autocast, GradScaler
# 导入数据增强模块
from common.augment import NERDataAugmenter
# 导入EMA模块
from common.ema import EMA
import torch.nn as nn

config = config_module.train_config
eval_config = config_module.eval_config
hyper_parameters = config["hyper_parameters"]

os.environ["TOKENIZERS_PARALLELISM"] = "true"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
config["num_workers"] = 6 if sys.platform.startswith("linux") else 0

# for reproductivity
torch.manual_seed(hyper_parameters["seed"])  # pytorch random seed
torch.backends.cudnn.deterministic = True

# 创建metrics实例uk jbmnmmn
metrics = MetricsCalculator()

# 设置日志记录
def setup_logger():
    log_dir = os.path.join(config["path_to_save_model"], config["exp_name"])
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    timestamp = time.strftime("%Y-%m-%d_%H.%M.%S", time.gmtime())
    log_file = os.path.join(log_dir, f"training_log_{timestamp}.txt")
    
    # 配置logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger()

logger = setup_logger()
logger.info(f"开始训练 - 配置参数: {json.dumps(hyper_parameters, ensure_ascii=False, indent=2)}")

if config["logger"] == "wandb" and config["run_type"] == "train":
    # init wandb
    wandb.init(project="GlobalPointer_" + config["exp_name"],
               config=hyper_parameters  # Initialize config
               )
    wandb.run.name = config["run_name"] + "_" + wandb.run.id

    model_state_dict_dir = wandb.run.dir
    wandb_logger = wandb
elif config["run_type"] == "train":
    model_state_dict_dir = os.path.join(config["path_to_save_model"], config["exp_name"],
                                        time.strftime("%Y-%m-%d_%H.%M.%S", time.gmtime()))
    if not os.path.exists(model_state_dict_dir):
        os.makedirs(model_state_dict_dir)

tokenizer = AutoTokenizer.from_pretrained(config["bert_path"], do_lower_case=False)

def load_data(data_path, data_type="train"):
    """读取数据集

    Args:
        data_path (str): 数据存放路径
        data_type (str, optional): 数据类型. Defaults to "train".

    Returns:
        (json): train和valid中一条数据格式：{"text":"","entity_list":[(start, end, label), (start, end, label)...]}
    """
    if data_type == "train" or data_type == "valid":
        datas = []
        with open(data_path, encoding="utf-8") as f:
            for line in f:
                line = json.loads(line)
                item = {}
                item["text"] = line["text"]
                item["entity_list"] = []
                for k, v in line['label'].items():
                    for spans in v.values():
                        for start, end in spans:
                            item["entity_list"].append((start, end, k))
                datas.append(item)
        return datas
    else:
        return json.load(open(data_path, encoding="utf-8"))

def load_entity_dict(ent2id_path):
    """加载实体词典"""
    return load_data(ent2id_path, "ent2id")

def get_entity_specific_params(ent2id_dict):
    """获取实体特定的标签平滑参数"""
    entity_specific_smoothing = hyper_parameters.get("entity_specific_smoothing", {})
    default_smoothing = entity_specific_smoothing.get("default", hyper_parameters["label_smoothing_factor"])
    
    # 准备实体特定的权重
    entity_weights = {}
    for entity_type in ent2id_dict.keys():
        entity_weights[entity_type] = entity_specific_smoothing.get(entity_type, default_smoothing)
    
    logger.info(f"数据集: {config['exp_name'].upper()}")
    logger.info(f"实体特定标签平滑: {entity_weights}")
    return entity_weights

def get_entity_thresholds(ent2id_dict):
    """获取实体特定的解码阈值"""
    entity_thresholds = eval_config.get("entity_thresholds", {})
    default_threshold = entity_thresholds.get("default", eval_config["decode_threshold"])
    
    # 为每个实体类型准备解码阈值
    decode_thresh_dict = {}
    for entity_type in ent2id_dict.keys():
        decode_thresh_dict[entity_type] = entity_thresholds.get(entity_type, default_threshold)
    
    logger.info(f"实体特定解码阈值: {decode_thresh_dict}")
    return decode_thresh_dict

# 延迟加载ent2id，在main函数中处理
# if config_module.common["exp_name"]:
#     ent2id_path = os.path.join(config_module.common["data_home"], config_module.common["exp_name"], config["ent2id"])
# else:
#     ent2id_path = os.path.join(config_module.common["data_home"], config["ent2id"])

# ent2id = load_entity_dict(ent2id_path)
# ent_type_size = len(ent2id)

# # 获取实体特定参数
# entity_smoothing_weights = get_entity_specific_params(ent2id)
# entity_thresh_dict = get_entity_thresholds(ent2id)

def data_generator(data_type="train"):
    """
    读取数据，生成DataLoader。
    """

    if data_type == "train":
        if config_module.common["exp_name"]:
            train_data_path = os.path.join(config_module.common["data_home"], config_module.common["exp_name"], config["train_data"])
            valid_data_path = os.path.join(config_module.common["data_home"], config_module.common["exp_name"], config["valid_data"])
        else:
            train_data_path = os.path.join(config_module.common["data_home"], config["train_data"])
            valid_data_path = os.path.join(config_module.common["data_home"], config["valid_data"])
        
        train_data = load_data(train_data_path, "train")
        valid_data = load_data(valid_data_path, "valid")
        
        # 如果启用数据增强，对训练数据进行增强
        if hyper_parameters.get("use_data_augment", False) and data_type == "train":
            logger.info(f"使用数据增强 - 当前数据集: {config['exp_name'].upper()}...")
            # 获取数据增强配置
            augment_prob = hyper_parameters.get("augment_prob", 0.3)
            augment_methods = hyper_parameters.get("augment_methods", ["synonym", "insert", "swap", "replace"])
            
            # 限制数据增强的方法数量，提高效率
            if len(augment_methods) > 2:
                augment_methods = augment_methods[:2]
                logger.info(f"为提高效率，限制数据增强方法数量为2: {augment_methods}")
            
            augmenter = NERDataAugmenter(aug_prob=augment_prob, methods=augment_methods)
            logger.info(f"数据增强配置: 概率={augment_prob}, 方法={augment_methods}")
            
            # 对训练数据进行增强，但限制增强后的数据量
            augmented_train_data = []
            max_augmented_samples = min(len(train_data), 2000)  # 限制增强样本数量
            
            for i, sample in enumerate(train_data):
                # 原始样本
                augmented_train_data.append(sample)
                
                # 只对部分样本进行增强
                if i < max_augmented_samples:
                    # 增强样本
                    augmented_sample = augmenter.augment(sample)
                    if augmented_sample != sample:  # 只添加确实被增强的样本
                        augmented_train_data.append(augmented_sample)
            
            logger.info(f"数据增强前训练样本数: {len(train_data)}, 增强后: {len(augmented_train_data)}")
            train_data = augmented_train_data
    elif data_type == "valid":
        if config_module.common["exp_name"]:
            valid_data_path = os.path.join(config_module.common["data_home"], config_module.common["exp_name"], config["valid_data"])
        else:
            valid_data_path = os.path.join(config_module.common["data_home"], config["valid_data"])
        valid_data = load_data(valid_data_path, "valid")
        train_data = []
    elif data_type == "test":
        if config_module.common["exp_name"]:
            valid_data_path = os.path.join(config_module.common["data_home"], config_module.common["exp_name"], config["test_data"])
        else:
            valid_data_path = os.path.join(config_module.common["data_home"], config["test_data"])
        valid_data = load_data(valid_data_path, "valid")
        train_data = []

    all_data = train_data + valid_data

    # TODO:句子截取
    max_tok_num = 0
    for sample in all_data:
        tokens = tokenizer(sample["text"])["input_ids"]
        max_tok_num = max(max_tok_num, len(tokens))
    assert max_tok_num <= hyper_parameters[
        "max_seq_len"], f'数据文本最大token数量{max_tok_num}超过预设{hyper_parameters["max_seq_len"]}'
    max_seq_len = min(max_tok_num, hyper_parameters["max_seq_len"])

    data_maker = DataMaker(tokenizer)

    if data_type == "train":
        # train_inputs = data_maker.generate_inputs(train_data, max_seq_len, ent2id)
        # valid_inputs = data_maker.generate_inputs(valid_data, max_seq_len, ent2id)
        train_dataloader = DataLoader(MyDataset(train_data),
                                      batch_size=hyper_parameters["batch_size"],
                                      shuffle=True,
                                      num_workers=config["num_workers"],
                                      drop_last=False,
                                      collate_fn=lambda x: data_maker.generate_batch(x, max_seq_len, ent2id)
                                      )
        valid_dataloader = DataLoader(MyDataset(valid_data),
                                      batch_size=hyper_parameters["batch_size"],
                                      shuffle=True,
                                      num_workers=config["num_workers"],
                                      drop_last=False,
                                      collate_fn=lambda x: data_maker.generate_batch(x, max_seq_len, ent2id)
                                      )
        # for batch in train_dataloader:
        #     print(batch[1].shape)
        #     print(hyper_parameters["batch_size"])
        #     break
        return train_dataloader, valid_dataloader
    else:
        # valid_inputs = data_maker.generate_inputs(valid_data, max_seq_len, ent2id)
        valid_dataloader = DataLoader(MyDataset(valid_data),
                                      batch_size=hyper_parameters["batch_size"],
                                      shuffle=True,
                                      num_workers=config["num_workers"],
                                      drop_last=False,
                                      collate_fn=lambda x: data_maker.generate_batch(x, max_seq_len, ent2id)
                                      )
        return valid_dataloader


def train_step(batch_train, model, optimizer, criterion, scaler):
    # batch_input_ids:(batch_size, seq_len)    batch_labels:(batch_size, ent_type_size, seq_len, seq_len)
    batch_samples, batch_input_ids, batch_attention_mask, batch_token_type_ids, batch_labels = batch_train
    batch_input_ids, batch_attention_mask, batch_token_type_ids, batch_labels = (
        batch_input_ids.to(device),
        batch_attention_mask.to(device),
        batch_token_type_ids.to(device),
        batch_labels.to(device)
    )
    
    # 使用混合精度训练
    if hyper_parameters.get("use_mixed_precision", False):
        with autocast('cuda' if torch.cuda.is_available() else 'cpu'):
            logits = model(batch_input_ids, batch_attention_mask, batch_token_type_ids)
            
            # 调试信息
            print(f"标签形状: {batch_labels.shape}")
            print(f"标签中1的数量: {batch_labels.sum().item()}")
            print(f"标签中非零元素的比例: {(batch_labels != 0).float().mean().item()}")
            print(f"模型输出形状: {logits.shape}")
            print(f"模型输出的最大值: {logits.max().item()}")
            print(f"模型输出的最小值: {logits.min().item()}")
            print(f"模型输出的均值: {logits.mean().item()}")
            print(f"模型输出的标准差: {logits.std().item()}")
            
            loss = criterion(batch_labels, logits)
            print(f"计算的损失值: {loss.item()}")
        
        # 梯度缩放和反向传播
        scaler.scale(loss).backward()
        
        # 梯度裁剪（必须在 unscale 之后）
        if hyper_parameters.get("use_gradient_clip", False):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                hyper_parameters.get("gradient_clip_value", 1.0)
            )
        
        # 更新参数
        scaler.step(optimizer)
        scaler.update()
    else:
        # 常规训练（不使用混合精度）
        logits = model(batch_input_ids, batch_attention_mask, batch_token_type_ids)
        loss = criterion(batch_labels, logits)
        
        loss.backward()
        
        # 梯度裁剪
        if hyper_parameters.get("use_gradient_clip", False):
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                hyper_parameters.get("gradient_clip_value", 1.0)
            )
        
        optimizer.step()
    
    optimizer.zero_grad()
    
    return loss.item(), logits


def train(model, dataloader, epoch, optimizer):
    model.train()

    # 创建 GradScaler - 使用新版本 API
    scaler = GradScaler('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 如果启用EMA，创建EMA对象
    if hyper_parameters.get("use_ema", False):
        ema = EMA(
            model, 
            decay=hyper_parameters.get("ema_decay", 0.999),
            update_interval=hyper_parameters.get("ema_update_interval", 10)
        )
        logger.info(f"使用EMA: decay={ema.decay}, update_interval={ema.update_interval}")

    # loss func
    def loss_fun(y_true, y_pred):
        """
        y_true:(batch_size, ent_type_size, seq_len, seq_len)
        y_pred:(batch_size, ent_type_size, seq_len, seq_len)
        """
        batch_size, ent_type_size = y_pred.shape[:2]
        y_true = y_true.reshape(batch_size * ent_type_size, -1)
        y_pred = y_pred.reshape(batch_size * ent_type_size, -1)
        
        # 使用标签平滑损失函数
        if hyper_parameters.get("use_label_smoothing", False):
            epsilon = hyper_parameters.get("label_smoothing_factor", 0.05)
            loss = multilabel_categorical_crossentropy_with_smoothing(
                y_true, 
                y_pred, 
                epsilon=epsilon
            )
        else:
            loss = multilabel_categorical_crossentropy(y_true, y_pred)
        return loss

    # scheduler
    if hyper_parameters["scheduler"] == "CAWR":
        T_mult = hyper_parameters["T_mult"]
        rewarm_epoch_num = hyper_parameters["rewarm_epoch_num"]
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                         len(train_dataloader) * rewarm_epoch_num,
                                                                         T_mult)
    elif hyper_parameters["scheduler"] == "Step":
        decay_rate = hyper_parameters["decay_rate"]
        decay_steps = hyper_parameters["decay_steps"]
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=decay_steps, gamma=decay_rate)
    else:
        scheduler = None

    pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    total_loss = 0.
    for batch_ind, batch_data in pbar:

        loss, logits = train_step(batch_data, model, optimizer, loss_fun, scaler)
        
        # 如果启用EMA，更新EMA参数
        if hyper_parameters.get("use_ema", False):
            ema.update()

        total_loss += loss

        avg_loss = total_loss / (batch_ind + 1)
        if scheduler is not None:
            scheduler.step()

        pbar.set_description(
            f'Project:{config["exp_name"]}, Epoch: {epoch + 1}/{hyper_parameters["epochs"]}, Step: {batch_ind + 1}/{len(dataloader)}')
        pbar.set_postfix({
            'batch_loss': f'{loss:.6f}',
            'avg_loss': f'{avg_loss:.6f}',
            'lr': optimizer.param_groups[0]["lr"]
        })

        if config["logger"] == "wandb" and batch_ind % config["log_interval"] == 0:
            wandb_logger.log({
                "epoch": epoch,
                "train_loss": avg_loss,
                "learning_rate": optimizer.param_groups[0]['lr'],
            })
    
    # 计算整个epoch的平均损失
    epoch_avg_loss = total_loss / len(dataloader)
    logger.info(f"训练损失 (Train Loss): {epoch_avg_loss:.6f}")
    
    # 返回EMA对象和平均训练损失，以便在验证时使用
    return ema if hyper_parameters.get("use_ema", False) else None, epoch_avg_loss


def valid_step(batch_valid, model):
    batch_samples, batch_input_ids, batch_attention_mask, batch_token_type_ids, batch_labels = batch_valid
    batch_input_ids, batch_attention_mask, batch_token_type_ids, batch_labels = (batch_input_ids.to(device),
                                                                                 batch_attention_mask.to(device),
                                                                                 batch_token_type_ids.to(device),
                                                                                 batch_labels.to(device)
                                                                                 )
    with torch.no_grad():
        logits = model(batch_input_ids, batch_attention_mask, batch_token_type_ids)
        # 计算验证损失
        valid_loss = multilabel_categorical_crossentropy(batch_labels, logits)
    
    sample_f1, sample_precision, sample_recall, sample_f0_5, sample_f2, type_metrics = metrics.get_evaluate_fpr(logits, batch_labels)

    return sample_f1, sample_precision, sample_recall, sample_f0_5, sample_f2, type_metrics, valid_loss.item()


def valid(model, dataloader, ema=None):
    model.eval()
    
    # 如果使用EMA，应用EMA参数
    if ema is not None:
        ema.apply_shadow()

    total_f1, total_precision, total_recall = 0., 0., 0.
    total_f0_5, total_f2 = 0., 0.
    total_valid_loss = 0.  # 添加验证损失累计变量
    
    # 用于累积每种实体类型的指标
    entity_types_metrics = {}
    
    for batch_data in tqdm(dataloader, desc="Validating"):
        f1, precision, recall, f0_5, f2, type_metrics, valid_loss = valid_step(batch_data, model)

        total_f1 += f1
        total_precision += precision
        total_recall += recall
        total_f0_5 += f0_5
        total_f2 += f2
        total_valid_loss += valid_loss  # 累计验证损失
        
        # 累积每种实体类型的指标
        for ent_type, metrics_dict in type_metrics.items():
            if ent_type not in entity_types_metrics:
                entity_types_metrics[ent_type] = {
                    "precision": 0., "recall": 0., "f1": 0., "f0.5": 0., "f2": 0., "count": 0, "support": 0
                }
            
            entity_types_metrics[ent_type]["precision"] += metrics_dict["precision"]
            entity_types_metrics[ent_type]["recall"] += metrics_dict["recall"]
            entity_types_metrics[ent_type]["f1"] += metrics_dict["f1"]
            entity_types_metrics[ent_type]["f0.5"] += metrics_dict["f0.5"]
            entity_types_metrics[ent_type]["f2"] += metrics_dict["f2"]
            entity_types_metrics[ent_type]["count"] += 1
            entity_types_metrics[ent_type]["support"] += metrics_dict["support"]

    # 计算整体平均指标
    avg_f1 = total_f1 / (len(dataloader))
    avg_precision = total_precision / (len(dataloader))
    avg_recall = total_recall / (len(dataloader))
    avg_f0_5 = total_f0_5 / (len(dataloader))
    avg_f2 = total_f2 / (len(dataloader))
    avg_valid_loss = total_valid_loss / (len(dataloader))  # 计算平均验证损失
    
    # 计算每种实体类型的平均指标
    for ent_type, metrics_dict in entity_types_metrics.items():
        if metrics_dict["count"] > 0:
            metrics_dict["precision"] /= metrics_dict["count"]
            metrics_dict["recall"] /= metrics_dict["count"]
            metrics_dict["f1"] /= metrics_dict["count"]
            metrics_dict["f0.5"] /= metrics_dict["count"]
            metrics_dict["f2"] /= metrics_dict["count"]
    
    # 加载实体ID到名称的映射
    id2ent = {id: ent for ent, id in ent2id.items()}
    
    logger.info("******************************************")
    logger.info(f'总体评估指标 - Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}, F1: {avg_f1:.4f}')
    logger.info(f'总体评估指标 - F0.5: {avg_f0_5:.4f} (偏向精确率), F2: {avg_f2:.4f} (偏向召回率)')
    logger.info(f'验证损失 (Valid Loss): {avg_valid_loss:.6f}')  # 添加验证损失打印
    logger.info("******************************************")
    logger.info("各实体类型评估指标:")
    
    # 定义实体类型的中文名称映射
    entity_type_names = {
        "organization": "组织机构",
        "name": "人名",
        "address": "地址",
        "company": "公司",
        "government": "政府",
        "book": "书籍",
        "game": "游戏",
        "movie": "电影",
        "position": "职位",
        "scene": "景点",
        "time": "时间"
    }
    
    # 打印每种实体类型的评估指标
    for type_id, metrics_dict in sorted(entity_types_metrics.items(), key=lambda x: id2ent.get(x[0], "")):
        ent_type = id2ent.get(type_id, f"未知类型-{type_id}")
        chinese_name = entity_type_names.get(ent_type, ent_type)
        
        logger.info(f'{ent_type} ({chinese_name}) - 样本数: {metrics_dict["support"]}')
        logger.info(f'  Precision: {metrics_dict["precision"]:.4f}, Recall: {metrics_dict["recall"]:.4f}, F1: {metrics_dict["f1"]:.4f}')
        logger.info(f'  F0.5: {metrics_dict["f0.5"]:.4f} (偏向精确率), F2: {metrics_dict["f2"]:.4f} (偏向召回率)')
    
    logger.info("******************************************")
    
    if config["logger"] == "wandb":
        # 基本指标记录
        log_dict = {
            "valid_precision": avg_precision, 
            "valid_recall": avg_recall, 
            "valid_f1": avg_f1,
            "valid_f0.5": avg_f0_5,
            "valid_f2": avg_f2,
            "valid_loss": avg_valid_loss  # 添加验证损失到wandb日志
        }
        
        # 添加每种实体类型的指标
        for type_id, metrics_dict in entity_types_metrics.items():
            ent_type = id2ent.get(type_id, f"unknown_{type_id}")
            log_dict[f"entity_{ent_type}_f1"] = metrics_dict["f1"]
            log_dict[f"entity_{ent_type}_precision"] = metrics_dict["precision"]
            log_dict[f"entity_{ent_type}_recall"] = metrics_dict["recall"]
            
        wandb_logger.log(log_dict)
    
    # 如果使用EMA，恢复原始参数
    if ema is not None:
        ema.restore()
        
    return avg_f1, avg_valid_loss  # 同时返回F1和验证损失


def decode_entities(logits, attention_mask, texts, id2ent=None):
    """
    从模型输出中解码实体
    """
    batch_size = logits.size(0)
    results = []
    
    for i in range(batch_size):
        pred_entities = []
        for ent_type_id, logit in enumerate(logits[i]):
            ent_type = id2ent[ent_type_id] if id2ent else ent_type_id
            # 获取该实体类型的阈值
            threshold = entity_thresh_dict.get(ent_type, eval_config["decode_threshold"])
            
            seq_len = attention_mask[i].sum().item()
            for j in range(seq_len):
                for k in range(j, seq_len):
                    if logit[j][k] > threshold:
                        pred_entities.append({
                            "start_idx": j,
                            "end_idx": k,
                            "entity": texts[i][j:k+1] if texts else None,
                            "type": ent_type,
                            "type_id": ent_type_id,
                            "score": logit[j][k].item()
                        })
        results.append(pred_entities)
    return results


if __name__ == '__main__':
    # 检查环境变量中的配置
    weibo_data_home = os.environ.get('WEIBO_DATA_HOME')
    if weibo_data_home:
        config_module.common["data_home"] = weibo_data_home
        config_module.train_config["data_home"] = weibo_data_home
        print(f"从环境变量获取data_home: {weibo_data_home}")
    
    # 重新构建ent2id路径
    print(f"Debug: config_module.common['data_home'] = {config_module.common['data_home']}")
    print(f"Debug: config_module.common['exp_name'] = '{config_module.common['exp_name']}'")
    print(f"Debug: config['ent2id'] = {config['ent2id']}")
    
    if config_module.common["exp_name"]:
        ent2id_path = os.path.join(config_module.common["data_home"], config_module.common["exp_name"], config["ent2id"])
    else:
        ent2id_path = os.path.join(config_module.common["data_home"], config["ent2id"])
    
    print(f"Debug: ent2id_path = {ent2id_path}")
    
    ent2id = load_entity_dict(ent2id_path)
    ent_type_size = len(ent2id)
    
    # 获取实体特定参数
    entity_smoothing_weights = get_entity_specific_params(ent2id)
    entity_thresh_dict = get_entity_thresholds(ent2id)
    
    if config["run_type"] == "train":
        train_dataloader, valid_dataloader = data_generator()

        # 使用配置中的GRU参数
        encoder = DebertaV2Model.from_pretrained(config["bert_path"], ignore_mismatched_sizes=True)
        gru_hidden_size = hyper_parameters.get("gru_hidden_size") 
        dropout = hyper_parameters.get("dropout", 0.1)
        use_layernorm = hyper_parameters.get("use_layernorm", True)
        model = GlobalPointer(encoder, ent_type_size, 64, 
                             gru_hidden_size=gru_hidden_size, 
                             dropout=dropout,
                             use_layernorm=use_layernorm)
        model = model.to(device)
        
        logger.info(f"模型参数: gru_hidden_size={gru_hidden_size}, dropout={dropout}, use_layernorm={use_layernorm}")
        logger.info(f"正则化配置: gradient_clip={hyper_parameters.get('use_gradient_clip', False)}, label_smoothing={hyper_parameters.get('use_label_smoothing', False)}")
        
        # 如果启用了数据增强，记录配置
        if hyper_parameters.get("use_data_augment", False):
            logger.info(f"数据增强配置: prob={hyper_parameters.get('augment_prob', 0.3)}, methods={hyper_parameters.get('augment_methods', ['synonym', 'insert'])}")
        
        # 如果启用了EMA，记录配置
        if hyper_parameters.get("use_ema", False):
            logger.info(f"EMA配置: decay={hyper_parameters.get('ema_decay', 0.999)}, update_interval={hyper_parameters.get('ema_update_interval', 10)}")

        # optimizer
        init_learning_rate = float(hyper_parameters["lr"])
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=init_learning_rate, 
            weight_decay=hyper_parameters.get("weight_decay", 0.01)
        )
        
        # 如果配置了warmup，则使用warmup
        warmup_ratio = hyper_parameters.get("warmup_ratio", 0)
        if warmup_ratio > 0:
            total_steps = len(train_dataloader) * hyper_parameters["epochs"]
            warmup_steps = int(total_steps * warmup_ratio)
            logger.info(f"使用学习率预热: warmup_steps={warmup_steps}, total_steps={total_steps}")
            from transformers import get_linear_schedule_with_warmup
            scheduler = get_linear_schedule_with_warmup(
                optimizer, 
                num_warmup_steps=warmup_steps, 
                num_training_steps=total_steps
            )

        max_f1 = 0.
        for epoch in range(hyper_parameters["epochs"]):
            logger.info(f"开始第 {epoch + 1}/{hyper_parameters['epochs']} 轮训练")
            
            # 训练阶段
            ema, train_loss = train(model, train_dataloader, epoch, optimizer)
            
            # 验证阶段
            valid_f1, valid_loss = valid(model, valid_dataloader, ema)
            
            # 打印训练和验证损失
            logger.info(f"Epoch {epoch + 1} - 训练损失: {train_loss:.6f}, 验证损失: {valid_loss:.6f}, 验证F1: {valid_f1:.4f}")
            
            if warmup_ratio > 0 and scheduler is not None:
                scheduler.step()
                
            if valid_f1 > max_f1:
                max_f1 = valid_f1
                logger.info(f"新的最佳F1分数: {max_f1:.4f}")
                if valid_f1 > config["f1_2_save"]:  # save the best model
                    model_state_num = len(glob.glob(model_state_dict_dir + "/model_state_dict_*.pt"))
                    model_save_path = os.path.join(model_state_dict_dir, f"model_state_dict_{model_state_num}.pt")
                    
                    # 如果使用EMA，保存EMA参数
                    if ema is not None:
                        ema.apply_shadow()
                        torch.save(model.state_dict(), model_save_path)
                        ema.restore()
                        logger.info(f"保存新的最佳模型(EMA参数): {model_save_path}, F1: {valid_f1}")
                    else:
                        torch.save(model.state_dict(), model_save_path)
                        logger.info(f"保存新的最佳模型: {model_save_path}, F1: {valid_f1}")
            logger.info(f"Best F1: {max_f1}")
            logger.info("******************************************")
            
            if config["logger"] == "wandb":
                wandb_logger.log({"Best_F1": max_f1})
        
        # 训练完成后，根据配置决定是否在测试集上评估
        if config.get("auto_test_after_train", True):
            logger.info("\n" + "="*60)
            logger.info("训练完成！开始在测试集上评估最佳模型...")
            logger.info("="*60)
            
            # 加载最佳模型
            saved_models = glob.glob(model_state_dict_dir + "/model_state_dict_*.pt")
            if saved_models:
                # 加载最后保存的模型（通常是最佳模型）
                best_model_path = sorted(saved_models)[-1]
                logger.info(f"加载最佳模型: {best_model_path}")
                model.load_state_dict(torch.load(best_model_path))
                
                # 在测试集上评估
                test_dataloader = data_generator(data_type="test")
                logger.info("在测试集上评估...")
                test_f1, test_loss = valid(model, test_dataloader, ema=None)
                
                logger.info("\n" + "="*60)
                logger.info(f"测试集最终结果 - F1: {test_f1:.4f}, Loss: {test_loss:.6f}")
                logger.info("="*60)
                
                if config["logger"] == "wandb":
                    wandb_logger.log({"Test_F1": test_f1, "Test_Loss": test_loss})
            else:
                logger.warning("未找到保存的模型，跳过测试集评估")
                logger.info(f"提示: 模型F1分数可能未达到保存阈值 f1_2_save={config['f1_2_save']}")
        else:
            logger.info("\n训练完成！（auto_test_after_train=False，跳过测试集评估）")
            
    elif config["run_type"] == "eval":
        # 此处的 eval 是为了评估测试集的 p r f1（如果测试集有标签的情况），无标签预测使用 evaluate.py
        from evaluate import load_model  # 延迟导入
        model = load_model()
        test_dataloader = data_generator(data_type="test")
        valid(model, test_dataloader)
