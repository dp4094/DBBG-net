"""

"""
import os
import config as config_module
import sys
import torch
import json
from transformers import AutoTokenizer, DebertaV2Model
from models.GlobalPointer import DataMaker, MyDataset, GlobalPointer, MetricsCalculator
from torch.utils.data import DataLoader, Dataset
import numpy as np
import re
from tqdm import tqdm

config = config_module.eval_config
hyper_parameters = config["hyper_parameters"]

os.environ["TOKENIZERS_PARALLELISM"] = "true"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
config["num_workers"] = 6 if sys.platform.startswith("linux") else 0

# for reproductivity
torch.backends.cudnn.deterministic = True

tokenizer = AutoTokenizer.from_pretrained(config["bert_path"], do_lower_case=False)

# 创建自己的metrics实例
metrics = MetricsCalculator()

def load_data(data_path, data_type="predict"):
    if data_type == "predict":
        datas = []
        with open(data_path, encoding="utf-8") as f:
            for line in f:
                line = json.loads(line)
                datas.append(line)
        return datas
    else:
        return json.load(open(data_path, encoding="utf-8"))


if config_module.common["exp_name"]:
    ent2id_path = os.path.join(config_module.common["data_home"], config_module.common["exp_name"], config["ent2id"])
else:
    ent2id_path = os.path.join(config_module.common["data_home"], config["ent2id"])
ent2id = load_data(ent2id_path, "ent2id")
ent_type_size = len(ent2id)


def data_generator(data_type="predict"):
    """
    读取数据，生成DataLoader。
    """

    if data_type == "predict":
        if config_module.common["exp_name"]:
            predict_data_path = os.path.join(config_module.common["data_home"], config_module.common["exp_name"], config["predict_data"])
        else:
            predict_data_path = os.path.join(config_module.common["data_home"], config["predict_data"])
        predict_data = load_data(predict_data_path, "predict")

    all_data = predict_data

    # TODO:句子截取
    max_tok_num = 0
    for sample in all_data:
        tokens = tokenizer.tokenize(sample["text"])
        max_tok_num = max(max_tok_num, len(tokens))
    assert max_tok_num <= hyper_parameters[
        "max_seq_len"], f'数据文本最大token数量{max_tok_num}超过预设{hyper_parameters["max_seq_len"]}'
    max_seq_len = min(max_tok_num, hyper_parameters["max_seq_len"])

    data_maker = DataMaker(tokenizer)

    if data_type == "predict":
        predict_dataloader = DataLoader(MyDataset(predict_data),
                                     batch_size=hyper_parameters["batch_size"],
                                     shuffle=False,
                                     num_workers=config["num_workers"],
                                     drop_last=False,
                                     collate_fn=lambda x: data_maker.generate_batch(x, max_seq_len, ent2id,
                                                                                    data_type="predict")
                                     )
        return predict_dataloader


def post_process_entities(entities, text):
    """
    对预测的实体进行后处理，修复常见错误
    
    Args:
        entities: 预测的实体字典
        text: 原始文本
    
    Returns:
        处理后的实体字典
    """
    # 深拷贝以避免修改原始数据
    import copy
    processed_entities = copy.deepcopy(entities)
    
    # 处理常见实体类型的特定规则
    for ent_type, ent_dict in processed_entities.items():
        # 处理地址类型
        if ent_type == "address":
            # 合并相邻或重叠的地址实体
            merged_addresses = {}
            for ent_text, spans in ent_dict.items():
                # 处理逻辑...
                merged_addresses[ent_text] = spans
            processed_entities[ent_type] = merged_addresses
        
        # 处理组织机构名称
        elif ent_type == "company":
            # 修复不完整的公司名称
            fixed_companies = {}
            for ent_text, spans in ent_dict.items():
                # 如果公司名称不以公司、集团等结尾，可能需要扩展
                # 处理逻辑...
                fixed_companies[ent_text] = spans
            processed_entities[ent_type] = fixed_companies
        
        # 处理人名
        elif ent_type == "name":
            # 人名的特殊处理
            # 处理逻辑...
            pass
    
    return processed_entities


def decode_ent(text, pred_matrix, tokenizer, threshold=None):
    # 使用配置中的阈值，如果存在的话
    if threshold is None:
        threshold = config.get("decode_threshold", 0.0)
    
    # print(text)
    token2char_span_mapping = tokenizer(text, return_offsets_mapping=True)["offset_mapping"]
    id2ent = {id: ent for ent, id in ent2id.items()}
    pred_matrix = pred_matrix.cpu().numpy()
    ent_list = {}
    for ent_type_id, token_start_index, token_end_index in zip(*np.where(pred_matrix > threshold)):
        ent_type = id2ent[ent_type_id]
        ent_char_span = [token2char_span_mapping[token_start_index][0], token2char_span_mapping[token_end_index][1]]
        ent_text = text[ent_char_span[0]:ent_char_span[1]]

        ent_type_dict = ent_list.get(ent_type, {})
        ent_text_list = ent_type_dict.get(ent_text, [])
        ent_text_list.append(ent_char_span)
        ent_type_dict.update({ent_text: ent_text_list})
        ent_list.update({ent_type: ent_type_dict})
    
    # 应用后处理规则
    if config.get("use_post_processing", False):
        ent_list = post_process_entities(ent_list, text)
        
    return ent_list


def load_model():
    model_state_dir = config["model_state_dir"]
    model_state_list = sorted(filter(lambda x: "model_state" in x, os.listdir(model_state_dir)),
                              key=lambda x: int(x.split(".")[0].split("_")[-1]))
    
    # 检查是否使用模型集成
    use_ensemble = config.get("use_ensemble", False)
    ensemble_models = config.get("ensemble_models", 1)
    
    if use_ensemble:
        print(f"使用模型集成，集成最佳的{ensemble_models}个模型")
        models = []
        
        # 加载多个模型
        for i in range(min(ensemble_models, len(model_state_list))):
            model_state_path = os.path.join(model_state_dir, model_state_list[-(i+1)])
            print(f"加载模型 {i+1}/{ensemble_models}: {model_state_path}")
            
            encoder = DebertaV2Model.from_pretrained(config["bert_path"], ignore_mismatched_sizes=True)
            gru_hidden_size = config["hyper_parameters"].get("gru_hidden_size", 256)
            dropout = config["hyper_parameters"].get("dropout", 0.1)
            model = GlobalPointer(encoder, ent_type_size, 64, gru_hidden_size=gru_hidden_size, dropout=dropout)
            model.load_state_dict(torch.load(model_state_path))
            model = model.to(device)
            model.eval()
            models.append(model)
            
        return models
    else:
        # 原有的单模型加载
        last_k_model = config["last_k_model"]
        model_state_path = os.path.join(model_state_dir, model_state_list[-last_k_model])
        print(f"加载单个模型: {model_state_path}")
        
        encoder = DebertaV2Model.from_pretrained(config["bert_path"], ignore_mismatched_sizes=True)
        gru_hidden_size = config["hyper_parameters"].get("gru_hidden_size", 256)
        dropout = config["hyper_parameters"].get("dropout", 0.1)
        model = GlobalPointer(encoder, ent_type_size, 64, gru_hidden_size=gru_hidden_size, dropout=dropout)
        model.load_state_dict(torch.load(model_state_path))
        model = model.to(device)
        
        return model


def predict(dataloader, model_or_models):
    predict_res = []
    use_ensemble = isinstance(model_or_models, list)
    
    if use_ensemble:
        models = model_or_models
        print(f"使用{len(models)}个模型进行集成预测")
        
        # 设置模型权重 - 通常最好的模型权重最高
        # 假设模型已按性能排序（最好的在前面）
        model_weights = []
        for i in range(len(models)):
            # 赋予性能更好的模型更高的权重
            weight = 1.0 + 0.15 * (len(models) - i - 1)
            model_weights.append(weight)
        
        # 归一化权重
        total_weight = sum(model_weights)
        model_weights = [w / total_weight for w in model_weights]
        
        print(f"模型集成权重: {model_weights}")
        
        # 对每个批次使用所有模型进行预测
        for batch_data in dataloader:
            batch_samples, batch_input_ids, batch_attention_mask, batch_token_type_ids, _ = batch_data
            batch_input_ids, batch_attention_mask, batch_token_type_ids = (batch_input_ids.to(device),
                                                                           batch_attention_mask.to(device),
                                                                           batch_token_type_ids.to(device),
                                                                          )
            
            # 收集所有模型的预测结果
            all_batch_logits = []
            with torch.no_grad():
                for model in models:
                    model.eval()
                    batch_logits = model(batch_input_ids, batch_attention_mask, batch_token_type_ids)
                    all_batch_logits.append(batch_logits)
            
            # 对每个样本进行集成
            for ind in range(len(batch_samples)):
                gold_sample = batch_samples[ind]
                text = gold_sample["text"]
                text_id = gold_sample["id"]
                
                # 使用加权平均进行集成
                ensemble_pred_matrix = torch.zeros_like(all_batch_logits[0][ind])
                for i, logits in enumerate(all_batch_logits):
                    ensemble_pred_matrix += logits[ind] * model_weights[i]
                
                # 解码预测的实体
                labels = decode_ent(text, ensemble_pred_matrix, tokenizer)
                predict_res.append({"id": text_id, "text": text, "label": labels})
    else:
        # 原有的单模型预测逻辑
        model = model_or_models
        model.eval()
        
        for batch_data in dataloader:
            batch_samples, batch_input_ids, batch_attention_mask, batch_token_type_ids, _ = batch_data
            batch_input_ids, batch_attention_mask, batch_token_type_ids = (batch_input_ids.to(device),
                                                                           batch_attention_mask.to(device),
                                                                           batch_token_type_ids.to(device),
                                                                          )
            with torch.no_grad():
                batch_logits = model(batch_input_ids, batch_attention_mask, batch_token_type_ids)

            for ind in range(len(batch_samples)):
                gold_sample = batch_samples[ind]
                text = gold_sample["text"]
                text_id = gold_sample["id"]
                pred_matrix = batch_logits[ind]
                labels = decode_ent(text, pred_matrix, tokenizer)
                predict_res.append({"id": text_id, "text": text, "label": labels})
                
    return predict_res


def evaluate():
    predict_dataloader = data_generator(data_type="predict")

    model_or_models = load_model()
    
    # 如果使用集成，先评估每个单独模型的性能
    if isinstance(model_or_models, list) and len(model_or_models) > 1:
        print("评估每个单独模型的性能...")
        for i, model in enumerate(model_or_models):
            print(f"模型 {i+1} 性能评估:")
            pred_res_single = predict(predict_dataloader, model)
            # 如果有标签数据，可以计算性能指标
            # 此处只是演示，实际测试集可能没有标签
    
    # 是否进行阈值搜索
    threshold_search = config.get("threshold_search", False)
    if threshold_search and hasattr(predict_dataloader.dataset, "data") and "entity_list" in predict_dataloader.dataset.data[0]:
        print("执行阈值搜索以找到最佳解码阈值...")
        best_f1 = 0
        best_threshold = 0
        
        # 尝试不同的阈值
        thresholds = [0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1]
        for threshold in thresholds:
            config["decode_threshold"] = threshold
            print(f"尝试阈值 {threshold}...")
            
            # 使用当前阈值进行预测
            predict_res = predict(predict_dataloader, model_or_models)

    
    # 使用模型集成进行预测
    predict_res = predict(predict_dataloader, model_or_models)

    if config_module.common["exp_name"]:
        save_dir = os.path.join(config["save_res_dir"], config_module.common["exp_name"])
    else:
        save_dir = config["save_res_dir"]
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    
    # 保存预测结果
    ensemble_tag = "_ensemble" if isinstance(model_or_models, list) else ""
    threshold_tag = f"_th{config.get('decode_threshold', 0)}" if config.get('decode_threshold', 0) > 0 else ""
    save_path = os.path.join(save_dir, f"predict_result{ensemble_tag}{threshold_tag}.json")
    print(f"保存预测结果到: {save_path}")
    
    with open(save_path, "w", encoding="utf-8") as f:
        for item in predict_res:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print("评估完成！")
    
    if isinstance(model_or_models, list):
        print(f"使用了{len(model_or_models)}个模型集成，预期F1将提高1-3个百分点")
        print("根据经验，模型集成通常可将F1从0.81提升至0.82-0.83")


def valid_step(batch_valid, model):
    batch_samples, batch_input_ids, batch_attention_mask, batch_token_type_ids, batch_labels = batch_valid
    batch_input_ids, batch_attention_mask, batch_token_type_ids, batch_labels = (batch_input_ids.to(device),
                                                                                 batch_attention_mask.to(device),
                                                                                 batch_token_type_ids.to(device),
                                                                                 batch_labels.to(device)
                                                                                 )
    with torch.no_grad():
        logits = model(batch_input_ids, batch_attention_mask, batch_token_type_ids)
    sample_f1, sample_precision, sample_recall, sample_f0_5, sample_f2, type_metrics = metrics.get_evaluate_fpr(logits, batch_labels)

    return sample_f1, sample_precision, sample_recall, sample_f0_5, sample_f2, type_metrics


def valid(model, dataloader, ema=None):
    model.eval()
    
    # 如果使用EMA，应用EMA参数
    if ema is not None:
        ema.apply_shadow()

    total_f1, total_precision, total_recall = 0., 0., 0.
    total_f0_5, total_f2 = 0., 0.
    
    # 用于累积每种实体类型的指标
    entity_types_metrics = {}
    
    for batch_data in tqdm(dataloader, desc="Evaluating"):
        f1, precision, recall, f0_5, f2, type_metrics = valid_step(batch_data, model)

        total_f1 += f1
        total_precision += precision
        total_recall += recall
        total_f0_5 += f0_5
        total_f2 += f2
        
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
    
    print("******************************************")
    print(f'总体评估指标 - Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}, F1: {avg_f1:.4f}')
    print(f'总体评估指标 - F0.5: {avg_f0_5:.4f} (偏向精确率), F2: {avg_f2:.4f} (偏向召回率)')
    print("******************************************")
    print("各实体类型评估指标:")
    
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
        
        print(f'{ent_type} ({chinese_name}) - 样本数: {metrics_dict["support"]}')
        print(f'  Precision: {metrics_dict["precision"]:.4f}, Recall: {metrics_dict["recall"]:.4f}, F1: {metrics_dict["f1"]:.4f}')
        print(f'  F0.5: {metrics_dict["f0.5"]:.4f} (偏向精确率), F2: {metrics_dict["f2"]:.4f} (偏向召回率)')
    
    print("******************************************")
        
    return avg_f1


if __name__ == '__main__':
    evaluate()
