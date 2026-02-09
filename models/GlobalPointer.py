"""

"""
import sys

sys.path.append("../")
from common.utils import Preprocessor
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.length = len(data)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.length


class DataMaker(object):
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.add_special_tokens = False
        self.preprocessor = Preprocessor(tokenizer, self.add_special_tokens)

    def generate_inputs(self, datas, max_seq_len, ent2id, data_type="train"):
        """生成喂入模型的数据

        Args:
            datas (list): json格式的数据[{'text':'','entity_list':[(start,end,ent_type),()]}]
            max_seq_len (int): 句子最大token数量
            ent2id (dict): ent到id的映射
            data_type (str, optional): data类型. Defaults to "train".

        Returns:
            list: [(sample, input_ids, attention_mask, token_type_ids, labels),(),()...]
        """

        ent_type_size = len(ent2id)  # 实体类别

        all_inputs = []
        for sample in datas:
            inputs = self.tokenizer(
                sample["text"],
                max_length=max_seq_len,
                truncation=True,
                padding='max_length'
            )

            labels = None
            if data_type != "predict":
                ent2token_spans = self.preprocessor.get_ent2token_spans(
                    sample["text"], sample["entity_list"]
                )
                labels = np.zeros((ent_type_size, max_seq_len, max_seq_len))
                for start, end, label in ent2token_spans:
                    labels[ent2id[label], start, end] = 1
            inputs["labels"] = labels

            input_ids = torch.tensor(inputs["input_ids"]).long()
            attention_mask = torch.tensor(inputs["attention_mask"]).long()
            token_type_ids = torch.tensor(inputs["token_type_ids"]).long()
            if labels is not None:
                labels = torch.tensor(inputs["labels"]).long()

            sample_input = (sample, input_ids, attention_mask, token_type_ids, labels)

            all_inputs.append(sample_input)
        return all_inputs

    def generate_batch(self, batch_data, max_seq_len, ent2id, data_type="train"):
        batch_data = self.generate_inputs(batch_data, max_seq_len, ent2id, data_type)
        sample_list = []
        input_ids_list = []
        attention_mask_list = []
        token_type_ids_list = []
        labels_list = []

        for sample in batch_data:
            sample_list.append(sample[0])
            input_ids_list.append(sample[1])
            attention_mask_list.append(sample[2])
            token_type_ids_list.append(sample[3])
            if data_type != "predict":
                labels_list.append(sample[4])

        batch_input_ids = torch.stack(input_ids_list, dim=0)
        batch_attention_mask = torch.stack(attention_mask_list, dim=0)
        batch_token_type_ids = torch.stack(token_type_ids_list, dim=0)
        batch_labels = torch.stack(labels_list, dim=0) if data_type != "predict" else None

        return sample_list, batch_input_ids, batch_attention_mask, batch_token_type_ids, batch_labels

    def decode_ent(self, pred_matrix):
        pass


class MetricsCalculator(object):
    def __init__(self):
        super().__init__()

    def get_sample_f1(self, y_pred, y_true):
        y_pred = torch.gt(y_pred, 0).float()
        return 2 * torch.sum(y_true * y_pred) / torch.sum(y_true + y_pred)

    def get_sample_precision(self, y_pred, y_true):
        y_pred = torch.gt(y_pred, 0).float()
        return torch.sum(y_pred[y_true == 1]) / (y_pred.sum() + 1)

    def get_evaluate_fpr(self, y_pred, y_true):
        y_pred = y_pred.cpu().numpy()
        y_true = y_true.cpu().numpy()
        pred = []
        true = []
        
        # 记录每个类别的预测和真实实体
        pred_by_type = {}
        true_by_type = {}
        
        for b, l, start, end in zip(*np.where(y_pred > 0)):
            pred.append((b, l, start, end))
            if l not in pred_by_type:
                pred_by_type[l] = []
            pred_by_type[l].append((b, l, start, end))
            
        for b, l, start, end in zip(*np.where(y_true > 0)):
            true.append((b, l, start, end))
            if l not in true_by_type:
                true_by_type[l] = []
            true_by_type[l].append((b, l, start, end))

        # 计算总体指标
        R = set(pred)
        T = set(true)
        X = len(R & T)
        Y = len(R)
        Z = len(T)
        precision = X / Y if Y > 0 else 0
        recall = X / Z if Z > 0 else 0
        
        # 计算F1、F0.5和F2
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
        f0_5 = (1 + 0.5**2) * precision * recall / ((0.5**2 * precision) + recall) if ((0.5**2 * precision) + recall) > 0 else 0
        f2 = (1 + 2**2) * precision * recall / ((2**2 * precision) + recall) if ((2**2 * precision) + recall) > 0 else 0
        
        # 计算每个类别的指标
        type_metrics = {}
        
        for l in set(list(pred_by_type.keys()) + list(true_by_type.keys())):
            R_l = set(pred_by_type.get(l, []))
            T_l = set(true_by_type.get(l, []))
            X_l = len(R_l & T_l)
            Y_l = len(R_l)
            Z_l = len(T_l)
            
            # 每个类别的精确率和召回率
            p_l = X_l / Y_l if Y_l > 0 else 0
            r_l = X_l / Z_l if Z_l > 0 else 0
            
            # 每个类别的F1、F0.5和F2
            f1_l = 2 * p_l * r_l / (p_l + r_l) if p_l + r_l > 0 else 0
            f0_5_l = (1 + 0.5**2) * p_l * r_l / ((0.5**2 * p_l) + r_l) if ((0.5**2 * p_l) + r_l) > 0 else 0
            f2_l = (1 + 2**2) * p_l * r_l / ((2**2 * p_l) + r_l) if ((2**2 * p_l) + r_l) > 0 else 0
            
            type_metrics[l] = {
                "precision": p_l,
                "recall": r_l,
                "f1": f1_l,
                "f0.5": f0_5_l,
                "f2": f2_l,
                "support": Z_l  # 支持度，即真实实体数量
            }
        
        return f1, precision, recall, f0_5, f2, type_metrics


class GlobalPointer(nn.Module):
    def __init__(self, encoder, ent_type_size, inner_dim, RoPE=True, gru_hidden_size=320, dropout=0.1,
                 use_layernorm=True):
        super().__init__()
        self.encoder = encoder
        self.ent_type_size = ent_type_size
        self.inner_dim = inner_dim
        self.hidden_size = encoder.config.hidden_size

        # 原有的BiGRU层
        self.bigru = nn.GRU(self.hidden_size,
                            gru_hidden_size,
                            num_layers=2,
                            bidirectional=True,
                            batch_first=True,
                            dropout=dropout if 2 > 1 else 0)

        # 添加BBFE模块
        # 1. 边界特征提取器 - 使用轻量级1D-CNN
        self.bbfe_size = 64  # 轻量级特征维度，小于gru_hidden_size
        self.head_extractor = nn.Conv1d(gru_hidden_size * 2, self.bbfe_size, kernel_size=3, padding=1)
        self.tail_extractor = nn.Conv1d(gru_hidden_size * 2, self.bbfe_size, kernel_size=3, padding=1)

        # 2. 特征投影层 - 将提取的特征投影回原始维度
        self.head_projector = nn.Linear(self.bbfe_size, gru_hidden_size * 2)
        self.tail_projector = nn.Linear(self.bbfe_size, gru_hidden_size * 2)

        # 3. 门控融合层
        self.head_gate = nn.Linear(gru_hidden_size * 2 + self.bbfe_size, gru_hidden_size * 2)
        self.tail_gate = nn.Linear(gru_hidden_size * 2 + self.bbfe_size, gru_hidden_size * 2)

        # 原有的dropout和layernorm
        self.dropout = nn.Dropout(dropout)
        self.use_layernorm = use_layernorm
        if use_layernorm:
            self.layernorm = nn.LayerNorm(gru_hidden_size * 2)

        # 原有的dense层
        self.dense = nn.Linear(gru_hidden_size * 2, self.ent_type_size * self.inner_dim * 2)
        self.RoPE = RoPE
    # def __init__(self, encoder, ent_type_size, inner_dim, RoPE=True, gru_hidden_size=320, dropout=0.1, use_layernorm=True):
    #     super().__init__()
    #     self.encoder = encoder
    #     self.ent_type_size = ent_type_size
    #     self.inner_dim = inner_dim
    #     self.hidden_size = encoder.config.hidden_size
    #
    #     # 降低dropout值为0.1，使用2层GRU
    #     self.bigru = nn.GRU(self.hidden_size,
    #                        gru_hidden_size,
    #                        num_layers=2,
    #                        bidirectional=True,
    #                        batch_first=True,
    #                        dropout=dropout if 2 > 1 else 0)  # 降低层间dropout
    #
    #     # 降低输出dropout
    #     self.dropout = nn.Dropout(dropout)
    #
    #     # 添加LayerNorm以提高泛化能力
    #     self.use_layernorm = use_layernorm
    #     if use_layernorm:
    #         self.layernorm = nn.LayerNorm(gru_hidden_size * 2)
    #
    #     # 修改dense层的输入维度为BiGRU的输出维度 (hidden_size -> gru_hidden_size*2)
    #     self.dense = nn.Linear(gru_hidden_size * 2, self.ent_type_size * self.inner_dim * 2)
    #
    #     self.RoPE = RoPE

    def sinusoidal_position_embedding(self, batch_size, seq_len, output_dim):
        position_ids = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(-1)

        indices = torch.arange(0, output_dim // 2, dtype=torch.float)
        indices = torch.pow(10000, -2 * indices / output_dim)
        embeddings = position_ids * indices
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = embeddings.repeat((batch_size, *([1] * len(embeddings.shape))))
        embeddings = torch.reshape(embeddings, (batch_size, seq_len, output_dim))
        embeddings = embeddings.to(self.device)
        return embeddings

    def forward(self, input_ids, attention_mask, token_type_ids):
        self.device = input_ids.device

        context_outputs = self.encoder(input_ids, attention_mask, token_type_ids)
        # last_hidden_state:(batch_size, seq_len, hidden_size)
        last_hidden_state = context_outputs[0]

        # 通过BiGRU层
        gru_output, _ = self.bigru(last_hidden_state)

        # 应用BBFE模块
        # 1. 转换维度以适应Conv1D (batch, seq_len, channels) -> (batch, channels, seq_len)
        gru_output_conv = gru_output.transpose(1, 2)

        # 2. 提取头尾边界特征
        head_features = self.head_extractor(gru_output_conv)  # (batch, bbfe_size, seq_len)
        tail_features = self.tail_extractor(gru_output_conv)  # (batch, bbfe_size, seq_len)

        # 3. 转回原始维度 (batch, bbfe_size, seq_len) -> (batch, seq_len, bbfe_size)
        head_features = head_features.transpose(1, 2)
        tail_features = tail_features.transpose(1, 2)

        # 4. 投影到原始维度
        head_features_proj = self.head_projector(head_features)  # (batch, seq_len, gru_hidden_size*2)
        tail_features_proj = self.tail_projector(tail_features)  # (batch, seq_len, gru_hidden_size*2)

        # 5. 计算门控信号
        head_gate_input = torch.cat([gru_output, head_features], dim=-1)
        tail_gate_input = torch.cat([gru_output, tail_features], dim=-1)

        head_gate_values = torch.sigmoid(self.head_gate(head_gate_input))
        tail_gate_values = torch.sigmoid(self.tail_gate(tail_gate_input))

        # 6. 门控融合
        enhanced_output = gru_output + head_gate_values * head_features_proj + tail_gate_values * tail_features_proj

        # 应用LayerNorm（如果启用）
        if self.use_layernorm:
            enhanced_output = self.layernorm(enhanced_output)

        # 应用dropout
        enhanced_output = self.dropout(enhanced_output)

        # 后续代码不变，使用enhanced_output替代原来的gru_output
        batch_size = enhanced_output.size()[0]
        seq_len = enhanced_output.size()[1]

        # outputs:(batch_size, seq_len, ent_type_size*inner_dim*2)
        outputs = self.dense(enhanced_output)
        outputs = torch.split(outputs, self.inner_dim * 2, dim=-1)
        # outputs:(batch_size, seq_len, ent_type_size, inner_dim*2)
        outputs = torch.stack(outputs, dim=-2)
        # qw,kw:(batch_size, seq_len, ent_type_size, inner_dim)
        qw, kw = outputs[..., :self.inner_dim], outputs[..., self.inner_dim:]

        if self.RoPE:
            # pos_emb:(batch_size, seq_len, inner_dim)
            pos_emb = self.sinusoidal_position_embedding(batch_size, seq_len, self.inner_dim)
            # cos_pos,sin_pos: (batch_size, seq_len, 1, inner_dim)
            cos_pos = pos_emb[..., None, 1::2].repeat_interleave(2, dim=-1)
            sin_pos = pos_emb[..., None, ::2].repeat_interleave(2, dim=-1)
            qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], -1)
            qw2 = qw2.reshape(qw.shape)
            qw = qw * cos_pos + qw2 * sin_pos
            kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], -1)
            kw2 = kw2.reshape(kw.shape)
            kw = kw * cos_pos + kw2 * sin_pos

        # logits:(batch_size, ent_type_size, seq_len, seq_len)
        logits = torch.einsum('bmhd,bnhd->bhmn', qw, kw)

        # padding mask
        pad_mask = attention_mask.unsqueeze(1).unsqueeze(1).expand(batch_size, self.ent_type_size, seq_len, seq_len)
        logits = logits * pad_mask - (1 - pad_mask) * 1e12

        # 排除下三角
        mask = torch.tril(torch.ones_like(logits), -1)
        logits = logits - mask * 1e12

        return logits / self.inner_dim ** 0.5

    # def forward(self, input_ids, attention_mask, token_type_ids):
    #     self.device = input_ids.device
    #
    #     context_outputs = self.encoder(input_ids, attention_mask, token_type_ids)
    #     # last_hidden_state:(batch_size, seq_len, hidden_size)
    #     last_hidden_state = context_outputs[0]
    #
    #     # 通过BiGRU层
    #     gru_output, _ = self.bigru(last_hidden_state)
    #
    #     # 应用LayerNorm（如果启用）
    #     if self.use_layernorm:
    #         gru_output = self.layernorm(gru_output)
    #
    #     # 应用dropout
    #     gru_output = self.dropout(gru_output)
    #     # gru_output: (batch_size, seq_len, gru_hidden_size*2)
    #
    #     batch_size = gru_output.size()[0]
    #     seq_len = gru_output.size()[1]
    #
    #     # outputs:(batch_size, seq_len, ent_type_size*inner_dim*2)
    #     outputs = self.dense(gru_output)
    #     outputs = torch.split(outputs, self.inner_dim * 2, dim=-1)
    #     # outputs:(batch_size, seq_len, ent_type_size, inner_dim*2)
    #     outputs = torch.stack(outputs, dim=-2)
    #     # qw,kw:(batch_size, seq_len, ent_type_size, inner_dim)
    #     qw, kw = outputs[..., :self.inner_dim], outputs[..., self.inner_dim:]
    #
    #     if self.RoPE:
    #         # pos_emb:(batch_size, seq_len, inner_dim)
    #         pos_emb = self.sinusoidal_position_embedding(batch_size, seq_len, self.inner_dim)
    #         # cos_pos,sin_pos: (batch_size, seq_len, 1, inner_dim)
    #         cos_pos = pos_emb[..., None, 1::2].repeat_interleave(2, dim=-1)
    #         sin_pos = pos_emb[..., None, ::2].repeat_interleave(2, dim=-1)
    #         qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], -1)
    #         qw2 = qw2.reshape(qw.shape)
    #         qw = qw * cos_pos + qw2 * sin_pos
    #         kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], -1)
    #         kw2 = kw2.reshape(kw.shape)
    #         kw = kw * cos_pos + kw2 * sin_pos
    #
    #     # logits:(batch_size, ent_type_size, seq_len, seq_len)
    #     logits = torch.einsum('bmhd,bnhd->bhmn', qw, kw)
    #
    #     # padding mask
    #     pad_mask = attention_mask.unsqueeze(1).unsqueeze(1).expand(batch_size, self.ent_type_size, seq_len, seq_len)
    #     # pad_mask_h = attention_mask.unsqueeze(1).unsqueeze(-1).expand(batch_size, self.ent_type_size, seq_len, seq_len)
    #     # pad_mask = pad_mask_v&pad_mask_h
    #     logits = logits * pad_mask - (1 - pad_mask) * 1e12
    #
    #     # 排除下三角
    #     mask = torch.tril(torch.ones_like(logits), -1)
    #     logits = logits - mask * 1e12
    #
    #     return logits / self.inner_dim ** 0.5
