"""

"""
import torch


def multilabel_categorical_crossentropy(y_true, y_pred):
    """
    https://kexue.fm/archives/7359
    多标签分类的交叉熵损失函数
    """
    y_pred = (1 - 2 * y_true) * y_pred  # -1 -> pos classes, 1 -> neg classes
    y_pred_neg = y_pred - y_true * 1e4  # mask the pred outputs of pos classes
    y_pred_pos = (y_pred - (1 - y_true) * 1e4)  # mask the pred outputs of neg classes
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    return (neg_loss + pos_loss).mean()


def multilabel_categorical_crossentropy_with_smoothing(y_true, y_pred, epsilon=0.1):
    """
    带有标签平滑的多标签分类交叉熵损失函数
    
    Args:
        y_true: 真实标签
        y_pred: 预测分数
        epsilon: 平滑因子
    """
    # 标签平滑：正标签变为1-epsilon，负标签变为epsilon/2
    y_true_smooth = y_true * (1 - epsilon) + epsilon/2
    
    # 使用平滑后的标签计算损失
    y_pred = (1 - 2 * y_true_smooth) * y_pred
    y_pred_neg = y_pred - y_true_smooth * 1e4
    y_pred_pos = (y_pred - (1 - y_true_smooth) * 1e4)
    
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    
    return (neg_loss + pos_loss).mean()


class EMA:
    """
    实现模型参数的指数移动平均(Exponential Moving Average)
    
    Args:
        model: 需要应用EMA的模型
        decay: EMA衰减率，越大表示历史权重占比越大
        update_interval: 更新间隔，每隔多少步更新一次EMA
    """
    def __init__(self, model, decay=0.999, update_interval=1):
        self.model = model
        self.decay = decay
        self.update_interval = update_interval
        self.shadow = {}
        self.backup = {}
        self.steps = 0
        
        # 初始化EMA参数
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """更新EMA参数"""
        self.steps += 1
        # 每隔update_interval步更新一次
        if self.steps % self.update_interval == 0:
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    assert name in self.shadow
                    new_average = self.decay * self.shadow[name] + (1.0 - self.decay) * param.data
                    self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        """应用EMA参数到模型"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]
    
    def restore(self):
        """恢复原始模型参数"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class Preprocessor(object):
    def __init__(self, tokenizer, add_special_tokens=False):
        super(Preprocessor, self).__init__()
        self.tokenizer = tokenizer
        self.add_special_tokens = add_special_tokens

    def get_ent2token_spans(self, text, entity_list):
        """实体列表转为token_spans

        Args:
            text (str): 原始文本
            entity_list (list): [(start, end, ent_type),(start, end, ent_type)...]
        """
        ent2token_spans = []

        inputs = self.tokenizer(text, add_special_tokens=self.add_special_tokens, return_offsets_mapping=True)
        token2char_span_mapping = inputs["offset_mapping"]
        text2tokens = self.tokenizer.tokenize(text, add_special_tokens=self.add_special_tokens)

        for ent_span in entity_list:
            ent = text[ent_span[0]:ent_span[1] + 1]
            ent2token = self.tokenizer.tokenize(ent, add_special_tokens=False)

            # 检查 ent2token 是否为空，避免 IndexError
            if not ent2token:
                print(f"Warning: Entity '{ent}' cannot be tokenized, skipping...")
                continue

            # 寻找ent的token_span
            token_start_indexs = [i for i, v in enumerate(text2tokens) if v == ent2token[0]]
            token_end_indexs = [i for i, v in enumerate(text2tokens) if v == ent2token[-1]]

            token_start_index = list(filter(lambda x: token2char_span_mapping[x][0] == ent_span[0], token_start_indexs))
            token_end_index = list(filter(lambda x: token2char_span_mapping[x][-1] - 1 == ent_span[1],
                                          token_end_indexs))  # token2char_span_mapping[x][-1]-1 减1是因为原始的char_span是闭区间，而token2char_span是开区间

            if len(token_start_index) == 0 or len(token_end_index) == 0:
                # print(f'[{ent}] 无法对应到 [{text}] 的token_span，已丢弃')
                continue
            token_span = (token_start_index[0], token_end_index[0], ent_span[2])
            ent2token_spans.append(token_span)

        return ent2token_spans
