"""
实现模型参数的指数移动平均(Exponential Moving Average)
"""
import torch


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