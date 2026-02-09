# 转换后的NER数据集配置文件
import os


# MSRA 数据集配置
msra_config = {
    "data_home": "./datasets/converted_ner/msra",
    "train_data": "train.json",
    "valid_data": "dev.json" if os.path.exists("./datasets/converted_ner/msra/dev.json") else "test.json",
    "test_data": "test.json",
    "ent2id": "ent2id.json",
    "hyper_parameters": {
        "lr": 5e-6,
        "batch_size": 12,
        "epochs": 20,
        "max_seq_len": 128,  # 根据数据集调整
        # 其他参数继承自默认配置
    }
}


# PeoplesDaily 数据集配置
peoplesdaily_config = {
    "data_home": "./datasets/converted_ner/peoplesdaily",
    "train_data": "train.json",
    "valid_data": "dev.json" if os.path.exists("./datasets/converted_ner/peoplesdaily/dev.json") else "test.json",
    "test_data": "test.json",
    "ent2id": "ent2id.json",
    "hyper_parameters": {
        "lr": 5e-6,
        "batch_size": 12,
        "epochs": 20,
        "max_seq_len": 128,  # 根据数据集调整
        # 其他参数继承自默认配置
    }
}


# Weibo 数据集配置
weibo_config = {
    "data_home": "./datasets/converted_ner/weibo",
    "train_data": "train.json",
    "valid_data": "dev.json" if os.path.exists("./datasets/converted_ner/weibo/dev.json") else "test.json",
    "test_data": "test.json",
    "ent2id": "ent2id.json",
    "hyper_parameters": {
        "lr": 5e-6,
        "batch_size": 12,
        "epochs": 20,
        "max_seq_len": 128,  # 根据数据集调整
        # 其他参数继承自默认配置
    }
}

