# 数据集目录

本目录包含项目使用的数据集。

## 当前使用的数据集

### CLUENER2020
- 路径: `cluener/`
- 说明: 中文细粒度命名实体识别数据集
- 包含: train.json, dev.json, test.json, ent2id.json

### Weibo NER
- 路径: `weibo/`
- 说明: 微博命名实体识别数据集
- 包含: train.json, dev.json, test.json, ent2id.json

## 归档数据集

未使用的数据集已移动到 `archive/` 目录。

## 数据格式

所有数据集使用统一的JSON格式：

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

## 添加自定义数据集

1. 创建数据集目录
2. 准备train.json, dev.json, test.json文件
3. 创建ent2id.json实体映射文件
4. 更新config.py中的数据集配置
