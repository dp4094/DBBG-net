"""
数据增强模块，用于NER任务的数据增强
"""
import random
import copy
import re
import jieba
import numpy as np
from functools import lru_cache

class NERDataAugmenter:
    """
    NER数据增强器，提供多种适合命名实体识别任务的数据增强方法
    """
    def __init__(self, aug_prob=0.3, methods=None):
        """
        初始化数据增强器
        
        Args:
            aug_prob: 应用增强的概率
            methods: 使用的增强方法列表，可选值为["synonym", "insert", "swap", "replace"]
        """
        self.aug_prob = aug_prob
        self.methods = methods or ["synonym", "insert", "swap", "replace"]
        # 同义词词典，根据CLUENER2020数据集的实体类型定制
        self.synonym_dict = {
            # 组织机构相关
            "公司": ["企业", "集团", "机构", "单位"],
            "集团": ["公司", "企业", "机构", "组织"],
            "机构": ["单位", "部门", "组织", "团体"],
            "学校": ["院校", "学院", "大学", "高校"],
            "大学": ["学院", "高校", "学府", "校园"],
            "医院": ["医疗中心", "诊所", "卫生院", "医疗机构"],
            
            # 地址相关
            "省": ["省份", "省区"],
            "市": ["城市", "都市", "市区"],
            "区": ["地区", "片区", "区域"],
            "县": ["县城", "县区"],
            "路": ["大道", "街道", "大街"],
            "街": ["路", "大街", "街道"],
            
            # 政府相关
            "部门": ["机构", "单位", "司", "处"],
            "委员会": ["委会", "工作组"],
            "政府": ["行政机关", "管理部门"],
            "局": ["办公室", "管理处", "中心"],
            
            # 职位相关
            "总裁": ["总经理", "董事长", "CEO"],
            "经理": ["主管", "负责人", "管理者"],
            "主任": ["负责人", "处长", "组长"],
            "教授": ["老师", "讲师", "导师"],
            "医生": ["医师", "大夫", "医护人员"],
            
            # 常见动词
            "研究": ["研发", "探索", "调研", "分析"],
            "发展": ["进步", "提高", "增长", "壮大"],
            "创新": ["革新", "创造", "改革", "更新"],
            "工作": ["职业", "事业", "劳动", "职责"],
            "学习": ["研读", "攻读", "学习", "修习"],
            "管理": ["治理", "经营", "掌管", "运营"],
            "建设": ["构建", "打造", "创建", "开发"],
            
            # 常见形容词
            "重要": ["关键", "核心", "主要", "关键性"],
            "优秀": ["卓越", "杰出", "出色", "优异"],
            "创新": ["先进", "前沿", "新型", "现代化"],
            "高效": ["高产", "高质", "优质", "高水平"],
            
            # 常见连接词（用于随机插入）
            "的": ["之", "所", "那个"],
            "和": ["与", "及", "同"],
            "在": ["于", "处于", "位于"],
            "对": ["针对", "面对", "朝向"],
        }
        
        # CLUENER2020数据集的常见实体样例
        self.entity_examples = {
            "address": ["北京市", "上海市", "广州市", "深圳市", "杭州市", "南京市", "成都市",
                      "武汉市", "西安市", "重庆市", "天津市", "苏州市", "郑州市", "长沙市",
                      "浙江省", "江苏省", "广东省", "四川省", "湖北省", "河南省", "福建省"],
            "book": ["三国演义", "红楼梦", "西游记", "水浒传", "活着", "围城", "平凡的世界",
                   "白鹿原", "人间失格", "百年孤独", "解忧杂货店", "三体", "追风筝的人"],
            "company": ["阿里巴巴", "腾讯", "百度", "京东", "华为", "小米", "字节跳动",
                      "美团", "网易", "苹果公司", "微软", "谷歌", "亚马逊", "Facebook"],
            "game": ["王者荣耀", "英雄联盟", "和平精英", "绝地求生", "原神", "我的世界",
                   "穿越火线", "魔兽世界", "地下城与勇士", "阴阳师", "第五人格"],
            "government": ["国务院", "外交部", "教育部", "财政部", "科技部", "工信部",
                         "公安部", "民政部", "司法部", "人力资源和社会保障部"],
            "movie": ["流浪地球", "战狼", "哪吒之魔童降世", "唐人街探案", "我和我的祖国",
                    "红海行动", "长津湖", "你好，李焕英", "中国机长", "我不是药神"],
            "name": ["张伟", "王芳", "李娜", "刘洋", "陈明", "杨丽", "赵勇", "周静",
                   "吴强", "孙燕", "朱峰", "徐敏", "胡杰", "郭静", "何勇"],
            "organization": ["中国科学院", "北京大学", "清华大学", "复旦大学", "浙江大学",
                           "上海交通大学", "南京大学", "武汉大学", "中国人民大学"],
            "position": ["总经理", "董事长", "首席执行官", "总裁", "副总裁", "部门经理",
                       "主任", "主管", "教授", "讲师", "医生", "护士", "工程师"],
            "time": ["上午", "下午", "晚上", "早晨", "凌晨", "周一", "周二", "周三",
                   "周四", "周五", "周六", "周日", "一月", "二月", "三月"]
        }
        
        # 预加载jieba词典
        jieba.initialize()
        
    # 使用缓存减少重复分词
    @lru_cache(maxsize=1024)
    def _cached_cut(self, text):
        """缓存分词结果以提高性能"""
        return list(jieba.cut(text))
        
    def augment(self, sample):
        """
        对单个样本进行数据增强
        
        Args:
            sample: 包含text和entity_list的样本
            
        Returns:
            增强后的样本
        """
        # 检查样本是否有效
        if not sample or "text" not in sample or "entity_list" not in sample:
            return sample
            
        # 检查实体列表是否有效
        if not self._validate_entities(sample["text"], sample["entity_list"]):
            return sample
            
        # 以一定概率应用增强
        if random.random() < self.aug_prob:
            # 根据配置选择增强方法
            available_methods = []
            
            if "synonym" in self.methods:
                available_methods.append(self.synonym_replacement)
            if "insert" in self.methods:
                available_methods.append(self.random_insertion)
            if "swap" in self.methods:
                available_methods.append(self.random_swap)
            if "replace" in self.methods:
                available_methods.append(self.entity_replacement)
            
            # 如果没有可用的方法，返回原样本
            if not available_methods:
                return sample
                
            # 随机选择一种增强方法
            method = random.choice(available_methods)
            try:
                augmented_sample = method(copy.deepcopy(sample))
                # 确保增强后的样本是有效的
                if self._validate_entities(augmented_sample["text"], augmented_sample["entity_list"]):
                    return augmented_sample
            except Exception as e:
                # 如果增强过程出错，返回原样本
                print(f"数据增强出错: {e}")
                return sample
        
        return sample
    
    def _validate_entities(self, text, entity_list):
        """验证实体列表是否有效"""
        if not entity_list:
            return True
            
        text_len = len(text)
        for start, end, label in entity_list:
            if start < 0 or end >= text_len or start > end:
                return False
        return True
    
    def augment_batch(self, samples):
        """
        对一批样本进行数据增强
        
        Args:
            samples: 样本列表
            
        Returns:
            增强后的样本列表
        """
        augmented_samples = []
        for sample in samples:
            augmented_samples.append(self.augment(sample))
        return augmented_samples
    
    def synonym_replacement(self, sample):
        """
        同义词替换，替换非实体部分的词语
        """
        text = sample["text"]
        entity_list = sample["entity_list"]
        
        # 提取所有实体的位置
        entity_positions = set()
        for start, end, label in entity_list:
            for i in range(start, end + 1):
                entity_positions.add(i)
        
        # 分词
        words = self._cached_cut(text)
        
        # 记录每个词的起始位置
        positions = []
        current_pos = 0
        for word in words:
            positions.append((current_pos, current_pos + len(word) - 1))
            current_pos += len(word)
        
        # 找出可以替换的词（不在实体中的词）
        replaceable_indices = []
        for i, (start, end) in enumerate(positions):
            # 检查这个词是否与任何实体重叠
            is_entity = False
            for pos in range(start, end + 1):
                if pos in entity_positions:
                    is_entity = True
                    break
            
            if not is_entity and words[i] in self.synonym_dict:
                replaceable_indices.append(i)
        
        # 如果没有可替换的词，返回原样本
        if not replaceable_indices:
            return sample
        
        # 随机选择一个词进行替换
        idx_to_replace = random.choice(replaceable_indices)
        word_to_replace = words[idx_to_replace]
        synonym = random.choice(self.synonym_dict.get(word_to_replace, [word_to_replace]))
        
        # 替换词语
        words[idx_to_replace] = synonym
        new_text = "".join(words)
        
        # 更新实体位置
        new_entity_list = []
        for start, end, label in entity_list:
            # 如果实体在替换位置之前，保持不变
            if end < positions[idx_to_replace][0]:
                new_entity_list.append((start, end, label))
            # 如果实体在替换位置之后，需要调整位置
            elif start > positions[idx_to_replace][1]:
                offset = len(synonym) - len(word_to_replace)
                new_entity_list.append((start + offset, end + offset, label))
            else:
                # 如果实体与替换位置重叠，保持原位置（这种情况不应该发生）
                new_entity_list.append((start, end, label))
        
        return {"text": new_text, "entity_list": new_entity_list}
    
    def random_insertion(self, sample):
        """
        随机插入，在非实体部分随机插入常用词
        """
        text = sample["text"]
        entity_list = sample["entity_list"]
        
        if len(text) > 100:  # 对于长文本，跳过插入操作以提高效率
            return sample
        
        # 提取所有实体的位置
        entity_positions = set()
        for start, end, label in entity_list:
            for i in range(start, end + 1):
                entity_positions.add(i)
        
        # 找出可以插入的位置（不在实体中的位置）
        insertable_positions = []
        for i in range(len(text) + 1):
            # 确保插入点不在实体内部或边界
            if i not in entity_positions and (i == 0 or i == len(text) or (i-1) not in entity_positions or i not in entity_positions):
                insertable_positions.append(i)
        
        # 如果没有可插入的位置，返回原样本
        if not insertable_positions:
            return sample
        
        # 随机选择一个位置进行插入
        pos_to_insert = random.choice(insertable_positions)
        
        # 随机选择一个词插入
        words_to_insert = ["的", "了", "和", "与", "在", "是", "有", "对", "上", "中"]
        word_to_insert = random.choice(words_to_insert)
        
        # 插入词语
        new_text = text[:pos_to_insert] + word_to_insert + text[pos_to_insert:]
        
        # 更新实体位置
        new_entity_list = []
        for start, end, label in entity_list:
            # 如果实体在插入位置之前，保持不变
            if end < pos_to_insert:
                new_entity_list.append((start, end, label))
            # 如果实体在插入位置之后，需要调整位置
            else:
                new_entity_list.append((start + len(word_to_insert), end + len(word_to_insert), label))
        
        return {"text": new_text, "entity_list": new_entity_list}
    
    def random_swap(self, sample):
        """
        随机交换，交换非实体部分的相邻词语
        """
        text = sample["text"]
        entity_list = sample["entity_list"]
        
        if len(text) > 100:  # 对于长文本，跳过交换操作以提高效率
            return sample
        
        # 提取所有实体的位置
        entity_positions = set()
        for start, end, label in entity_list:
            for i in range(start, end + 1):
                entity_positions.add(i)
        
        # 分词
        words = self._cached_cut(text)
        
        # 记录每个词的起始位置
        positions = []
        current_pos = 0
        for word in words:
            positions.append((current_pos, current_pos + len(word) - 1))
            current_pos += len(word)
        
        # 找出可以交换的词对（不在实体中的相邻词）
        swappable_pairs = []
        for i in range(len(words) - 1):
            start1, end1 = positions[i]
            start2, end2 = positions[i + 1]
            
            # 检查这两个词是否与任何实体重叠
            is_entity1 = False
            is_entity2 = False
            
            for pos in range(start1, end1 + 1):
                if pos in entity_positions:
                    is_entity1 = True
                    break
            
            for pos in range(start2, end2 + 1):
                if pos in entity_positions:
                    is_entity2 = True
                    break
            
            # 如果两个词都不在实体中，可以交换
            if not is_entity1 and not is_entity2:
                swappable_pairs.append(i)
        
        # 如果没有可交换的词对，返回原样本
        if not swappable_pairs:
            return sample
        
        # 随机选择一对词进行交换
        idx_to_swap = random.choice(swappable_pairs)
        words[idx_to_swap], words[idx_to_swap + 1] = words[idx_to_swap + 1], words[idx_to_swap]
        
        # 重新构建文本
        new_text = "".join(words)
        
        # 由于交换可能导致实体位置计算复杂，我们简化处理
        # 只保留交换位置之前的实体，并调整交换位置之后的实体
        start1, end1 = positions[idx_to_swap]
        start2, end2 = positions[idx_to_swap + 1]
        
        new_entity_list = []
        for start, end, label in entity_list:
            # 如果实体在交换位置之前，保持不变
            if end < start1:
                new_entity_list.append((start, end, label))
            # 如果实体在交换位置之后，需要调整位置
            elif start > end2:
                new_entity_list.append((start, end, label))
            # 如果实体与交换位置重叠，保持原位置（这种情况不应该发生）
            else:
                new_entity_list.append((start, end, label))
        
        return {"text": new_text, "entity_list": new_entity_list}
    
    def entity_replacement(self, sample):
        """
        实体替换，用同类型的其他实体替换当前实体
        """
        text = sample["text"]
        entity_list = sample["entity_list"]
        
        if not entity_list:
            return sample
        
        # 随机选择一个实体进行替换
        entity_idx = random.randint(0, len(entity_list) - 1)
        start, end, label = entity_list[entity_idx]
        
        # 获取实体文本和类型
        entity_text = self.get_entity_text(text, (start, end, label))
        
        # 将CLUENER标签映射到我们的实体样例库
        label_mapping = {
            "address": "address",
            "book": "book",
            "company": "company",
            "game": "game",
            "government": "government",
            "movie": "movie",
            "name": "name",
            "organization": "organization",
            "position": "position",
            "time": "time"
        }
        
        mapped_label = label_mapping.get(label, None)
        
        # 如果该类型的实体有样例库，则从中选择一个替换
        if mapped_label in self.entity_examples:
            # 选择一个不同于原实体的替换
            replacements = [e for e in self.entity_examples[mapped_label] if e != entity_text]
            if not replacements:
                return sample
                
            new_entity = random.choice(replacements)
            
            # 替换实体
            new_text = text[:start] + new_entity + text[end+1:]
            
            # 更新实体列表
            new_entity_list = []
            length_diff = len(new_entity) - (end - start + 1)
            
            for i, (e_start, e_end, e_label) in enumerate(entity_list):
                if i == entity_idx:
                    # 更新被替换的实体
                    new_entity_list.append((e_start, e_start + len(new_entity) - 1, e_label))
                elif e_start > end:
                    # 更新在替换实体之后的实体位置
                    new_entity_list.append((e_start + length_diff, e_end + length_diff, e_label))
                else:
                    # 保持替换实体之前的实体位置不变
                    new_entity_list.append((e_start, e_end, e_label))
            
            return {"text": new_text, "entity_list": new_entity_list}
        
        return sample
    
    @staticmethod
    def get_entity_text(text, entity):
        """
        获取实体文本
        """
        start, end, label = entity
        # 确保索引在有效范围内
        if 0 <= start <= end < len(text):
            return text[start:end+1]
        return "" 