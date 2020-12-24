# -*- coding: utf-8 -*-
# @Time : 2020/12/23 15:28
# @Author : Jclian91
# @File : model_predict.py
# @Place : Yangpu, Shanghai
# 模型预测脚本

import time
import json
import numpy as np

from model_train import token_dict, OurTokenizer
from keras.models import load_model
from keras_bert import get_custom_objects

maxlen = 256

# 加载训练好的模型
model = load_model("multi-label-ee.h5", custom_objects=get_custom_objects())
tokenizer = OurTokenizer(token_dict)
with open("label.json", "r", encoding="utf-8") as f:
    label_dict = json.loads(f.read())

s_time = time.time()
# 预测示例语句
text = "昨天18：30，陕西宁强县胡家坝镇向家沟村三组发生山体坍塌，5人被埋。当晚，3人被救出，其中1人在医院抢救无效死亡，2人在送医途中死亡。今天凌晨，另外2人被发现，已无生命迹象。"
text = "注意！济南可能有雷电事故｜英才学院14.9亿被收购｜八里桥蔬菜市场今日拆除，未来将建新的商业综合体"
text = "截止到11日13：30 ，因台风致浙江32人死亡，16人失联。具体如下：永嘉县岩坦镇山早村23死9失联，乐清6死，临安区岛石镇银坑村3死4失联，临海市东塍镇王加山村3失联。"
text = "据猛龙随队记者JoshLewenberg报道，消息人士透露，猛龙已将前锋萨加巴-科纳特裁掉。此前他与猛龙签下了一份Exhibit10合同。在被裁掉后，科纳特下赛季大概率将前往猛龙的发展联盟球队效力。"
text = "计算机行业大变革？甲骨文中国区裁员，IBM收购红帽公司"
text = "美的置业：贵阳项目挡墙垮塌致8人遇难已责令全面停工"


# 利用BERT进行tokenize
text = text[:maxlen]
x1, x2 = tokenizer.encode(first=text)

X1 = x1 + [0] * (maxlen-len(x1)) if len(x1) < maxlen else x1
X2 = x2 + [0] * (maxlen-len(x2)) if len(x2) < maxlen else x2

# 模型预测并输出预测结果
prediction = model.predict([[X1], [X2]])
one_hot = np.where(prediction > 0.5, 1, 0)[0]


print("原文: %s" % text)
print("预测标签: %s" % [label_dict[str(i)] for i in range(len(one_hot)) if one_hot[i]])
e_time = time.time()
print("cost time:", e_time-s_time)