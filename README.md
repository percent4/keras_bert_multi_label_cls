本项目采用Keras和Keras-bert实现文本多标签分类任务。

### 维护者

- jclian91

### 数据集

#### 2020语言与智能技术竞赛：事件抽取任务

本项目以 2020语言与智能技术竞赛：事件抽取任务 中的数据作为多分类标签的样例数据，借助多标签分类模型来解决。

### 代码结构

```
.
├── chinese_L-12_H-768_A-12（BERT中文预训练模型）
│   ├── bert_config.json
│   ├── bert_model.ckpt.data-00000-of-00001
│   ├── bert_model.ckpt.index
│   ├── bert_model.ckpt.meta
│   └── vocab.txt
├── data（数据集）
├── label.json（类别词典，生成文件）
├── model_evaluate.py（模型评估脚本）
├── model_predict.py（模型预测脚本）
├── model_train.py（模型训练脚本）
└── requirements.txt
```

## 模型效果

#### sougou数据集

模型参数: batch_size = 8, maxlen = 256, epoch=10

评估结果:

```
accuracy:  0.8524699599465955
hamming loss:  0.001869158878504673
```

### 项目启动

1. 将BERT中文预训练模型chinese_L-12_H-768_A-12放在chinese_L-12_H-768_A-12文件夹下
2. 所需Python第三方模块参考requirements.txt文档
3. 自己需要分类的数据按照data/train.py的格式准备好
4. 调整模型参数，运行model_train.py进行模型训练
5. 运行model_evaluate.py进行模型评估
6. 运行model_predict.py对新文本进行评估