# About TimeMAE

## File Structure

- `dataset.py`
  定义数据集的类，定义了该数据集的一些内置方法

- `datautils.py`
  数据集分类和处理

- `model`
  - `layers.py`
    `transformer` 的 layers 层
  - `TimeMAE.py`
    TimeMAE 模型

- `args.py`
  根据 `datautils` 中处理数据集的方法设置参数。
  并设置其他参数，比如 cuda 之类

- `classification.py`
  分类器，只有两个方法

- `loss.py`
  计算 loss

- `process.py`
  定义训练类和训练函数

- `visualize.py`
  可视化结果

- `main.py`
  运行整个流程

- `run.sh`
  脚本文件，用于运行 `main.py` 和设置超参数
  运行三次，猜测可能是计算误差。

## Dataset
仅介绍 HAR 数据集：使用 load_HAR 函数，可以发现：TRAIN_DATA_ALL, TRAIN_DATA, TEST_DATA 分别用于预训练、微调和测试。其中各个数据集的规模如下：

![](assets/1.png)

其中 TRAIN_DATA_ALL 是 TRAIN 和 VAL 的拼接。

**重点：**
在设定了 batch-size 之后，进入 encoder 的 tensor 结构是：(batch_size, sequence_length, dimension) =（128， 7， 64）


## Model and Encoder

在模型中，使用自定义多层的 transformer 模型 TransformerBlock 做 Encoder，如下：
```python
class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        d_model = args.d_model
        attn_heads = args.attn_heads
        d_ffn = 4 * d_model
        layers = args.layers
        dropout = args.dropout
        enable_res_parameter = args.enable_res_parameter
        # TRMs
        self.TRMs = nn.ModuleList(
            [TransformerBlock(d_model, attn_heads, d_ffn, enable_res_parameter, dropout) for i in range(layers)])

    def forward(self, x):
        for TRM in self.TRMs:
            x = TRM(x, mask=None)
        return x
```

- d_model 表示 Transformer 编码器的隐藏层维度。
- attn_heads 表示注意力头的数量。
- d_ffn 是 FeedForward 层的隐藏层维度，这里设置为隐藏层维度的 4 倍。
- layers 表示 Transformer 编码器的层数。
- dropout 是 dropout 概率。
- enable_res_parameter 是一个布尔值，表示是否启用残差连接中的可学习参数。
- self.TRMs 是一个由多个 TransformerBlock 组成的列表，构建了 Transformer 编码器。

## Run
在服务器上搭建环境，运行之后的结果为（仅 HAR 数据集）：

![](assets/2.png)

与论文中对比，可见结果非常符合。

![](assets/3.png)

## Modify

要求用 BERT 的网络架构和网络参数初始化模型，并替换 TimeMAE 的Encoder。这里做了一些尝试，修改了 TimeMAE 的 Encoder 部分：

```python
from transformers import BertModel, BertConfig

class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        bert_config = BertConfig(
            hidden_size=args.d_model,
            num_hidden_layers=args.layers,
            num_attention_heads=args.attn_heads,
            intermediate_size=4 * args.d_model,
            hidden_dropout_prob=args.dropout,
            attention_probs_dropout_prob=args.dropout
        )
        self.bert_model = BertModel(config=bert_config)

    def forward(self, x):
        x = self.bert_model(x)
        x = x.last_hidden_state
        return x
```

但是这里遇到了一系列问题：BertModel 接收的参数是一个规模为 （batch_size, sequence_length）的 **整形** tensor，而这里的时间序列数据是一个规模为 (batch_size, sequence_length, dimension) =（128， 7， 64）的浮点型 tensor。使用了一些降维方式，但是仍然无效。

## TODO

研究如何将数据映射为 BertModel 可接受的输入方式。

> 不知是否是我方向有点错误？


## 论文精读

### MCC & MRR 优化
在论文摘要中，**Masked Codeword Classification (MCC)** 任务和 **Masked Representation Regression (MRR)** 优化是两种预文本任务，用于自监督学习框架中以提高模型学习效果。下面是对这两个任务的解释：

#### Masked Codeword Classification (MCC) ：掩码码字分类

- **目的**：MCC任务的目的是提高模型对时间序列数据中各个部分（尤其是被随机掩盖的部分）的理解和表示能力。
- **工作方式**：在进行MCC任务时，模型首先将时间序列的一部分随机掩盖。然后，模型需要预测这些被掩盖部分的内容。这一预测不是直接恢复原始数据，而是将掩盖的部分映射到一个预定义的词汇表（或代码字集）上，模型需要预测每个掩盖部分最可能对应的代码字。
- **作用**：这种方法强迫模型学习时间序列中局部信息的高层次抽象表示，并通过分类任务来加深对时间序列局部特征的理解。

#### Masked Representation Regression (MRR) 优化

- **目的**：MRR优化的目标是提升模型在处理和理解时间序列数据中被掩盖部分的能力，特别是在连续值预测方面。
- **工作方式**：在MRR任务中，模型同样需要处理含有被随机掩盖部分的时间序列。不同于MCC的是，MRR要求模型预测被掩盖部分的实际数值表示，而非进行分类。这通常涉及到一个回归任务，模型需要生成一个接近于原始被掩盖部分的连续数值向量。
- **作用**：通过这种方式，MRR促进模型学习到更精细、更接近于数据真实分布的表示。这有助于模型捕获时间序列数据的细微变化，提高模型对数据的整体理解。

总体而言，MCC和MRR这两种预文本任务通过不同的方法促进模型更好地学习和理解时间序列数据，增强模型的泛化能力。MCC通过分类任务加强对离散特征的学习，而MRR通过回归任务提升对连续数值特征的理解，二者共同作用，使模型能够更全面、更深入地捕捉时间序列数据的特性。

### self-supervised learning

自监督学习是一种特殊的无监督学习，但它模拟了监督学习的过程。在自监督学习中，模型从输入数据本身自动生成伪标签，然后使用这些伪标签来训练。这种方法利用数据的内在结构来学习数据的表示，而不是依赖于外部标注。自监督学习的典型应用包括自然语言处理中的语言模型预训练、图像处理中的预训练模型等。

### 关于时间序列数据

1. 本文聚焦的问题在于：虽然先进的传感器设备使收集时间序列数据变得容易，但是在现实场景中，不可能全是先进的传感器，所以某些场景中获得大量准确的标注时间序列数据是耗时的，容易出错的。然而传统的 Transformer 架构严重依赖大量的训练标签。

2. 前人的工作：self-supervised learning 在从未标注数据中学习可转移的模型参数已经取得了巨大的成功。这种方法基本方案是首先获得一个预训练（pre-training）模型，从原始未标注数据中优化一个设定的前置任务。然后通过微调（finetuning）形成最终的模型。当前工作中，对比学习范式最为流行。对比学习的共同范式是学习 embedding，并且假设这些 embedding 对各种规模输入的扭曲具有不变性。尽管这些方法很流行，但是**不变性假设在现实世界中可能不总是成立**。并且它在开发数据增强策略中带来太多归纳偏差，并且在负采样中引入额外的偏差。更大的缺点是：它们本质上使用**单向编码器的（unidirectional encoder）**方案来学习时间序列的表示，限制了**上下文表示（contextual representations）**的提取。
3. 前述问题目前有很好的结果：掩码自编码器（Masked AutoEncoder）ps：缩写就是 MAE。

