# Learning Transferable TS Classifier

## Abstract

1. 本篇论文着眼于多领域时间序列数据的可迁移知识。由于不同领域中时间序列数据的特性存在显著差异，例如通道数量和时间分辨率的变化。

2. 论文中提出了 CrossTimeNet ，一个关键特性是新设计的时间序列标记化模块，它能够基于重构优化过程有效地将原始时间序列转换为一系列离散令牌。

3. 论文将预训练语言模型（PLM）视为编码器网络的初始化，探究将PLM学到的知识转移到时间序列领域的可行性，以发展通用时间序列表示。

## Architecture

`CrossTimeNet` 包含三个组件：

1. 时间序列 tokenization：将连续的时间序列数据转成离散的 tokens，这是为了建立跨领域的统一表征。

2. 跨领域 自监督预训练模型：这个阶段需要完成 双向 token 预测任务。在这个任务中，随机的 tokens 被 mask，用于逼迫模型推断 missing 的信息。由此学习到强大的时间序列数据的表征。

3. 下游的特殊领域任务 微调：模型经历特殊的调整去适应和擅长于某个领域的任务，比如分类任务。微调过程串联了从预训练过程中得到的大量的 knowledge。

   微调任务的操作是很仔细的，为了保证预训练模型既精通于特殊的任务，同样保持了之前从跨领域任务中学到的洞察力。

## Tokenization

时间序列的处理困难主要来自于：channel 数量可变、物理现象表征的不同、时间分辨率不同。

为了解决这些问题：有几种方式

1. channel independent：但是这种方式忽略了channel 之间的依赖关系。

## 单词

depict：描绘

tailor-made：特制的

potent：强大的

potential：有潜力的

harness：串联

meticulously：细致的，一丝不苟的

proficiency：熟练，精通

insight：洞察力

distinct：不同的，有区别的，清楚的

inherent：固有的

variability：可变性

modality：形式，形态

temporal resolution：时间分辨率，可以理解为 `DeltaTime`