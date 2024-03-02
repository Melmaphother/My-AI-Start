# python用于机器学习常用库的简要介绍

## torch

> pytorch库有最低和最高python版本限制，目前（2024/1/4）尝试得正式版 pytorch **最高 python 版本限制是 3.10**，最低 python 版本按照 pytorch 官网的说法是 3.8

> 由于 conda 解析速度过慢，所以改用 pip 下载，pytorch 官网找下载 command
>
> 下载速度： eduroam 非拥挤网络 10~20MB/s

**Torch库**（PyTorch）：

- Torch是一个开源的深度学习框架，专门用于构建和训练神经网络。
- PyTorch提供了灵活的张量（tensor）操作和自动微分功能，使神经网络的构建和训练更加容易。
- PyTorch有强大的GPU支持，能够在GPU上高效运行神经网络，从而加速训练过程。

## sympy  and scipy

符号计算（sympy）和数值计算（scipy）

## matplotlib

绘图

```bash
conda install matplotlib
```

## pandas

```bash
conda install pandas
```

**Pandas库**（Python Data Analysis Library）：

- Pandas是一个开源的Python数据分析库，广泛用于数据处理和分析。
- 主要的数据结构是DataFrame和Series，它们使数据的导入、清洗、转换和分析变得更加简单。
- Pandas允许用户处理各种数据类型，包括时间序列数据、表格数据、SQL表格等。
- 提供了广泛的数据操作和分析功能，如索引、切片、过滤、排序、合并、分组、聚合等。
- 可以将数据导入和导出到各种文件格式，如CSV、Excel、SQL数据库等。

## numpy

```bash
conda install numpy
# 一般自带
```

**NumPy库（Numerical Python）**

是一个开源的Python库，用于进行科学计算和数值计算。NumPy提供了一个强大的多维数组对象（称为`ndarray`），以及广泛的数学函数和工具，用于在这些数组上执行各种操作。

## tqdm

```bash
conda install tqdm
```

**TQDM库（"TQDM: Fast, Extensible Progress Bar for Loops and CLI"）**

来自拉丁词 "taqaddum"（也可写作"taqdīm"），意为"前进"、"进步"或"前进的步骤"。

用于在循环、迭代和命令行界面（CLI）中显示进度条，以提供直观的进度反馈。TQDM允许你轻松地跟踪长时间运行的任务的进展，对于处理大型数据集、训练深度学习模型、爬虫等任务特别有用。

## sklearn or scikit-learn

"sklearn" 和 "scikit-learn" 是指同一个库，即用于机器学习和数据挖掘的 Python 库。官方名称为 "scikit-learn"，但由于其名称有点长，因此人们经常使用缩写 "sklearn" 来引用这个库。这两者是等效的，都指代相同的库。

scikit-learn（sklearn）是一个广泛用于机器学习和数据挖掘任务的开源库，它提供了各种用于监督学习、无监督学习、特征工程、模型评估和模型选择等任务的工具和算法。它建立在 Python 的 NumPy 和 SciPy 库之上，提供了用户友好的 API 和丰富的功能，使机器学习任务变得更加容易。

## transformer

"transformer" 是一个用于自然语言处理（NLP）任务的 Python 库，主要用于预训练和微调大型语言模型，如BERT、GPT、RoBERTa等。这些模型是在深度学习领域取得了重大突破的转换器架构（Transformer）的基础上构建的。

以下是 "transformer" 库的一些主要特点和功能：

1. **支持多种预训练模型**："transformer" 支持多种预训练的 NLP 模型，包括 BERT、GPT、RoBERTa、T5 等，可以用于各种自然语言处理任务，如文本分类、命名实体识别、文本生成等。
2. **简化模型加载和微调**：库提供了用于加载和微调这些预训练模型的简单 API，使用户能够方便地将它们应用于自己的文本数据。
3. **支持多种任务**："transformer" 库支持多种常见的 NLP 任务，包括文本分类、文本生成、文本匹配、问答系统等，以及用于这些任务的示例代码。
4. **GPU 支持**："transformer" 充分利用 GPU 进行模型训练和推断，提高了处理大规模文本数据的效率。

## tenserflow

## Huggingface



