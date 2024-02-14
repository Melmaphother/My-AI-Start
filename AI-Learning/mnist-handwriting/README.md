# MNIST-HandWriting

## 简述
作为深度学习的 Hello World，使用 MNIST 数据集实现手写数字识别可能是除 Linear Regression 之外最简单的 
深度学习应用了。这个 Program 实现了类似的功能。

虽然这个应用很基础也很简单，但是包含了卷积神经网络（CNN）构建的基本流程：
- 数据集准备
- 数据预处理（归一化等），分离测试集和训练集
- 定义模型（一般继承 nn.Module）
- 调整超参数（可以使用 argparse 库）
- 定义优化器（SGD，Momentum，Adam 等）并训练模型
- 测试模型，得出 accuracy（准确率、召回率等）和 loss

## 文件结构说明

- dataset.py：数据集处理
  - 数据集下载
  - 数据集转成 Tensor 并归一化
  - 数据集加载 DataLoader
  这些操作均分离成 train 集和 test 集

- model.py：模型设置
  - 定义卷积层和全连接层
  - 定义前馈函数

- args.py：超参数设置
  - 批次
  - 学习率
  - 训练轮次
  - 指定是否为 cuda
  - 优化器参数 Momentum

- trainer.py：训练和测试类
  - train 函数
  - test 函数

- process.py：运行整个流程
