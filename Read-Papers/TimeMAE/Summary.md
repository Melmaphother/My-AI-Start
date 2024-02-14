# TimeMAE
## 前述
论文地址：

读论文实在是 **太痛苦了**！

我觉得 Chinese Programmer 相比于 American Programmer 需要改进的（差距最大的）就是：

1. 大学时代的教育，包括教育水平、资源。
2. 原生英文水平，这真的很难后天改变。
3. GFW，不过这个其实问题也不算很大。
   反正也不是没有网开一面，有2的话各种论坛阅读也没什么难度
   不得不承认的是，如果没有GFW，大家都去用 Google 那一套，那应该就没有 Tencent or Baidu
   （虽然说这也不应该是这两家公司让我很不爽的资本）
   那可能现在在这里摆的就不是 Chinese Programmer 和 American Programmer 的比较，而是 California Programmer 
   Mississippi Programmer 的比较了🤣。
   

要不来点《中国程序员失掉自信力了吗》？😀。

## 文件结构解释

- dataset.py
  定义数据集的类，定义了该数据集的一些内置方法

- datautils.py
  数据集分类和处理

- model
  - layers.py
    transformer 的 layers 层
  - TimeMAE.py
    TimeMAE 模型

- args.py
  根据 datautils 中处理数据集的方法设置参数。
  并设置其他参数，比如 cuda 之类

- classification.py
  分类器，只有两个方法

- loss.py
  计算 loss

- process.py
  定义训练类和训练函数

- visualize.py
  可视化结果

- main.py
  运行整个流程

- run.sh
  脚本文件，用于运行 main.py 和设置参数
