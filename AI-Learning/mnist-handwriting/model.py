import torch.nn as nn


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # 1是输入通道数，10是输出通道数，kernel_size是卷积核大小
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()  # 二维dropout
        # 下面 Linear 为线性层 (全连接层)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
    
    def forward(self, x):
        # x: [batch_size, 1, 28, 28]
        x = nn.functional.relu(nn.functional.max_pool2d(self.conv1(x), 2))
        # x: [batch_size, 10, 12, 12]
        x = nn.functional.relu(nn.functional.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # x: [batch_size, 20, 4, 4]
        x = x.view(-1, 320)
        # x: [batch_size, 320]
        x = nn.functional.relu(self.fc1(x))
        # x: [batch_size, 50]
        x = nn.functional.dropout(x, training=self.training)
        x = self.fc2(x)
        # x: [batch_size, 10]
        return nn.functional.log_softmax(x, dim=1) # softmax 归一化
