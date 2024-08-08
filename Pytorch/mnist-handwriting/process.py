from dataset import load_mnist
from model import ConvNet
from args import args
from trainer import Trainer
import torch.optim as optim

# 准备数据
train_loader, test_loader = load_mnist(args.batch_size, args.test_batch_size)
# 定义模型
model = ConvNet().to(args.device)
# 定义优化器
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

# 训练
for epoch in range(1, args.epochs + 1):
    Trainer.train(model, args.device, train_loader, optimizer, epoch)
    Trainer.test(model, args.device, test_loader)