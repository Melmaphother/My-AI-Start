import torch
import torch.nn.functional as F


class Trainer():
    def train(model, device, train_loader, optimizer, epoch):
        model.train()
        # batch_idx：这是当前批次的索引号，从 0 开始递增。
        # data：每一个批次中的数据
        # target：每一个批次中数据对应的标签（标准的 supervised learning）
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()  # 梯度清零
            output = model(data)
            loss = F.nll_loss(output, target)  # 负对数似然损失
            loss.backward()
            optimizer.step()
            if (batch_idx + 1) % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
        # len(train_loader) = len(train_loader.dataset) / batch_size = 60000 / 64 = 937.5 向上取整

    def test(model, device, test_loader):
        model.eval()  # 将模型设置为评估模式
        test_loss = 0
        correct = 0
        _len = len(test_loader.dataset)
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)  # 获取最大值的索引
                correct += pred.eq(target.view_as(pred)
                                   ).sum().item()  # 预测正确的数量

        test_loss /= _len
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, _len, 100. * correct / _len))
