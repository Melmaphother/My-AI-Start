from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def load_mnist(batch_size=64, test_batch_size=1000):
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transforms.Compose([
                                   transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
    # 0.1307和0.3081是MNIST数据集的全局平均值和标偏差
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transforms.Compose([
                                  transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # shuffle=True表示每个epoch都会打乱数据集

    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True)

    return train_loader, test_loader
