import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import time

# 1.准备数据
# 对数据进行载入及有相应变换,将Compose看成一种容器，他能对多种数据变换进行组合
# 传入的参数是一个列表，列表中的元素就是对载入的数据进行的各种变换操作
transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Grayscale(num_output_channels=1),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])  # 修改的位置

# 首先获取手写数字的训练集和测试集
# root 用于指定数据集在下载之后的存放路径
# transform 用于指定导入数据集需要对数据进行那种变化操作
# train是指定在数据集下载完成后需要载入那部分数据，
# 如果设置为True 则说明载入的是该数据集的训练集部分
# 如果设置为FALSE 则说明载入的是该数据集的测试集部分
data_train = datasets.MNIST(root="./data/",
                            transform=transform,
                            train=True,
                            download=False)

data_test = datasets.MNIST(root="./data/",
                           transform=transform,
                           train=False)

data_loader_train = torch.utils.data.DataLoader(dataset=data_train,
                                                batch_size=512,
                                                shuffle=True)
data_loader_test = torch.utils.data.DataLoader(dataset=data_test,
                                               batch_size=512,
                                               shuffle=True)


# 2.构建模型


class Model(torch.nn.Module):  # 方法一
    def __init__(self):
        super().__init__()  # 继承父类的__init__()
        self.conv1 = nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(stride=2, kernel_size=2)
        )
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(14 * 14 * 128, 1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(1024, 10)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(-1, 14 * 14 * 128)
        x = self.dense(x)
        return x


model = Model()
# 将所有的模型参数移动到GPU上
if torch.cuda.is_available():
    model.cuda()
cost = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
# print(model)

# 卷积神经网络模型进行模型训练和参数优化的代码
n_epochs = 1

for epoch in range(n_epochs):
    running_loss = 0.0
    running_correct = 0
    print("Epoch  {}/{}".format(epoch, n_epochs))
    print("-" * 10)
    for data in data_loader_train:
        X_train, y_train = data
        # 有GPU加下面这行，没有不用加
        # X_train, y_train = X_train.cuda(), y_train.cuda()
        X_train, y_train = Variable(X_train), Variable(y_train)
        outputs = model(X_train)
        _, pred = torch.max(outputs.data, 1)
        optimizer.zero_grad()
        loss = cost(outputs, y_train)
        loss.backward()
        optimizer.step()
        running_loss = loss.item()
        running_correct = torch.sum(pred == y_train.data)
        print("Loss: {:.4f}  train accuracy: {:.3f}%".format(running_loss, 100 * running_correct / len(y_train.data)))

    print("train ok ")
    testing_correct = 0
    for data in data_loader_test:
        X_test, y_test = data
        # 有GPU加下面这行，没有不用加
        # X_test, y_test = X_test.cuda(), y_test.cuda()
        X_test, y_test = Variable(X_test), Variable(y_test)
        outputs = model(X_test)
        _, pred = torch.max(outputs, 1)
        testing_correct += torch.sum(pred == y_test.data)
        # print("test accuracy {}%".format(100 * testing_correct / 512))
        # print(testing_correct)

    print("Loss is :{:.4f},Test Accuracy is:{:.4f}".format(
        running_loss / len(data_train), 100 * testing_correct / len(data_test)))

