import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.optim import lr_scheduler
from matplotlib import pyplot as plt

# 下载mnist数据集
mnist_train = datasets.MNIST('mnist', train=True, transform=transforms.ToTensor(), download=True)
mnist_train = DataLoader(mnist_train, batch_size=64, shuffle=True, num_workers=4)

mnist_test = datasets.MNIST('mnist', train=False, transform=transforms.ToTensor(), download=True)
mnist_test = DataLoader(mnist_test, batch_size=64, num_workers=4)

# 自编码器的网络结构
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        # 编码器的网络结构
        self.encoder = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 20),
            nn.ReLU()
        )
        # 译码器的网络结构
        self.decoder = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Sigmoid()
        )

    def forward(self, x):
        batchsz = x.size(0)
        x = x.view(batchsz, -1)  # 将输入数据展平为一维向量
        x = self.encoder(x)  # 经过编码器得到特征表示
        features = x.clone()  # 保存特征表示
        x = self.decoder(x)  # 经过译码器重构数据
        x = x.view(batchsz, 1, 28, 28)  # 将数据重新转换为图片的形状
        return x, features

def test(model, dataloader, criterion):
    model.eval()  # 设置为评估模式
    running_loss = 0.0  # 用于累计每个batch的损失值
    correct = 0  
    total = 0  

    with torch.no_grad():  # 不需要计算梯度
        for images, labels in dataloader:
            images = images.to(device)  
            labels = labels.to(device)  
            reconstructions, features = model(images)  # 通过模型进行前向传播，得到预测结果
            labels = labels.unsqueeze(1)  # 保持标签维度为 [batch_size, 1]
            loss = criterion(features, images)  # 计算损失函数值
            running_loss += loss.item()  # 累加损失函数值
            _, predicted = features.max(1)  # 获取预测结果中概率最高的类别
            total += labels.size(0)  
            correct += predicted.eq(labels).sum().item()  

    accuracy = correct / total  # 计算准确率
    return accuracy

# 训练过程
def train(epochs, model, criterion, optimizer, scheduler, train_loader):
    model.train()  # 将模型设置为训练模式
    loss_set = []  
    
    for epoch in range(epochs):
        scheduler.step()  # 调整学习率
        total_loss = 0.0  # 用于累计每个epoch的损失值

        for i, (img, label) in enumerate(train_loader):
            img = img.to(device)  # 将输入数据移动到GPU
            label = label.to(device)  # 将标签数据移动到GPU
            optimizer.zero_grad()  # 梯度归零，清除之前的梯度信息
            output, features = model(img)  # 通过模型进行前向传播，得到输出和特征表示
            loss = criterion(output, img)  # 计算损失函数值
            loss.backward()  # 反向传播，计算梯度
            optimizer.step()  # 更新模型参数
            total_loss += loss.item()  # 累加损失函数值
            _, predicted = torch.max(features.data, 1)  # 获取预测结果中概率最高的类别
            
            # 打印当前的训练信息
            print("Epoch: {}/{}, Step: {}, Loss: {:.4f}".format(epoch + 1, epochs, i + 1, loss.item()))
            loss_set.append(loss.item())  
    return loss_set  



# 测试过程
def Image_all(model, test_loader):
    model.eval()  # 将模型设置为评估模式
    N = 8  # 子图的行数
    M = 8  # 子图的列数

    with torch.no_grad():  # 不需要计算梯度
        images, _ = next(iter(test_loader))  # 获取测试数据集的一个batch
        images = images.to(device)  # 将测试数据移动到指定设备GPU
        _images, _ = model(images)  # 通过模型进行前向传播，得到重构图像

    p1 = plt.figure(1)  # 创建第一个图形窗口

    # 绘制原始图像
    for i in range(N * M):
        plt.subplot(N, M, i + 1)
        plt.imshow(images[i].cpu().numpy().squeeze(), cmap='gray_r')
        plt.xticks([])
        plt.yticks([])

    p2 = plt.figure(2)  # 创建第二个图形窗口

    # 绘制重构图像
    for i in range(N * M):
        plt.subplot(N, M, i + 1)
        plt.imshow(_images[i].cpu().numpy().squeeze(), cmap='gray_r')
        plt.xticks([])
        plt.yticks([])

    plt.show()  # 显示图像


def Image_sep(num):
    model.eval()  # 将模型设置为评估模式
    N = 2  # 子图的行数
    M = 5  # 子图的列数
    num_images = 10  # 要显示的图像数量
    cnt = 0  # 计数器，用于控制显示的图像数量

    p1 = plt.figure(1)  
    p2 = plt.figure(2)  

    dataiter = iter(mnist_test)  # 创建测试数据集的迭代器
    images, labels = next(dataiter)  # 获取测试数据集的一个batch的图像和标签

    while cnt < num_images:
        for i, (image, label) in enumerate(zip(images, labels)):
            if label == num:
                plt.figure(1)
                plt.subplot(N, M, cnt + 1)
                plt.imshow(image.cpu().numpy().squeeze(), cmap='gray_r')
                plt.xticks([])
                plt.yticks([])

                with torch.no_grad():
                    image = image.unsqueeze(0).to(device)
                    reconstructed_image, _ = model(image)

                plt.figure(2)
                plt.subplot(N, M, cnt + 1)
                plt.imshow(reconstructed_image.squeeze().cpu().numpy(), cmap='gray_r')
                plt.xticks([])
                plt.yticks([])

                cnt += 1
                if cnt == num_images:
                    break

        try:
            images, labels = next(dataiter)  # 获取下一个batch的图像和标签
        except StopIteration:
            break

    plt.show() 



if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
    model = AutoEncoder().to(device)  # 创建自编码器模型，并将其移动到指定设备上
    epochs_num = 2  # 训练的总轮数
    num_epochs = 50  # 测试的总轮数
    test_accuracy_set = []  # 存储测试准确率的列表
    criterion = nn.MSELoss()  # 定义损失函数
    learn_rate = 1e-3  # 学习率
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)  # 创建优化器
    scheduler = lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.1)  # 设置学习率衰减策略
    loss_set = train(epochs_num, model, criterion, optimizer, scheduler, mnist_train)  # 进行训练并返回损失值列表
    index_train = [i for i in range(len(loss_set))]  # 创建用于绘制训练损失曲线的横坐标列表
    fig1 = plt.figure(1)  
    plt.plot(index_train, loss_set)  
    plt.xlabel("Steps")  
    plt.ylabel("Loss")  
    plt.show()  

    Image_all(model, mnist_test)  # 显示所有测试图像和对应的重构图像

    for _num in range(0, 10):
        Image_sep(_num)  # 分别显示每个数字类别的原始图像和对应的重构图像

    for epoch in range(num_epochs):
        test_accuracy = test(model, mnist_test, criterion)  # 计算测试准确率
        test_accuracy_set.append(test_accuracy)  # 将测试准确率添加到列表中

    index_test = [i for i in range(len(test_accuracy_set))]  # 创建用于绘制测试准确率曲线的横坐标列表
    fig2 = plt.figure(2)  
    plt.plot(index_test, test_accuracy_set)  # 绘制测试准确率曲线
    plt.xlabel("Epochs")  
    plt.ylabel("Accuracy")  
    plt.show()  

