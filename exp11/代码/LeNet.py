import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt

# 定义LeNet模型
import torch
import torch.nn as nn

class LeNet(nn.Module):
    def __init__(self, num_classes):
        super(LeNet, self).__init__()
        # 定义特征提取部分
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5),  # 输入通道数为3，输出通道数为6，卷积核大小为5x5
            nn.ReLU(inplace=True),  # 使用ReLU激活函数
            nn.MaxPool2d(kernel_size=2, stride=2),  # 最大池化层，池化窗口大小为2x2，步幅为2
            nn.Conv2d(6, 16, kernel_size=5),  # 输入通道数为6，输出通道数为16，卷积核大小为5x5
            nn.ReLU(inplace=True),  # 使用ReLU激活函数
            nn.MaxPool2d(kernel_size=2, stride=2)  # 最大池化层，池化窗口大小为2x2，步幅为2
        )
        # 定义分类部分
        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),  # 全连接层，输入大小为16x5x5，输出大小为120
            nn.ReLU(inplace=True),  # 使用ReLU激活函数
            nn.Linear(120, 84),  # 全连接层，输入大小为120，输出大小为84
            nn.ReLU(inplace=True),  # 使用ReLU激活函数
            nn.Linear(84, num_classes)  # 全连接层，输入大小为84，输出大小为num_classes
        )

    def forward(self, x):
        x = self.features(x)  
        x = torch.flatten(x, 1)  # 将特征展平为一维向量
        x = self.classifier(x)  
        return x



train_path=r'D:\train'
test_path=r'D:\test'

# 数据预处理和转换
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 创建数据集
train_dataset = ImageFolder(train_path, transform=transform)
test_dataset = ImageFolder(test_path, transform=transform)

# 创建数据加载器
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 创建LeNet模型
num_classes = 5
model = LeNet(num_classes)

# 将模型放到GPU上
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001) # 学习率

def train(model, dataloader, criterion, optimizer):
    model.train()  # 设置为训练模式
    running_loss = 0.0  # 用于累计每个batch的损失值
    correct = 0  
    total = 0  

    for images, labels in dataloader:
        images = images.to(device)  
        labels = labels.to(device)  

        optimizer.zero_grad()  # 清除梯度

        outputs = model(images)  # 通过模型进行前向传播，得到预测结果
        loss = criterion(outputs, labels)  # 计算损失函数值
        loss.backward()  # 反向传播，计算梯度
        optimizer.step()  

        running_loss += loss.item()  # 累加损失函数值
        _, predicted = outputs.max(1)  # 获取预测结果中概率最高的类别
        total += labels.size(0)  
        correct += predicted.eq(labels).sum().item()  

    epoch_loss = running_loss / len(dataloader)  # 计算平均损失函数值
    accuracy = correct / total  # 计算准确率

    return epoch_loss, accuracy


# 定义测试函数
def test(model, dataloader, criterion):
    model.eval()  # 设置为评估模式
    running_loss = 0.0  # 用于累计每个batch的损失值
    correct = 0  
    total = 0  

    with torch.no_grad():  # 不需要计算梯度
        for images, labels in dataloader:
            images = images.to(device)  
            labels = labels.to(device)  

            outputs = model(images)  # 通过模型进行前向传播，得到预测结果
            loss = criterion(outputs, labels)  # 计算损失函数值

            running_loss += loss.item()  # 累加损失函数值
            _, predicted = outputs.max(1)  # 获取预测结果中概率最高的类别
            total += labels.size(0)  
            correct += predicted.eq(labels).sum().item()  

    epoch_loss = running_loss / len(dataloader)  # 计算平均损失函数值
    accuracy = correct / total  # 计算准确率

    return epoch_loss, accuracy


# 设置迭代次数和记录损失和准确率的列表
num_epochs = 50
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

# 迭代训练和测试过程
for epoch in range(num_epochs):
    train_loss, train_accuracy = train(model, train_loader, criterion, optimizer)
    test_loss, test_accuracy = test(model, test_loader, criterion)

    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)

    # 打印训练和测试结果
    print(f'Epoch {epoch+1}/{num_epochs}: '
          f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, '
          f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.1f}')

# 画出loss和准确率曲线图
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train')
plt.plot(test_accuracies, label='Test')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
