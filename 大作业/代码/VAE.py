import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.optim import lr_scheduler
from matplotlib import pyplot as plt

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 下载MNIST数据集
mnist_train = datasets.MNIST('mnist', train=True, transform=transforms.ToTensor(), download=True)
mnist_train = DataLoader(mnist_train, batch_size=64, shuffle=True, num_workers=4)

mnist_test = datasets.MNIST('mnist', train=False, transform=transforms.ToTensor(), download=True)
mnist_test = DataLoader(mnist_test, batch_size=64, num_workers=4)

# 变分自编码器的网络结构
class VariationalAutoencoder(nn.Module):
    def __init__(self):
        super(VariationalAutoencoder, self).__init__()
        # 编码器的网络结构
        self.encoder = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 20),
            nn.ReLU()
        )
        # 解码器的网络结构
        self.decoder = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Sigmoid()
        )
        # 均值和方差的线性层
        self.fc_mu = nn.Linear(20, 10)
        self.fc_logvar = nn.Linear(20, 10)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x):
        batchsz = x.size(0)
        x = x.view(batchsz, -1)
        x = self.encoder(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        z = self.reparameterize(mu, logvar)
        features = z.clone()
        x = self.decoder(z)
        x = x.view(batchsz, 1, 28, 28)
        return x, mu, logvar, features

def loss_function(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x.view(-1, 1, 28, 28), reduction='sum')

    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def test(model, dataloader):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            reconstructions, mu, logvar, _ = model(images)
            loss = loss_function(reconstructions, images, mu, logvar)
            running_loss += loss.item()

    average_loss = running_loss / len(dataloader.dataset)
    return average_loss

def train(epochs, model, optimizer, scheduler, train_loader):
    model.train()  # 将模型设置为训练模式
    loss_set = []

    for epoch in range(epochs):
        scheduler.step()  # 调整学习率
        total_loss = 0.0  # 用于累计每个epoch的损失值

        for i, (img, label) in enumerate(train_loader):
            img = img.to(device)  # 将输入数据移动到GPU
            img = img.view(-1, 784).to(device)  # 将输入数据展平为二维张量
            label = label.to(device)  # 将标签数据移动到GPU

            optimizer.zero_grad()  # 梯度归零，清除之前的梯度信息
            recon_batch, mu, logvar,_ = model(img)  # 通过模型进行前向传播，得到输出和潜在变量
            loss = loss_function(recon_batch, img, mu, logvar)  # 计算损失函数值
            loss.backward()  # 反向传播，计算梯度
            optimizer.step()  # 更新模型参数

            total_loss += loss.item()  # 累加损失函数值

            # 打印当前的训练信息
            print("Epoch: {}/{}, Step: {}, Loss: {:.4f}".format(epoch + 1, epochs, i + 1, loss.item()))
            loss_set.append(loss.item())

    return loss_set


def Image_all(model, test_loader):
    model.eval()
    N = 8
    M = 8
    with torch.no_grad():
        images, _ = next(iter(test_loader))
        images = images.to(device)
        _images, _, _, _ = model(images)

    p1 = plt.figure(1)
    for i in range(N * M):
        plt.subplot(N, M, i + 1)
        plt.imshow(images[i].cpu().numpy().squeeze(), cmap='gray_r')
        plt.xticks([])
        plt.yticks([])

    p2 = plt.figure(2)
    for i in range(N * M):
        plt.subplot(N, M, i + 1)
        plt.imshow(_images[i].cpu().numpy().squeeze(), cmap='gray_r')
        plt.xticks([])
        plt.yticks([])

    plt.show()

def Image_sep(model, num):
    model.eval()
    N = 2
    M = 5
    num_images = 10
    cnt = 0

    p1 = plt.figure(1)
    p2 = plt.figure(2)

    dataiter = iter(mnist_test)
    images, labels = next(dataiter)

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
                    reconstructed_image, _, _, _ = model(image)

                plt.figure(2)
                plt.subplot(N, M, cnt + 1)
                plt.imshow(reconstructed_image.squeeze().cpu().numpy(), cmap='gray_r')
                plt.xticks([])
                plt.yticks([])

                cnt += 1
                if cnt == num_images:
                    break

        try:
            images, labels = next(dataiter)
        except StopIteration:
            break

    plt.show()

if __name__ == '__main__':
    model = VariationalAutoencoder().to(device)
    epochs_num = 2
    num_epochs = 50
    test_loss_set = []
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.1)
    loss_set = train(epochs_num, model, optimizer, scheduler, mnist_train)
    index_train = [i for i in range(len(loss_set))]
    fig1 = plt.figure(1)
    plt.plot(index_train, loss_set)
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.show()

    Image_all(model, mnist_test)

    for _num in range(0, 10):
        Image_sep(model, _num)

    for epoch in range(num_epochs):
        test_loss = test(model, mnist_test)
        test_loss_set.append(test_loss)

    index_test = [i for i in range(len(test_loss_set))]
    fig2 = plt.figure(2)
    plt.plot(index_test, test_loss_set)
    plt.xlabel("Epochs")
    plt.ylabel("Test Loss")
    plt.show()
