import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import gzip
import numpy as np

def load_images(filename):
    with gzip.open(filename, 'rb') as f:
        f.read(16)   # 跳过文件头部信息
        buf = f.read()   # 读取图像数据
        data = np.frombuffer(buf, dtype=np.uint8)  # 转换为 NumPy 数组
        data = data.reshape(-1, 28*28)             # 重塑为二维数组
    return data   

def load_labels(filename):
    with gzip.open(filename, 'rb') as f:
        f.read(8)     # 跳过文件头部信息
        buf = f.read() # 读取标签数据
        data = np.frombuffer(buf, dtype=np.uint8)   # 转换为 NumPy 数组
    return data

# 加载MNIST数据集
x_train = load_images('train-images-idx3-ubyte.gz')
y_train = load_labels('train-labels-idx1-ubyte.gz')
x_test = load_images('t10k-images-idx3-ubyte.gz')
y_test = load_labels('t10k-labels-idx1-ubyte.gz')

# 数据预处理
x_train = x_train / 255.0  # 归一化像素值到 0-1 范围
x_test = x_test / 255.0    # 归一化像素值到 0-1 范围
y_train = tf.keras.utils.to_categorical(y_train)    #对标签进行 one-hot 编码
y_test = tf.keras.utils.to_categorical(y_test)      #对标签进行 one-hot 编码

# 定义模型
model = Sequential([
    Dense(256, activation='sigmoid', input_shape=(784,)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# 编译模型(使用多分类交叉熵损失categorical_crossentropy)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 训练模型（指定每个批次的大小为 128，迭代训练 10 个周期（epochs），将测试集划分为训练集的 10% 用于验证模型性能）
history = model.fit(x_train, y_train, batch_size=128, epochs=10, validation_split=0.1)

# 在测试集上评估模型
loss, accuracy = model.evaluate(x_test, y_test)
print("测试集上的准确率：", accuracy)

# 输出每次预测和正确结果的对比
predictions = model.predict(x_test)
predicted_labels = tf.argmax(predictions, axis=1)
true_labels = tf.argmax(y_test, axis=1)

for i in range(len(x_test)):
    print("预测值：", predicted_labels[i].numpy(), "真实值：", true_labels[i].numpy())

# 可视化损失函数
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('the change of loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['train', 'test'])
plt.show()

# 可视化学习率对准确率的影响
learning_rates = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01]
accuracies = []

for lr in learning_rates:
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=128, epochs=5, verbose=0)
    _, accuracy = model.evaluate(x_test, y_test)
    accuracies.append(accuracy)

plt.plot(learning_rates, accuracies, 'o-')
plt.title('The Influence of Learning rate on Accuracy')
plt.xlabel('Learning Rate')
plt.ylabel('Accuracy')
plt.show()
