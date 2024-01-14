import random
def split_dataset(data, train_ratio):
    random.shuffle(data)  # 随机打乱数据集

    total_samples = len(data)
    train_samples = int(total_samples * train_ratio)
    
    train_data = data[:train_samples+1]
    test_data = data[train_samples+1:]

    return train_data, test_data

# 读取训练集数据
with open('train.txt', 'r') as train_file:
    train_data = train_file.readlines()

# 读取测试集数据
with open('test.txt', 'r') as test_file:
    test_data = test_file.readlines()

# 合并数据集
all_data = train_data + test_data

# 分割数据集
train_ratio = 0.8  # 训练集所占比例
train_data, test_data = split_dataset(all_data, train_ratio)

# 将训练集和测试集分别保存到不同的文件中
with open('train_split.txt', 'w') as train_file:
    train_file.writelines(train_data)

with open('test_split.txt', 'w') as test_file:
    test_file.writelines(test_data)
