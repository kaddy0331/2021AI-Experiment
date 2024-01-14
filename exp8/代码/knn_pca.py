import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

# 读取文件并预处理数据
def read_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip().split(' ')
            label = int(line[1])  # 情感编号
            text = ' '.join(line[3:])  # 句子内容
            data.append((label, text))
    return data

# 计算TF-IDF特征并应用PCA进行降维
def calculate_tfidf_features(train_data, test_data, k):
    train_texts = [text for _, text in train_data]
    test_texts = [text for _, text in test_data]
    
    # 使用TfidfVectorizer计算TF-IDF特征
    vectorizer = TfidfVectorizer()
    train_features = vectorizer.fit_transform(train_texts)
    test_features = vectorizer.transform(test_texts)
    
    # 使用PCA进行降维
    pca = PCA(n_components=k)  # 设置要保留的主成分个数
    train_features_pca = pca.fit_transform(train_features.toarray())
    test_features_pca = pca.transform(test_features.toarray())
    
    return train_features_pca, test_features_pca

# KNN
def knn(train_features_normalized, train_labels, test_features_normalized, k):
    train_features_normalized = train_features_normalized
    test_features_normalized = test_features_normalized

    predicted_labels = []
    
    for test_sample in test_features_normalized:
        distances = []
        
        for i, train_sample in enumerate(train_features_normalized):
            # 计算欧氏距离
            distance = np.linalg.norm(test_sample - train_sample)
            distances.append((distance, train_labels[i]))
            
            # 计算曼哈顿距离
            #distance = np.linalg.norm(test_sample - train_sample,ord=1)
            #distances.append((distance, train_labels[i]))

            # 计算切比雪夫距离
            #distance = np.max(np.abs(test_sample - train_sample))
            #distances.append((distance, train_labels[i]))

        # 根据距离进行排序
        distances.sort(key=lambda x: x[0])
        
        # 选择前 k 个距离最近的样本
        k_nearest = distances[:k]
        
        # 统计最近样本中出现最多的标签
        labels = [label for _, label in k_nearest]
        predicted_label = max(set(labels), key=labels.count)
        predicted_labels.append(predicted_label)
    
    return predicted_labels


# 计算准确率
def calculate_accuracy(predicted_labels, true_labels):
    correct_count = sum(1 for predicted, true in zip(predicted_labels, true_labels) if predicted == true)
    total_count = len(predicted_labels)
    accuracy = correct_count / total_count
    return accuracy

# 读取训练集和测试集数据
train_data = read_data('train_split1.txt')
test_data = read_data('test_split1.txt')

# 设置PCA降维的主成分个数
m = 100

# 计算TF-IDF特征并应用PCA进行降维
train_features, test_features = calculate_tfidf_features(train_data, test_data, m)

# 获取训练集和测试集的标签
train_labels = [label for label, _ in train_data]
test_labels = [label for label, _ in test_data]

# 归一化处理
train_features_normalized = normalize(train_features)
test_features_normalized = normalize(test_features)

k=20

# 使用KNN算法
predicted_labels = knn(train_features_normalized, train_labels, test_features_normalized, k)

# 输出每次的预测情感编号和正确情感编号
for predicted, true in zip(predicted_labels, test_labels):
    print("预测情感编号: {}, 正确情感编号: {}".format(predicted, true))

# 计算准确率
accuracy = calculate_accuracy(predicted_labels, test_labels)
print("准确率: {:.2%}".format(accuracy))
