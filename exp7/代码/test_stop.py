import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

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

# 读取停用词表
def read_stopwords(file_path):
    stopwords = []
    with open(file_path, 'r',encoding="utf-8") as file:
        for line in file:
            word = line.strip()
            stopwords.append(word)
    return stopwords

# 计算TF-IDF特征
def calculate_tfidf_features(train_data, test_data, stopwords):
    train_texts = [text for _, text in train_data]
    test_texts = [text for _, text in test_data]
    
    # 使用TfidfVectorizer计算TF-IDF特征
    vectorizer = TfidfVectorizer(stop_words=stopwords)
    train_features = vectorizer.fit_transform(train_texts)
    test_features = vectorizer.transform(test_texts)
    
    return train_features, test_features

# 实现朴素贝叶斯分类器（包括拉普拉斯平滑）
def naive_bayes(train_features, train_labels, test_features):
    num_classes = len(set(train_labels))
    num_train_samples, num_features = train_features.shape

    # 计算每个类别的先验概率
    priors = [0] * num_classes
    for label in train_labels:
        priors[label - 1] += 1
    priors = [count / num_train_samples for count in priors]

    # 计算每个特征在每个类别下的条件概率
    likelihoods = [[1] * num_features for _ in range(num_classes)]
    total_counts = [2] * num_classes  # 拉普拉斯平滑中的分母项，初始值为2
    for i in range(num_train_samples):
        label = train_labels[i] - 1
        feature_vector = train_features[i]
        total_counts[label] += feature_vector.sum()
        likelihoods[label] += feature_vector.toarray()[0]

    # 计算每个特征在每个类别下的条件概率（取对数）
    for i in range(num_classes):
        total_count = total_counts[i]
        likelihoods[i] = [math.log((count + 1) / (total_count + num_features)) for count in likelihoods[i]]

    # 对测试样本进行分类
    test_labels = []
    for i in range(test_features.shape[0]):
        test_vector = test_features[i]
        scores = [math.log(prior) + sum(likelihood * feature for likelihood, feature in zip(likelihoods[j], test_vector.toarray()[0]))
                  for j, prior in enumerate(priors)]
        predicted_label = scores.index(max(scores)) + 1
        test_labels.append(predicted_label)

    return test_labels

# 计算准确率
def calculate_accuracy(predicted_labels, true_labels):
    correct_count = sum(1 for predicted, true in zip(predicted_labels, true_labels) if predicted == true)
    total_count = len(predicted_labels)
    accuracy = correct_count / total_count
    return accuracy

# 读取训练集和测试集数据
train_data = read_data('train.txt')
test_data = read_data('test.txt')

# 读取停用词表
stopwords = read_stopwords('ban.txt')

# 计算TF-IDF特征
train_features, test_features = calculate_tfidf_features(train_data, test_data, stopwords)

# 获取训练集和测试集的标签
train_labels = [label for label, _ in train_data]
test_labels = [label for label, _ in test_data]

# 归一化处理
train_features_normalized = normalize(train_features)
test_features_normalized = normalize(test_features)

# 使用朴素贝叶斯分类器进行分类
predicted_labels = naive_bayes(train_features_normalized, train_labels, test_features_normalized)

# 输出每次的预测情感编号和正确情感编号
for predicted, true in zip(predicted_labels, test_labels):
    print("预测情感编号: {}, 正确情感编号: {}".format(predicted, true))

# 计算准确率
accuracy = calculate_accuracy(predicted_labels, test_labels)
print("准确率: {:.2%}".format(accuracy))
