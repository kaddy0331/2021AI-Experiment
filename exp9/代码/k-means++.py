import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import calinski_harabasz_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def kmeans_plusplus(X, k, max_iterations=100):
    # 使用 K-means++ 初始化点
    centroids = [X[np.random.choice(X.shape[0])]]
    while len(centroids) < k:
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=-1)
        min_distances = np.min(distances, axis=1)
        probabilities = min_distances / np.sum(min_distances)
        next_centroid = X[np.random.choice(X.shape[0], p=probabilities)]
        centroids.append(next_centroid)

    prev_centroids = np.zeros_like(centroids)
    labels = np.zeros(X.shape[0])

    for _ in range(max_iterations):
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=-1)
        labels = np.argmin(distances, axis=1)

        for i in range(k):
            centroids[i] = np.mean(X[labels == i], axis=0)

        if np.allclose(centroids, prev_centroids):
            break

        prev_centroids = centroids.copy()

    return labels, centroids


# 读取文本数据
data = []
with open('train.txt', 'r', encoding='utf-8') as file:
    for line in file:
        data.append(line.strip().split(' ', 3)[-1])  # 仅保留文本内容部分

# 使用CountVectorizer提取文本特征
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data).toarray()

# 进行K-means++聚类
k = 6  # 聚类的簇数
labels, centroids = kmeans_plusplus(X, k)

# 计算 calinski_harabasz_score
score = calinski_harabasz_score(X, labels)

# 打印聚类中心
print("聚类中心:")
for i, center in enumerate(centroids):
    print("Cluster", i+1, ":", vectorizer.inverse_transform(center.reshape(1, -1)))

# 使用TSNE进行降维和可视化
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

# 绘制聚类结果的散点图
plt.figure(figsize=(10, 6))
colors = ['b', 'g', 'r', 'c', 'm', 'y']  # 聚类簇的颜色
for i in range(k):
    cluster_points = X_tsne[labels == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=colors[i], label="Cluster "+str(i+1))
plt.legend()
plt.title("Clustering Visualization (K=6)")
plt.show()

# 打印 calinski_harabasz_score
print("Calinski-Harabasz Score:", score)
