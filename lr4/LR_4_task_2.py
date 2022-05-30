import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

iris = load_iris()

X = iris['data']
y = iris['target']

num_clusters = 3

kmeans = KMeans(n_clusters = num_clusters)
kmeans.fit(X)

y_pred = kmeans.predict(X)

centers = kmeans.cluster_centers_

for i in range(X.shape[1] - 1):
    for j in range(i + 1, X.shape[1]):
        plt.scatter(X[:, i], X[:, j], c = y_pred, s = 50, cmap = 'viridis')
        plt.scatter(centers[:, i], centers[:, j], c = 'red', s = 150)
        plt.figure()
        plt.show()