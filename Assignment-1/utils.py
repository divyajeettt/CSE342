import numpy as np


class PCA:
    def __init__(self, data: np.ndarray):
        self.data = data
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)
        self.standardized = (self.data - self.mean) / self.std

    def covariance(self):
        return np.cov(self.standardized.T)

    def eigen(self):
        covariance_matrix = self.covariance()
        eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)
        eigen_pairs = [(abs(eigen_values[i]), eigen_vectors[:, i]) for i in range(len(eigen_values))]
        return sorted(eigen_pairs, key=(lambda pair: pair[0]), reverse=True)

    def explained_variance(self):
        eigen_pairs = self.eigen()
        eigen_total = sum(eigen_value for (eigen_value, _) in eigen_pairs)
        return np.array([eigen_value / eigen_total for (eigen_value, _) in eigen_pairs])

    def explained_variance_ratio(self):
        explained_variance = self.explained_variance()
        return explained_variance / sum(explained_variance)

    def cumulative_explained_variance(self):
        return np.cumsum(self.explained_variance())

    def fit(self, X: np.ndarray, dim: int = 2):
        eigen_pairs = self.eigen()
        W = np.array([eigen_vector for (_, eigen_vector) in eigen_pairs[:dim]]).T
        return (self.standardized @ W)


class KNN:
    def __init__(self, k: int):
        self.k = k

    def fit(self, x_train: np.ndarray, y_train: np.ndarray):
        self.x_train = x_train
        self.y_train = y_train

    def predict(self, x_test: np.ndarray):
        y_pred = np.zeros(x_test.shape[0])
        for i in range(x_test.shape[0]):
            x = x_test[i]
            distances = np.linalg.norm(self.x_train - x, axis=1)
            k_nearest = np.argsort(distances)[:self.k]
            labels = self.y_train[k_nearest]
            y_pred[i] = np.argmax(np.bincount(labels.astype("int")))
        return y_pred


class KMeans:
    def __init__(self, k: int, max_iters: int = 100, tol: float = 1e-5):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol

    def fit(self, x_train: np.ndarray):
        self.x_train = x_train
        self.curr_centroids = np.random.rand(self.k, self.x_train.shape[1])
        self.prev_centroids = np.zeros(self.curr_centroids.shape)
        self.curr_labels = np.zeros(self.x_train.shape[0])
        self.prev_labels = np.zeros(self.curr_labels.shape)

    def train(self):
        iters = 0
        while iters < self.max_iters or not np.allclose(self.prev_centroids, self.curr_centroids, atol=self.tol):
            for i in range(self.x_train.shape[0]):
                distances = np.linalg.norm(self.curr_centroids - self.x_train[i], axis=1)
                self.curr_labels[i] = np.argmin(distances)
            iters += 1
            self.prev_centroids = self.curr_centroids
            self.update_centroids()
            self.prev_labels = self.curr_labels
        return self.curr_centroids, self.curr_labels

    def update_centroids(self):
        for i in range(self.k):
            self.curr_centroids[i] = np.mean(self.x_train[self.curr_labels == i], axis=0)

    def clusters(self):
        self.data_clusters = [self.x_train[self.curr_labels == i] for i in range(self.k)]
        return self.data_clusters


class SilhouetteAnalysis:
    def fit(self, kmeans: KMeans):
        self.kmeans = kmeans
        self.x_train = self.kmeans.x_train
        self.labels = self.kmeans.curr_labels

    def silhouette_scores(self):
        s = np.zeros(self.x_train.shape[0])
        for i in range(self.x_train.shape[0]):
            a = self.a(i)
            b = self.b(i)
            s[i] = (b - a) / max(a, b)
        self.sample_scores = s
        self.mean_score = np.mean(s)
        return self.sample_scores

    def a(self, i: int):
        return np.mean([np.linalg.norm(self.x_train[i] - sample) for sample in self.x_train[self.labels == self.labels[i]]])

    def b(self, i: int):
        return np.min([np.mean([np.linalg.norm(self.x_train[i] - sample) for sample in self.x_train[self.labels == j]]) for j in range(self.kmeans.k) if j != self.labels[i]])

    def clusterwise_scores(self):
        self.silhouette_scores()
        self.cluster_scores = np.array([np.mean(self.sample_scores[self.labels == i]) for i in range(self.kmeans.k)])
        return self.cluster_scores