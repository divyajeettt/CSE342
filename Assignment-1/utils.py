import numpy as np


class PCA:
    def __init__(self, data: np.ndarray):
        self.data = data[:, np.std(data, axis=0) != 0]
        self.mean = np.mean(self.data, axis=0)
        self.std = np.std(self.data, axis=0)
        self.standardized = (self.data - self.mean) / self.std

    def eigen(self):
        covariance_matrix = np.cov(self.standardized.T)
        eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)
        eigen_pairs = [(eigen_value, eigen_vector) for eigen_value, eigen_vector in zip(eigen_values, eigen_vectors.T)]
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

    def fit(self, dim: int = 2):
        eigen_pairs = self.eigen()
        W = np.array([eigen_vector for (_, eigen_vector) in eigen_pairs[:dim]]).T
        self.transformed = self.standardized @ W
        return (self.transformed@W.T) * self.std + self.mean


class KNN:
    def __init__(self, k: int):
        self.k = k

    def fit(self, x_train: np.ndarray, y_train: np.ndarray):
        assert x_train.shape[0] == y_train.shape[0]
        self.x_train = x_train
        self.y_train = y_train

    def predict(self, x_sample: np.ndarray):
        k_nearest = np.argsort(np.linalg.norm(self.x_train - x_sample, axis=1))[:self.k]
        labels = self.y_train[k_nearest]
        return np.argmax(np.bincount(labels))

    def test(self, x_test: np.ndarray, y_test: np.ndarray):
        y_pred = np.zeros(y_test.shape)
        correct = 0
        for i, x_sample in enumerate(x_test, start=1):
            pred = self.predict(x_sample)
            y_pred[i - 1] = pred
            correct += (y_test[i - 1] == y_pred[i - 1])
            print("Test Sample:", str(i).zfill(4), f"Accuracy: {correct*100/i} %", end="\r")
        print()
        return y_pred

    def accuracy(self, y_test: np.ndarray, y_pred: np.ndarray):
        return np.mean(y_pred == y_test, axis=0)


class KMeans:
    def __init__(self, k: int, max_iters: int = 100, tol: float = 1e-5):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol

    def fit(self, data: np.ndarray):
        self.data = data
        self.n = self.data.shape[0]
        self.curr_centroids = np.random.rand(self.k, self.data.shape[1])
        self.prev_centroids = np.zeros(self.curr_centroids.shape)
        self.curr_labels = np.zeros(self.n)
        self.prev_labels = np.zeros(self.curr_labels.shape)

    def train(self):
        iters = 0
        while not self.termination(iters):
            for i in range(self.n):
                distances = np.linalg.norm(self.curr_centroids - self.data[i], axis=1)
                self.curr_labels[i] = np.argmin(distances)
            iters += 1
            self.update_params()
        return self.curr_centroids, self.curr_labels

    def termination(self, iters: int):
        return iters >= self.max_iters or np.allclose(self.prev_centroids, self.curr_centroids, atol=self.tol)

    def update_params(self):
        self.prev_centroids = self.curr_centroids.copy()
        self.update_centroids()
        self.prev_labels = self.curr_labels.copy()

    def update_centroids(self):
        for i in range(self.k):
            self.curr_centroids[i] = np.mean(self.data[self.curr_labels == i], axis=0)

    def clusters(self):
        self.data_clusters = [self.data[self.curr_labels == i] for i in range(self.k)]
        return self.data_clusters


class SilhouetteAnalysis:
    def fit(self, kmeans: KMeans):
        self.kmeans = kmeans
        self.data = self.kmeans.data
        self.n = self.kmeans.n
        self.labels = self.kmeans.curr_labels

    def silhouette_scores(self):
        s = np.zeros(self.n)
        for i in range(self.n):
            a = self.a(i)
            b = self.b(i)
            s[i] = (b - a) / max(a, b)
            print("Sample:", str(i+1).zfill(4), "Silhouette Score:", s[i], end="\r")
        print()
        self.sample_scores = s
        return self.sample_scores

    def a(self, i: int):
        return np.mean([np.linalg.norm(self.data[i] - sample) for sample in self.data[self.labels == self.labels[i]]])

    def b(self, i: int):
        return np.min([np.mean([np.linalg.norm(self.data[i] - sample) for sample in self.data[self.labels == j]]) for j in range(self.kmeans.k) if j != self.labels[i]])

    def run_analysis(self):
        self.silhouette_scores()
        print("AVERAGE SILHOUETTE SCORE:", np.mean(self.sample_scores))
        scores = [np.mean(self.sample_scores[self.labels == i]) for i in range(self.kmeans.k)]
        for i, score in enumerate(scores, start=1):
            print(f"Average Silhouette Score for Cluster {i}: {score}")


class FuzzyCMeans(KMeans):
    def __init__(self, c: int, m: float = 2, max_iters: int = 100, tol: float = 1e-5):
        super().__init__(c, max_iters, tol)
        self.c = c
        self.m = m

    def fit(self, data: np.ndarray):
        super().fit(data)
        self.curr_membership = np.random.rand(self.n, self.c)
        self.prev_membership = np.zeros(self.curr_membership.shape)

    def update_params(self):
        super().update_params()
        self.prev_membership = self.curr_membership.copy()
        self.update_membership()

    def update_membership(self):
        for i in range(self.n):
            dist = np.linalg.norm(self.data[i] - self.curr_centroids, axis=1)
            self.curr_membership[i, :] = 1 / np.sum((dist[:, None] / dist[None, :]) ** (2 / (self.m - 1)), axis=1)

    def update_centroids(self):
        for j in range(self.c):
            total = weighted = 0
            for i in range(self.n):
                total += self.curr_membership[i, j] ** self.m
                weighted += self.curr_membership[i, j] ** self.m * self.data[i]
            self.curr_centroids[j] = weighted / total

    def termination(self, iters: int):
        return iters >= self.max_iters or np.allclose(self.prev_membership, self.curr_membership, atol=self.tol)

    def objective(self):
        return np.sum([np.linalg.norm(self.data[i] - self.curr_centroids[j]) ** 2 * self.curr_membership[i, j] ** self.m for i in range(self.n) for j in range(self.c)])


class MeanShift:
    def __init__(self, bandwidth: float, tol: float = 1e-1):
        self.bandwidth = bandwidth
        self.tol = tol

    def fit(self, data: np.ndarray):
        self.data = data
        self.labels = np.zeros(self.data.shape[0])
        self.unlabeled = self.data.shape[0]
        self.centroids = {}

    def train(self):
        label = 1
        while self.unlabeled > 0:
            i = np.random.choice(np.where(self.labels == 0)[0])
            self.data[i] = self.shift(self.data[i])
            unlabeled = (self.labels == 0).astype(int)
            in_bandwidth = (np.linalg.norm(self.data - self.data[i], axis=1) < self.bandwidth).astype(int)
            self.labels[(unlabeled * in_bandwidth).astype(bool)] = label
            self.unlabeled -= np.sum(self.labels == label)
            key = tuple(np.round(self.data[i], 2))
            if key not in self.centroids:
                self.centroids[key] = self.data[i]
            label += 1
            print("PROGRESS:", (self.data.shape[0]-self.unlabeled)/self.data.shape[0] * 100, "%", end="\r")
        print()
        self.centroids = np.array([np.array(centroid) for centroid in self.centroids.values()])
        return self.data

    def shift(self, x: np.ndarray):
        prev_x = x
        while np.linalg.norm((curr_x := self.neighbourhood_mean(prev_x)) - prev_x) > self.tol:
            prev_x = curr_x
        return curr_x

    def neighbourhood_mean(self, x: np.ndarray):
        return np.mean(self.data[np.linalg.norm(self.data - x, axis=1) < self.bandwidth], axis=0)


class ICA:
    def __init__(self, n: int, tol: float = 1e-3):
        self.n = n
        self.tol = tol

    def fit(self, mixing_matrix: np.ndarray, s: np.ndarray):
        self.mixing_matrix = mixing_matrix
        self.s = s
        self.x = self.mixing_matrix @ self.s

    def center(self):
        self.x -= np.mean(self.x, axis=1, keepdims=True)

    def whiten(self):
        self.cov = np.cov(self.x)
        self.eigenvalues, self.eigenvectors = np.linalg.eig(self.cov)
        self.D = np.diag(1 / np.sqrt(self.eigenvalues))
        self.whitening_matrix = self.eigenvectors @ self.D @ self.eigenvectors.T
        self.whitened = self.whitening_matrix @ self.x

    def transform(self):
        self.center()
        self.whiten()
        self.w = np.random.randn(self.n, self.n)
        for i in range(self.n):
            wni = self.wni(i)
            while not np.isclose(self.w[i].T @ wni, 1.0, atol=self.tol):
                if i > 0:
                    wni -= np.sum([wni.T @ self.w[j, :] * self.w[j, :] for j in range(i)], axis=0)
                self.w[i, :] = wni / np.linalg.norm(wni)
                wni = self.wni(i)
        return self.w

    def wni(self, i: int):
        A = np.mean([np.tanh(self.w[i] @ xk) * xk for xk in self.whitened.T], axis=0)
        B = np.mean([1 - np.tanh(self.w[i] @ xk)**2 for xk in self.whitened.T], axis=0)
        w = A - self.w[i]*B
        return w / np.linalg.norm(w)