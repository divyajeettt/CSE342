import numpy as np


class PCA:
    """
    Implements the Principal Component Analysis algorithm for dimensionality reduction.
    Each PCA object is initialized with a dataset.
    The same PCA object can be used to reduce the dimensionality of the data to any number of dimensions.
    :attrs:
        data: The original dataset.
        mean: The mean of the dataset.
        std: The standard deviation of the dataset.
        standardized: The standardized dataset.
        transformed: The transformed (and standardized) dataset.
    """

    def __init__(self, data: np.ndarray):
        self.data = data[:, np.std(data, axis=0) != 0]
        self.mean = np.mean(self.data, axis=0)
        self.std = np.std(self.data, axis=0)
        self.standardized = (self.data - self.mean) / self.std

    def eigen(self):
        """
        Returns the eigenvalues and eigenvectors of the covariance matrix of the standardized dataset
        in descending order of absolute eigenvalues.
        """
        covariance_matrix = np.cov(self.standardized.T)
        eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)
        eigen_pairs = [(eigen_value, eigen_vector) for eigen_value, eigen_vector in zip(eigen_values, eigen_vectors.T)]
        return sorted(eigen_pairs, key=(lambda pair: pair[0]), reverse=True)

    def explained_variance(self):
        """
        Returns the explained variance of each Principal Component.
        """
        eigen_pairs = self.eigen()
        eigen_total = sum(eigen_value for (eigen_value, _) in eigen_pairs)
        return np.array([eigen_value / eigen_total for (eigen_value, _) in eigen_pairs])

    def explained_variance_ratio(self):
        """
        Returns the explained variance ratio of each Principal Component.
        """
        explained_variance = self.explained_variance()
        return explained_variance / sum(explained_variance)

    def cumulative_explained_variance(self):
        """
        Returns the cumulative explained variance of the Principal Components
        in decreasing order of their absolute eigenvalues.
        """
        return np.cumsum(self.explained_variance())

    def fit(self, dim: int = 2):
        """
        Returns the transformed dataset.
        :param dim: The number of dimensions to reduce the dataset to.
        """
        assert dim <= self.data.shape[1]
        eigen_pairs = self.eigen()
        W = np.array([eigen_vector for (_, eigen_vector) in eigen_pairs[:dim]]).T
        self.transformed = self.standardized @ W
        return (self.transformed@W.T) * self.std + self.mean


class KNN:
    """
    Implements the K-Nearest Neighbors algorithm for classification.
    Each KNN object is initialized with a value of k.
    The same KNN object can be fitted with different training data and used to predict the labels of test data.
    :attrs:
        k: The value of k.
        x_train: The training data.
        y_train: The training labels.
    """

    def __init__(self, k: int):
        assert k >= 1
        self.k = k

    def fit(self, x_train: np.ndarray, y_train: np.ndarray):
        """
        Fits the KNN object with the training data.
        :param x_train: The training data.
        :param y_train: The training labels.
        """
        assert x_train.shape[0] == y_train.shape[0]
        self.x_train = x_train
        self.y_train = y_train

    def predict(self, x_sample: np.ndarray):
        """
        Predicts the label of a single test sample and return it.
        :param x_sample: The test sample.
        """
        k_nearest = np.argsort(np.linalg.norm(self.x_train - x_sample, axis=1))[:self.k]
        labels = self.y_train[k_nearest]
        return np.argmax(np.bincount(labels))

    def test(self, x_test: np.ndarray, y_test: np.ndarray):
        """
        Runs predictions on the test data and returns the predicted labels.
        Returns the predicted labels.
        :param x_test: The test data.
        :param y_test: The test labels.
        """
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
        """
        Returns the accuracy score (between 0 and 1) of the predictions.
        :param y_test: The test labels.
        :param y_pred: The predicted labels.
        """
        return np.mean(y_pred == y_test, axis=0)


class KMeans:
    """
    Implements the K-Means algorithm for clustering.
    Each KMeans object can be trained to cluster the given points.
    :attrs:
        k: The value of k.
        max_iters: The maximum number of iterations in which the clustering can be done.
        tol: Tolerance level for convergence.
        data: The fitted data for clustering.
        n: Number of samples in data.
        curr_centroids: The array of centroids of clustering.
        curr_labels: The array of labels of all data points.
        data_clusters: The array containing the clustered data.
    """

    def __init__(self, k: int, max_iters: int = 100, tol: float = 1e-5):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol

    def fit(self, data: np.ndarray):
        """
        Fits the data onto the model for clustering.
        :param data: The input data.
        """
        self.data = data
        self.n = self.data.shape[0]
        self.curr_centroids = np.random.rand(self.k, self.data.shape[1])
        self.prev_centroids = np.zeros(self.curr_centroids.shape)
        self.curr_labels = np.zeros(self.n)
        self.prev_labels = np.zeros(self.curr_labels.shape)

    def train(self):
        """
        Runs the clustering algorithm and returns the centroids and labels of the fitted data.
        """
        iters = 0
        while not self.termination(iters):
            for i in range(self.n):
                distances = np.linalg.norm(self.curr_centroids - self.data[i], axis=1)
                self.curr_labels[i] = np.argmin(distances)
            iters += 1
            self.update_params()
        return self.curr_centroids, self.curr_labels

    def termination(self, iters: int):
        """
        The termination condition for the K-Means Algorithm.
        :param iters: The current number of iteration.
        """
        return iters >= self.max_iters or np.allclose(self.prev_centroids, self.curr_centroids, atol=self.tol)

    def update_params(self):
        """
        Modularly updates the centroids and labels of the data during iteration.
        """
        self.prev_centroids = self.curr_centroids.copy()
        self.update_centroids()
        self.prev_labels = self.curr_labels.copy()

    def update_centroids(self):
        """
        Updates the centroids during iteration.
        """
        for i in range(self.k):
            self.curr_centroids[i] = np.mean(self.data[self.curr_labels == i], axis=0)

    def clusters(self):
        """
        Assuming that labels have been generated, returns the clustered data.
        """
        self.data_clusters = [self.data[self.curr_labels == i] for i in range(self.k)]
        return self.data_clusters


class SilhouetteAnalysis:
    """
    Implements the Silhouette Analysis algorithm to evaluate clustering algorithms.
    Each SilhouetteAnalysis object is fitted with a (currently) KMeans object it must evaluate.
    The same SilhouetteAnalysis object can be reused to fit another KMeans model.
    :attrs:
        kmeans: The KMeans object it must evaluate.
        data: The input data used by the clustering algorithm.
        n: The number of samples in the input data.
        labels: The labels returned by clustering algorithm.
        sample_scores: The Silhouette Scores of the currently fitted clustering
    """

    def fit(self, kmeans: KMeans):
        """
        Fits the given KMeans object onto the model.
        :param kmeans: The KMeans object to evaluate.
        """
        self.kmeans = kmeans
        self.data = self.kmeans.data
        self.n = self.kmeans.n
        self.labels = self.kmeans.curr_labels

    def silhouette_scores(self):
        """
        Calculates sample-wise silhouette scores and returns them.
        """
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
        """
        a(i) is the average distance between a point and all the points in the same cluster.
        The smaller the a(i) value, the better and tighter the clustering.
        :param i: Index of the current data sample.
        """
        return np.mean([np.linalg.norm(self.data[i] - sample) for sample in self.data[self.labels == self.labels[i]]])

    def b(self, i: int):
        """
        b(i) is the average distance between a point and all the points not in the same cluster.
        The larger the b(i) value, the worse and overlapping the clustering.
        :param i: Index of the current data sample.
        """
        return np.min([np.mean([np.linalg.norm(self.data[i] - sample) for sample in self.data[self.labels == j]]) for j in range(self.kmeans.k) if j != self.labels[i]])

    def run_analysis(self):
        """
        Runs the analysis and displays the average silhouette scores.
        """
        self.silhouette_scores()
        print("AVERAGE SILHOUETTE SCORE:", np.mean(self.sample_scores))
        scores = [np.mean(self.sample_scores[self.labels == i]) for i in range(self.kmeans.k)]
        for i, score in enumerate(scores, start=1):
            print(f"Average Silhouette Score for Cluster {i}: {score}")


class FuzzyCMeans(KMeans):
    """
    Implements the Fuzzy C-Means alogorithm for clustering.
    FuzzyCMeans is a child of KMeans due to the overlapping functionality.
    Additional attributes:
        c: c = k, the number of clusters.
        m: The fuzziness index.
        curr_membership: The membership matrix of clustering.
    """

    def __init__(self, c: int, m: float = 2, max_iters: int = 100, tol: float = 1e-5):
        super().__init__(c, max_iters, tol)
        self.c = c
        self.m = m

    def fit(self, data: np.ndarray):
        """
        Fits the data onto the model for clustering.
        :param data: The input data.
        """
        super().fit(data)
        self.curr_membership = np.random.rand(self.n, self.c)
        self.prev_membership = np.zeros(self.curr_membership.shape)

    def update_params(self):
        """
        Modularly updates the centroids, labels, and memberships of the data during iteration.
        """
        super().update_params()
        self.prev_membership = self.curr_membership.copy()
        self.update_membership()

    def update_membership(self):
        """
        Updates the membership for each sample in the data during iteration.
        """
        for i in range(self.n):
            dist = np.linalg.norm(self.data[i] - self.curr_centroids, axis=1)
            self.curr_membership[i, :] = 1 / np.sum((dist[:, None] / dist[None, :]) ** (2 / (self.m - 1)), axis=1)

    def update_centroids(self):
        """
        Updates the centroids during iteration.
        """
        for j in range(self.c):
            total = weighted = 0
            for i in range(self.n):
                total += self.curr_membership[i, j] ** self.m
                weighted += self.curr_membership[i, j] ** self.m * self.data[i]
            self.curr_centroids[j] = weighted / total

    def termination(self, iters: int):
        """
        The termination condition for the Fuzzy C-Means Algorithm.
        :param iters: The current number of iteration.
        """
        return iters >= self.max_iters or np.allclose(self.prev_membership, self.curr_membership, atol=self.tol)

    def objective(self):
        """
        Returns the value of the objective function J(U, V) associated with the Fuzzy C-Means algorithm.
        """
        return np.sum([np.linalg.norm(self.data[i] - self.curr_centroids[j]) ** 2 * self.curr_membership[i, j] ** self.m for i in range(self.n) for j in range(self.c)])


class MeanShift:
    """
    Implements the Mean Shift algorithm for image clustering.
    Each MeanShift object has a bandwidth and a tolerance, and can be reused on different images.
    The best/tried input format is array of [R G B x y] pixels.
    :attrs:
        bandwidth: The neighborhood of a point is decided using the bandwidth.
        tol: Tolerance level for convergence.
        data: The fitted data for clustering.
        labels: The array of generated labels.
        unlabeled: The count of currently unlabeled data samples.
        centroids: The array of generated centroids of the clusters.
    """

    def __init__(self, bandwidth: float, tol: float = 1e-1):
        self.bandwidth = bandwidth
        self.tol = tol

    def fit(self, data: np.ndarray):
        """
        Fits the data onto the model for clustering.
        :param data: The input data.
        """
        self.data = data
        self.labels = np.zeros(self.data.shape[0])
        self.unlabeled = self.data.shape[0]
        self.centroids = {}

    def train(self):
        """
        Runs the clustering algorithm and returns the mean-shifted data.
        """
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
        """
        Iteratively shifts a data point to its correct position by moving it towards the region of highest
        density in its neighborhood.
        :param x: The data sample to shift.
        """
        prev_x = x
        while np.linalg.norm((curr_x := self.neighborhood_mean(prev_x)) - prev_x) > self.tol:
            prev_x = curr_x
        return curr_x

    def neighborhood_mean(self, x: np.ndarray):
        """
        Returns the mean of the points in the neighborhood of x (points within a distance of bandwidth).
        :param x: The data sample.
        """
        return np.mean(self.data[np.linalg.norm(self.data - x, axis=1) < self.bandwidth], axis=0)


class ICA:
    """
    Implements the Independent Component Analysis algorithm for unmixing independent linearly mixed non-gaussian signals.
    A single ICA object can separate out only a fixed number of signals.
    The ICA object generates a suitable unmixing matrix that undoes the mixing.
    :attrs:
        n: The number of linearly-mixed signals.
        tol: Tolerance level for convergence.
        mixing_matrix: The matrix that mixed the signals (Ideally, this should be hidden from the ICA).
        s: A vector of the signals to mix (Ideally, the ICA should only receive the mixed signals).
        x: The vector contaning the mixed signals.
        whitening_matrix: The matrix that performs the whitening.
        whitened: A whitened copy of the input signals.
        w: The unmixing matrix.
    """

    def __init__(self, n: int, tol: float = 1e-3):
        self.n = n
        self.tol = tol

    def fit(self, mixing_matrix: np.ndarray, s: np.ndarray):
        """
        Creates the mixed signals with the given mixing matrix and array of signals
        """
        self.mixing_matrix = mixing_matrix
        self.s = s
        self.x = self.mixing_matrix @ self.s

    def center(self):
        """
        Centers the data about the mean in place.
        """
        self.x -= np.mean(self.x, axis=1, keepdims=True)

    def whiten(self):
        """
        Whitens the data, i.e. de-correlates it to make all covarainces 0 and individual varainces 1.
        """
        cov = np.cov(self.x)
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        D = np.diag(1 / np.sqrt(eigenvalues))
        self.whitening_matrix = eigenvectors @ D @ eigenvectors.T
        self.whitened = self.whitening_matrix @ self.x

    def transform(self):
        """
        Generates the unmixing matrix by independent component analysis.
        """
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
        """
        Generates the vector wni to update the current column in the unmixing matrix.
        :param i: The number of column.
        """
        A = np.mean([np.tanh(self.w[i] @ xk) * xk for xk in self.whitened.T], axis=0)
        B = np.mean([1 - np.tanh(self.w[i] @ xk)**2 for xk in self.whitened.T], axis=0)
        w = A - self.w[i]*B
        return w / np.linalg.norm(w)