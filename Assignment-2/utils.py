import numpy as np


class LinearRegression:
    """
    Implements the Logistic Regression algorithm for binary classification.
    :attrs:
        x_train: The dataset to be used for the algorithm.
        y_train: The labels of the dataset.
    """

    def __init__(self, x_train: np.ndarray, y_train: np.ndarray):
        self.x_train = x_train
        self.y_train = y_train

    def train(self):
        """
        Trains the model. Uses the Normal Equation.
        """
        self.weights = np.linalg.inv(self.x_train.T @ self.x_train) @ self.x_train.T @ self.y_train

    def predict(self, x: np.ndarray):
        """
        Returns the prediction of a point/data.
        :param x: The point/data to be predicted.
        """
        return x @ self.weights

    def rss(self, x: np.ndarray, y: np.ndarray):
        """
        Returns the residual sum of squares.
        :param x: The test dataset.
        :param y: The test labels.
        """
        return np.sum((y - self.predict(x)) ** 2)

    def tss(self, y: np.ndarray):
        """
        Returns the total sum of squares.
        :param y: The test labels.
        """
        return np.sum((y - y.mean()) ** 2)

    def r2_score(self, x: np.ndarray, y: np.ndarray):
        """
        Returns the R2 score.
        :param x: The test dataset.
        :param y: The test labels.
        """
        return 1 - self.rss(x, y) / self.tss(y)

    def rmse(self, x: np.ndarray, y: np.ndarray):
        """
        Returns the root mean squared error.
        :param x: The test dataset.
        :param y: The test labels.
        """
        return np.sqrt(self.rss(x, y) / x.shape[0])


class LogisticRegression:
    """
    Implements the Logistic Regression algorithm for binary classification.
    :attrs:
        x_train: The dataset to be used for the algorithm.
        y_train: The labels of the dataset.
        alpha: The learning rate.
        tol: Tolerance level for convergence.
        max_iters: The maximum number of iterations in which the clustering can be done.
    """

    def __init__(self, x_train: np.ndarray, y_train: np.ndarray, alpha: float = 0.01, tol: float=1e-3, max_iters: int=100):
        self.x_train = x_train
        self.y_train = y_train
        self.alpha = alpha
        self.tol = tol
        self.max_iters = max_iters
        self.weights = np.random.rand(x_train.shape[1])

    def sigmoid(self, x: np.ndarray):
        """
        Returns the sigmoid at a point.
        :param x: The point at which the sigmoid is to be calculated.
        """
        return 1 / (1 + np.exp(-self.weights.T @ x.T))

    def train(self):
        """
        Trains the model.
        """
        iters = 0
        while iters < self.max_iters:
            old = self.weights.copy()
            self.weights -= self.alpha * (self.sigmoid(self.x_train) - self.y_train) @ self.x_train
            iters += 1
            if np.linalg.norm(self.weights - old) < self.tol:
                break

    def predict(self, x: np.ndarray):
        """
        Returns the prediction of a point/data.
        :param x: The point/data to be predicted.
        """
        return self.sigmoid(x) >= 0.5

    def accuracy(self, x_test: np.ndarray, y_test: np.ndarray):
        """
        Returns the accuracy of the model.
        """
        return np.average(self.predict(x_test) == y_test)


class FDA:
    """
    Implements the Fisher Discriminant Analysis algorithm for dimensionality reduction.
    :attrs:
        data: The dataset to be used for the algorithm.
        labels: The labels of the dataset.
        means: The means of the dataset.
        Sw: The within-class scatter matrix.
        Sb: The between-class scatter matrix.
    """

    def __init__(self, data: np.ndarray, labels: np.ndarray):
        self.data = data
        self.labels = labels
        self.means = np.array([data[labels == i].mean(axis=0) for i in np.unique(labels)])
        self.Sw = np.sum([np.cov(data[labels == i], rowvar=False) for i in np.unique(labels)], axis=0)
        self.Sb = np.cov(self.means, rowvar=False)

    def fisher_vector(self):
        """
        Returns the Fisher vector.
        """
        return np.linalg.inv(self.Sw) @ (self.means[0] - self.means[1])

    def transform(self):
        """
        Returns the transformed point.
        """
        return self.data @ self.fisher_vector()


class LDA:
    """
    Implements the Linear Discriminant Analysis algorithm for dimensionality reduction.
    :attrs:
        data: The dataset to be used for the algorithm.
        labels: The labels of the dataset.
        means: The means of the dataset.
        Sw: The within-class scatter matrix.
        Sb: The between-class scatter matrix.
    """

    def __init__(self, data: np.ndarray, labels: np.ndarray):
        self.data = data
        self.labels = labels
        self.means = np.array([data[labels == i].mean(axis=0) for i in np.unique(labels)])
        self.Sw = np.sum([np.cov(data[labels == i], rowvar=False) for i in np.unique(labels)], axis=0)
        self.Sb = np.cov(self.means, rowvar=False)

    def eigen(self):
        """
        Returns the eigenvector of the matrix Sw-1Sb of the dataset
        having the largest absolute eigenvalue.
        """
        eigen_values, eigen_vectors = np.linalg.eig(np.linalg.inv(self.Sw) @ self.Sb)
        eigen_pairs = [(eigen_value, eigen_vector) for eigen_value, eigen_vector in zip(eigen_values, eigen_vectors.T)]
        return max(eigen_pairs, key=(lambda x: abs(x[0])))[1]

    def transform(self):
        """
        Returns the transformed dataset.
        """
        return self.data @ self.eigen()