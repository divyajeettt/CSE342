import pandas as pd
import numpy as np


class Dataset:
    """
    A class to load and preprocess the dataset.
    :param path: The path to the csv file.
    """

    def __init__(self, path):
        """
        __init__ self
        The initialized object checks if the data is training type or testing type
        and parses labels accordingly.
        Note: The training data should have a column named "category" and the testing data should not.
        """
        df = pd.read_csv(path, index_col=0)
        self.labels = None
        self.y = None
        self.num_classes = 0
        if "category" in df.columns:
            unique_labels = df["category"].unique()
            self.labels = dict(zip(unique_labels, range(len(unique_labels))))
            self.num_classes = len(self.labels)
            df["category"] = df["category"].map(self.labels)
            self.y = df["category"].values
            self.x = df.drop("category", axis=1).values
        else:
            self.x = df.values

    def to_one_hot(self):
        """
        Converts the output to be predicted to one-hot encoded vectors.
        """
        if self.y is None:
            raise ValueError("No labels found in dataset")
        else:
            self.y = np.eye(self.num_classes)[self.y]

    def get_cat_to_label(self):
        """
        Returns a dictionary mapping the numerical labels to their original textual representation.
        """
        return dict(zip(self.labels.values(), self.labels.keys()))


def write_to_csv(path, labels, mapping):
    """
    Writes the predictions to a csv file.
    :param path: The path to the csv file.
    :param labels: The labels to be written.
    :param mapping: A dictionary mapping the numerical labels to their original textual representation.
    """
    df = pd.DataFrame({
        "Id": range(len(labels)), "Category": [mapping[i] for i in labels]
    })
    df.set_index("Id", inplace=True)
    df.to_csv(path)