# Assignment-1

## Problem 1

Problem 1 deals with training a machine learning model using the [$k$-Nearest Neighbors Algorithm](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm) to predict the class of images from the famous [MNIST as JPG Dataset](https://www.kaggle.com/datasets/scolianni/mnistasjpg).

A [Principal Component Analysis (PCA) Algorithm](https://en.wikipedia.org/wiki/Principal_component_analysis) was implemented to reduce the dimensionality of the data from 784 to 5, 25, and 125 principal components. The results of the PCA algorithm were then used to train the k-NN model separately.

Lastly, the explained variances of the PCA algorithm were plotted against the number of principal components to find out the number of principal components that explained 80% of the variance in the data.

## Problem 2

On a given [dataset of 2-dimensional points](https://drive.google.com/file/d/1-0zx-cXze6ja777SN_NkMYVCzidU2lXw/view?usp=sharing), the problem required us to implement a [$k$-Means Clustering Algorithm](https://en.wikipedia.org/wiki/K-means_clustering) to cluster the points into $k$ clusters.

To analyse the clustering, the [Silhouette Analysis Algorithm](https://en.wikipedia.org/wiki/Silhouette_(clustering)) was implemented to calculate the Silhouette Coefficient for each point in the dataset. This determined the optimal value of $k$ for the best clustering.

With the best $k$ at hand, the [Fuzzy $C$-Means Clustering Algorithm](https://en.wikipedia.org/wiki/Fuzzy_clustering) was implemented to cluster the points into $C = k$ clusters. The objective function $J(U, V)$ was then reported at the given values of $c$, $m$, and $\beta$.

## Problem 3

The problem required us to perform the [Mean Shift Clustering Algorithm](https://en.wikipedia.org/wiki/Mean_shift) for separating a given [image](https://drive.google.com/file/d/15-6l7_51OZ3wIw37d8a6SX2dfWxUQGl4/view?usp=sharing) into distinct parts.

It involved searching for a suitable bandwidth to make sure the vegetables in the image look as separated as possible.

## Problem 4

Given two signals, a sinusoid and a ramp wave, this problem required us to mix them using a given mixing matrix and then recover the original signals by implementing the [Independent Component Analysis (ICA) Algorithm](https://en.wikipedia.org/wiki/Mean_shift).

## Problem 5

Given a list of some points, this problem required us to manually use the $k$-NN Algorithm to predict the target variables for a new sample having certain features. The list of points is attached in the Assignment document.

## Directory Structure

To ensure functioning, the following points should be noted:

- All `.ipynb` notebooks must be in the same directory.
- `Datasets.zip` must be unzipped into the same directory as the notebooks.
- The `utils.py` file must be in the same directory as the notebooks.

In essence, this directory structure should be followed:

```bash
Assignment-1
├── Datasets
│   └── Question-*
├── Assignment-1.pdf
├── *.ipynb
├── Question-5.*
├── utils.py
└── README.md
```

## References

- [How to efficently use NumPy](https://towardsdatascience.com/numpy-python-made-efficient-f82a2d84b6f7)
- [Official NumPy Documentation](https://numpy.org/doc/stable/)
- [Principal Component Analysis with Python](https://www.geeksforgeeks.org/principal-component-analysis-with-python/)
- [How to make $k$-NN Algorithm more efficient](https://stackoverflow.com/questions/51688568/faster-knn-algorithm-in-python)
- [$k$-Means Clustering with Silhouette Analysis](https://dzone.com/articles/kmeans-silhouette-score-explained-with-python-exam)
- [Fuzzy Clustering](https://www.geeksforgeeks.org/ml-fuzzy-clustering/)
- [Difference between PCA and ICA](https://www.geeksforgeeks.org/ml-independent-component-analysis/)
