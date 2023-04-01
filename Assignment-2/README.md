# Assignment-2

## Problem 1

Problem 1 requires us to detect potential outliers in a given [dataset of properties of glass](https://www.kaggle.com/datasets/uciml/glass) using the [Mahalanobis Distance Algorithm](https://en.wikipedia.org/wiki/Mahalanobis_distance) and the [Local Outlier Factor (LOF) Algorithm](https://en.wikipedia.org/wiki/Local_outlier_factor).

The dataset is cleaned and preprocessed and the practical distances are calculated using the above algorithms. The [Otsu Thresholding Method](https://en.wikipedia.org/wiki/Otsu%27s_method) is used to determine the best thresholds of separation between the outliers and the inliers.

## Problem 2

On a given [dataset of heart patients](https://www.kaggle.com/datasets/zhaoyingzhu/heartcsv), the problem required us to implement the [Logistic Regression Algorithm](https://en.wikipedia.org/wiki/Logistic_regression) to classify patients into two categories: those with heart disease and those without.

Another machine learning model using the [Fisher Discriminant Analysis (FDA) Algorithm](https://en.wikipedia.org/wiki/Linear_discriminant_analysis#Fisher's_linear_discriminant) was also implemented to preprocess the dataset into the one-dimensional format maximising the separation between the two classes.

The data preprocessed using FDA is also run through the Logistic Regression model and their results were compared.

A set of data reduced to 5, 10, and 15 dimensions using the Principal Component Analysis (PCA) Algorithm was also processed through FDA. These datasets were used to train the Logistic Regression model and their results were compared.

## Problem 3

This problem required us to implement the [Linear Regression Algorithm](https://en.wikipedia.org/wiki/Linear_regression) to predict a continuous target variable on a [dataset of real estate prices](https://www.kaggle.com/datasets/quantbruce/real-estate-price-prediction). The dataset was preprocessed and the model was trained using the [Normal Equation Approach](https://en.wikipedia.org/wiki/Ordinary_least_squares#Derivation_of_the_normal_equation).

The model is evaluated using its [$R^{2}$ Score](https://en.wikipedia.org/wiki/Coefficient_of_determination#:~:text=R2%20is%20a%20measure,predictions%20perfectly%20fit%20the%20data.) and its [Root Mean Squared Error](https://en.wikipedia.org/wiki/Root-mean-square_deviation).

A derivation of the Normal Equation approach (as asked for in the problem) is provided in the notebook.

## Problem 4 and Problem 5

This problem deals with the famous [IRIS dataset](https://www.kaggle.com/datasets/uciml/iris).

Problem 4 deals with training a machine learning model using the $k$-Nearest Neighbors Algorithm with $k=5$ to classify the data into different classes. The data was then processed through the [Linear Discriminant Analysis (LDA) Algorithm](https://en.wikipedia.org/wiki/Linear_discriminant_analysis) and the model was retrained.

For Problem 5, the [Multinomial Logistic Regression Algorithm](https://en.wikipedia.org/wiki/Multinomial_logistic_regression) was implemented and the model was trained on the same dataset. An explanation of the algorithm and how Logistic Regression is generalized to multi-class classification problems is given in the notebook.

Finally, the results and performance of all the models were compared.

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
├── Assignment-2.pdf
├── *.ipynb
├── utils.py
└── README.md
```

## References

- [FLDA by Prof. Olga Veksler](https://www.csd.uwo.ca/~oveksler/Courses/CS434a_541a/Lecture8.pdf)
- [The Normal Equation for Linear Regression](https://eli.thegreenplace.net/2015/the-normal-equation-and-matrix-calculus/)
- [The limitations of the Normal Equation approach](https://towardsdatascience.com/normal-equation-a-matrix-approach-to-linear-regression-4162ee17024)
- [Multinomial Logistic Regression](https://towardsdatascience.com/multiclass-classification-algorithm-from-scratch-with-a-project-in-python-step-by-step-guide-485a83c79992)
