# CSE342 Course Project

## Project Description

The project, to be done in pairs, was held as a private [Kaggle Challenge](https://www.kaggle.com/competitions/sml-project/).
It required us to build a classifier to segregate images into one of 19 different classes of fruits.

## Dataset

The training and testing datasets can be found [**HERE**](https://www.kaggle.com/competitions/sml-project/data) on the Kaggle Challenge page.

## Approach

The project report discusses in detail the failure and success of various approaches we tried.
The final approach used was a Multinomial Logisitic Regression Model with a pipeline of PCA and LDA for dimnsionality reduction.
This way, we were able to achieve a score of $0.82692$ on the final testing dataset and ranked $5^{th}$ on the leaderboard out of 57 competing teams.

To simulate a run of the project, follow the steps given in `main.ipynb`.

## Contributors

- [Siddhant Rai Viksit (2021529)](mailto:siddhant21565@iiitd.ac.in)
- [Divyajeet Singh (2021529)](mailto:divyajeet21529@iiitd.ac.in)