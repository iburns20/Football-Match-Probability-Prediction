# Football Match Probability Prediction

This project focuses on predicting the outcome of football matches (home win, away win, or draw) using machine learning techniques. The dataset contains over 150,000 historical football matches from 860+ leagues and 9500 teams, spanning from 2019 to 2021.

## Project Overview

- **Objective**: Predict the probability of match outcomes (home win, away win, draw) based on a historical dataset of football matches.
- **Dataset**: Kaggle competition dataset containing over 150,000 football matches with 187 features.
- **Techniques**: 
  - **Dimensionality Reduction**: PCA, FLD, t-SNE
  - **Classification Models**: Bayesian, Non-Bayesian, Supervised, Unsupervised, Parametric, Non-parametric models, including MPP, Random Forest, Neural Networks, and HAC.
  - **BKS Fusion**: A framework combining multiple classifiers to improve prediction accuracy.
- **Goal**: To implement various machine learning models and compare their performance based on log-loss evaluation.

## Key Features

- **Preprocessing**: Includes feature selection, scaling, and dimensionality reduction using PCA and FLD.
- **Models**:
  - Maximum Posterior Probability (MPP) with Euclidean and Mahalanobis distance metrics.
  - Random Forest Classifier with feature weighting.
  - Neural Networks with backpropagation.
  - Hierarchical Agglomerative Clustering (HAC).
  - BKS fusion technique to combine predictions from MPP classifiers.
- **Evaluation**: Models are evaluated using log-loss and k-fold cross-validation.

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/football-match-prediction.git
cd football-match-prediction
pip install -r requirements.txt
```

## Usage

To train and test the models, run the following command:

```bash
python main.py
```
Ensure the dataset is placed in the `data/` directory before running the script.

## Performance

- **Neural Networks**: Best performing model with a log-loss of 0.8724.
- **Random Forest**: Achieved a log-loss of 1.0300.
- **BKS Fusion**: Combined predictions from MPP models to improve accuracy, with a log-loss of 1.0277.

## Contributions

- **Ian Burns**: Dataset description, feature selection, feature reduction, methods.
- **Gabriel Carson**: Model explanation, performance evaluation.
- **Ethan Head**: Performance evaluation and model comparisons.

## References

- Kaggle Football Match Probability Prediction competition: [Link](https://www.kaggle.com/competitions/football-match-probability-prediction)
- Relevant academic papers and Kaggle discussions.


