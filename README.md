# Retailer Product Return Prediction

## Table of Contents
1. [Introduction](#introduction)
2. [Project Overview](#project-overview)
3. [Data Analysis](#data-analysis)
4. [Model Development](#model-development)
5. [Model Deployment](#model-deployment)
6. [Conclusion](#conclusion)
7. [References](#references)
8. [Cloning Instructions](#cloning-instructions)

## Introduction
Welcome to the Retailer Product Return Prediction project! In this project, we aim to tackle the challenge of predicting product returns for a retailer who sells shoes across multiple shops, both online and offline.

## Project Overview
Project is structured into three main phases:

### Data Analysis
Started by conducting thorough data analysis to gain insights into the factors driving product returns. This analysis included exploring various aspects of the dataset such as the distribution of returns across different categories, correlation analysis, and visualizations.

### Model Development
Developed machine learning models to predict product returns based on the available dataset. We experimented with different algorithms such as Decision Trees, Random Forest, Logistic Regression, and Naive Bayes, aiming to find the best-performing model for our prediction task.

### Model Deployment
Once identified the best-performing model, deployed it using Streamlit, allowing stakeholders to interactively explore the predictions and gain insights from the model.

## Data Analysis
The data analysis revealed several interesting findings about the dataset. Some key insights include:

- Distribution of returns across different categories such as brand, model group, and product group.
- Correlation analysis to identify relationships between features and return likelihood.
- Visualizations to illustrate trends and patterns in the data.

For a detailed analysis, please refer to the dedicated branch in our repository.

## Model Development
Experimented with several machine learning algorithms, including Decision Trees, Random Forest, Logistic Regression, and Naive Bayes. Here are the analysis results of each model:

| Model           | Accuracy | Precision | Recall | F1-score | ROC AUC | Time (seconds) |
|-----------------|----------|-----------|--------|----------|---------|----------------|
| Decision Trees  | 95.56%   | 94.01%    | 97.31% | 95.63%   | 95.56%  | 24.98          |
| Random Forest   | 97.53%   | 96.47%    | 98.66% | 97.55%   | 99.56%  | 634.67         |
| Logistic Model  | 50.07%   | 50.02%    | 49.44% | 49.73%   | 50.24%  | 2.56           |
| Naive Bayes     | 49.98%   | 49.94%    | 61.39% | 55.07%   | 50.43%  | 1.74           |

Based on the analysis results, the Random Forest model outperformed the other models in terms of accuracy, precision, recall, and F1-score.

In addition to traditional machine learning models, we also explored the use of deep learning techniques, specifically Artificial Neural Networks (ANN) implemented with Keras. Here are the analysis results of the ANN model:

| Model                 | Accuracy | Precision | Recall  | F1-score | Confusion Matrix     |
|-----------------------|----------|-----------|---------|----------|----------------------|
| Artificial Neural Net | 88.01%   | 89.79%    | 85.79%  | 87.74%   | [[305335, 33027], [48103, 290373]] |

The ANN model achieved an accuracy of 88.01%, with precision, recall, and F1-score of 89.79%, 85.79%, and 87.74% respectively. The confusion matrix illustrates the distribution of true positive, true negative, false positive, and false negative predictions.

## Model Deployment
The Random Forest model is deployed using Streamlit, allowing stakeholders to input new data and obtain predictions on product returns in real-time. The Streamlit app can be accessed [here](link-to-streamlit-app).

## Conclusion
In conclusion, our project successfully addressed the challenge of predicting product returns for the retailer. The deployed model provides valuable insights into return likelihood, enabling the retailer to take proactive measures to minimize returns and improve customer satisfaction.

## References
- Streamlit Documentation: [https://docs.streamlit.io/](https://docs.streamlit.io/)
- Scikit-learn Documentation: [https://scikit-learn.org/stable/documentation.html](https://scikit-learn.org/stable/documentation.html)

## Cloning Instructions
To clone this repository and run the code locally, follow these steps:

1. Open a terminal or command prompt.
2. Navigate to the directory where you want to clone the repository.
3. Run the following command:
