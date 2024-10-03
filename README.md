# Boston Housing Price Prediction

This project aims to predict house prices using the Boston Housing dataset. The notebook covers data exploration, preprocessing, feature engineering, model building, and evaluation using various machine learning techniques.

## Table of Contents

1. [Introduction](#introduction)
2. [Data Exploration and Visualization](#data-exploration-and-visualization)
3. [Data Preprocessing and Feature Engineering](#data-preprocessing-and-feature-engineering)
4. [Model Building and Evaluation](#model-building-and-evaluation)
5. [Advanced ML Challenge](#advanced-ml-challenge)
6. [Results](#results)

## Introduction

The goal of this project is to predict the median value of owner-occupied homes (`MEDV`) in the Boston area using various features from the dataset.

## Data Exploration and Visualization

1. **Load the Boston Housing dataset and display the first 5 rows.**
    - `data.head()`

2. **Generate descriptive statistics.**
    - `data.describe()`

3. **Create a correlation heatmap for all features in the dataset.**
    - `createHeatmap(data)`

4. **Plot a scatter plot of 'RM' (average number of rooms) vs. 'MEDV' (median value of owner-occupied homes).**
    - `createScatterPlot(data, 'RM', 'MEDV')`

## Data Preprocessing and Feature Engineering

1. **Check for missing values in the dataset and handle them appropriately.**
    - `data.info()`
    - `data.isnull().sum()`

2. **Normalize the numerical features using `StandardScaler`.**
    - `scaler = StandardScaler()`
    - `scaled_inputs = scaler.transform(scaler_data)`

3. **Create a new feature that represents the ratio of 'LSTAT' to 'RM'.**
    - `scaled_inputs_df['lstat_rm_ratio'] = scaled_inputs_df["LSTAT"] / scaled_inputs_df["RM"]`

## Model Building and Evaluation

1. **Implement a Linear Regression model to predict house prices.**
    - Split the data into training and testing sets (80-20 split).
    - Train the model on the training data.
    - Make predictions on the test data.
    - Calculate and print the Mean Squared Error and R-squared score.

2. **Implement a Random Forest Regressor for the same prediction task.**
    - Use the same train-test split as in Task 1.
    - Train the model with at least 100 trees.
    - Make predictions on the test data.
    - Calculate and print the Mean Squared Error and R-squared score.
    - Plot feature importance for the top 5 most important features.

## Advanced ML Challenge

1. **Implement a simple neural network using PyTorch to predict house prices.**
    - Preprocess the data appropriately for neural network input.
    - Design a network with at least one hidden layer.
    - Train the model for a suitable number of epochs.
    - Evaluate the model's performance using Mean Squared Error and R-squared score.

## Results

The results of the models are compared based on Mean Squared Error (MSE) and R-squared (R²) score:

- **Linear Regression**
    - Mean Squared Error: `mse_linear`
    - R Squared Error: `rse_linear`

- **Random Forest Regressor**
    - Mean Squared Error: `mse_forest`
    - R Squared Error: `rse_forest`

- **Neural Network**
    - Mean Squared Error: `mse`
    - R Squared Error: `r2`

From the results, we can conclude that the `RandomForestRegressor` has a lower error rate compared to `LinearRegression` and `LinearRegression_NeuralNetwork`.

## Conclusion

This project demonstrates the process of predicting house prices using various machine learning models. The Random Forest Regressor outperformed the other models in terms of prediction accuracy.