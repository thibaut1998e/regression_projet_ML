# Regression Project – Machine Learning Best Practices

## 📌 Project Goal

The main objective of this project is to explore and apply **best practices in machine learning regression tasks**. The project focuses on preprocessing techniques, feature engineering, model tuning, and evaluation strategies, using a well-known real-world dataset.

## 📊 Dataset: Boston Housing

This project utilizes the **Boston Housing dataset**, a classic dataset in the machine learning community. It contains information collected by the U.S Census Service concerning housing in the area of Boston, Massachusetts.

Each instance describes a town or neighborhood, and the target variable is the **median value of owner-occupied homes (in $1000s)**. The dataset includes 13 numerical/categorical features such as crime rate, proportion of non-retail business acres, nitric oxide concentration, average number of rooms per dwelling, and more.

> If further detail is needed about specific attributes in the dataset, feel free to ask!

## 🛠️ Workflow Overview

The following steps have been carried out in the project:

1. **Handling Missing Values**  
   - Missing data was imputed using two methods:
     - Replacement by **mean** values
     - Replacement using the **closest (nearest)** neighbors approach

2. **Encoding Categorical Attributes**  
   - Categorical features were transformed using **One-Hot Encoding** to convert them into a numerical format suitable for regression models.

3. **Normalization**  
   - All numerical features were normalized to ensure consistent scale and improve the performance of distance-based algorithms.

4. **Feature Selection**  
   Various strategies were applied to select the most informative features:
   - Evaluating **correlation** between input features and the target variable
   - Applying **Principal Component Analysis (PCA)** for dimensionality reduction

5. **Model Tuning and Evaluation**  
   - The dataset was split into **training and test sets**
   - **Cross-validation** techniques were used to tune hyperparameter (here the hyperparameter tested was the number of features selected for regression)
   - Final evaluation was performed on the **test set** to assess model performance


