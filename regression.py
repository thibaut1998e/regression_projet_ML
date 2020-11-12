
"""
HousingData.csv
attributes : 
crim : per capita crime rate by town
zn : proportion of residential land zoned for lots over 25,000 sq.ft.
indus
proportion of non-retail business acres per town.
chas
Charles River dummy variable (= 1 if tract bounds river; 0 otherwise).
nox
nitrogen oxides concentration (parts per 10 million).
rm
average number of rooms per dwelling.
age
proportion of owner-occupied units built prior to 1940.
dis
weighted mean of distances to five Boston employment centres.
rad
index of accessibility to radial highways.
tax
full-value property-tax rate per \$10,000.
ptratio
pupil-teacher ratio by town.
black
1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town.
lstat
lower status of the population (percent).
medv (variable to explain)
median value of owner-occupied homes in \$1000s
"""
import numpy as np
import pandas as pd
from sklearn import *
import math
import matplotlib.pyplot as plt

#print(X['MEDV'])

def get_raw_data(csv_path, variable_to_explain_idx=None):
    """read the data from the csv, replace missing values, shuffle the lines and return the input matrix X
    and its associated values y
    y contains column variable_to_explain_idx and X all the other columns
    if variable_to_explain_idx is none it is set to the last column"""
    data = pd.read_csv(csv_path)
    data_array = data.values
    perm = np.random.permutation(data_array.shape[0])
    data_array = data_array[perm, :] #shuffle the lines
    replace_missing_values(data_array)
    if variable_to_explain_idx is None:
        variable_to_explain_idx = data_array.shape[1]-1
    nb_attribute = data_array.shape[1]
    X = data_array[:, [i for i in range(nb_attribute) if i != variable_to_explain_idx]]
    y = data_array[:, variable_to_explain_idx]
    return X, y


def replace_missing_values(data):
    """replace the missing values by the mean of the corresponding attribute """
    for j in range(data.shape[1]):
        mean_j = 0
        cpt = 0
        for i in range(data.shape[0]):
            if not math.isnan(data[i][j]):
                mean_j += data[i][j]
                cpt += 1
        mean_j/=cpt
        for i in range(data.shape[0]):
            if math.isnan(data[i][j]):
                data[i][j] = mean_j


def normalize(X):
    min_max_scaler = preprocessing.MinMaxScaler()
    X_scaled = min_max_scaler.fit_transform(X)
    return X_scaled


def cov_mat(X):
    #to prevent forgetting the transpose
    return np.cov(X.T)


def covariance(v1, v2):
    #returns the covariance between the 2 vectors
    mean_v1 = np.mean(v1)
    mean_v2 = np.mean(v2)
    return np.mean([(v1[i]-mean_v1) * (v2[i] - mean_v2) for i in range(len(v1))])


def correlation(v1, v2):
    return covariance(v1,v2)/(np.std(v1)*np.std(v2))


def correlation_input_output(X, y):
    #return the correlations between the input variables and the output variable
    return np.array([correlation(X[:, i], y) for i in range(X.shape[1])])


def select_feature_with_highest_correlation(X, y, n_features=5):
    correlations = correlation_input_output(X, y)
    abs_corr = np.abs(correlations)
    perm_to_sort = abs_corr.argsort() #get the permutation used to sort thearray
    perm_to_sort = perm_to_sort[::-1] #reverse
    features_highest_cor = perm_to_sort[:n_features]
    print(features_highest_cor)
    X_prim = X[:, features_highest_cor]
    return X_prim, features_highest_cor


def test_model_with_cross_validation(X, y, n_spits=10):
    from sklearn.model_selection import KFold
    from sklearn.linear_model import LinearRegression
    kf = KFold(n_splits=n_spits)
    total_error = 0
    for train_idx, test_idx in kf.split(X):
        X_train, X_test, y_train, y_test = X[train_idx, :], X[test_idx, :], y[train_idx], y[test_idx]
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        error = np.mean(abs(y_pred-y_test))
        total_error += error
    return total_error/n_spits






X, y = get_raw_data('HousingData.csv')
X = normalize(X)
X, _ = select_feature_with_highest_correlation(X, y, n_features=3)
error = test_model_with_cross_validation(X, y)
print('the average error (L1 norm) computed with cross validation is : ', error)

















