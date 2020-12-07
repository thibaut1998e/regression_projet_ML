
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
import copy as cp
from sklearn.linear_model import LinearRegression
from sklearn.utils import shuffle
from scipy.stats import norm

# print(X['MEDV'])




def get_raw_data(csv_path, variable_to_explain_name=None, prop_test_data=0.05, replace_missing_method=1):
    #Thibaut
    """read the data from the csv, replace missing values, shuffle the lines and return the input matrix X
    and its associated values y
    y contains column variable_to_explain_idx and X all the other columns
    if variable_to_explain_idx is none it is set to the last column
    """

    data = pd.read_csv(csv_path)
    data = shuffle(data)
    if variable_to_explain_name is None:
        variable_to_explain_name = data.columns[-1]
    y = data[variable_to_explain_name]
    X = data[[c for c in data.columns if c != variable_to_explain_name]]
    X, attribute_names = one_hot_encode(X)
    X = X.values
    y = y.values
    replace_missing_values(X, replace_missing_method)
    n_train = int((1 - prop_test_data) * X.shape[0])
    X_train, y_train, X_test, y_test = X[:n_train, :], y[:n_train], X[n_train:, :], y[n_train:]
    return X_train, y_train, X_test, y_test, np.array(attribute_names)

def c_type(df, column):
    #Sanaa
    i = 0
    data_array = df[column].values
    while i < len(data_array):
        if type(data_array[i]) == str: return str
        if math.isnan(data_array[i]) == False:
            c_type = type(data_array[i])
            break

        i = i + 1
    return c_type


def one_hot_encode(df):
    #Sanaa
    """one hote encode the categorical attributes of a pandas data set"""
    columns = df.columns
    #print(columns)
    data = df
    non_cat = [np.float64, np.int64, np.float32, np.int32]
    ys = []
    attribute_names = []
    for i in columns:
        if c_type(df, i) not in non_cat:
            y = pd.get_dummies(df[i], prefix=i)
            attribute_names += list(y.columns)
            data = data.drop(columns=[i])
            ys.append(y)
        else:
            attribute_names.append(i)
    L = [data] + ys
    result = pd.concat(L, axis=1, ignore_index=True)

    return result, attribute_names


def replace_missing_values(data, method=1):
    #Thibaut, Anthony
    """replace the missing values by the mean of the corresponding attribute (method 1) or by the value of the closest observation (method 2)"""
    if method in [1, 'mean'] :
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
    elif method in [2, 'closest'] :
        for i in range(data.shape[0]) :
            present_data = []
            for j in range(data.shape[1]) :
                if not math.isnan(data[i][j]) :
                    present_data.append(j)
            if len(present_data) < data.shape[1] :
                closest = 0
                dist_closest = 100000000
                for i_ in range(data.shape[0]) :
                    if i_ != i :
                        dist = np.sqrt(sum([(data[i][j] - data[i_][j]) **2 for j in present_data]))
                        if not math.isnan(dist) and dist < dist_closest :
                            all_missing_data_present = True
                            for j in range(data.shape[1]) :
                                if math.isnan(data[i][j]) :
                                    if math.isnan(data[i_][j]) :
                                        all_missing_data_present = False
                                        break
                            if all_missing_data_present :
                                closest = i_
                                dist_closest = dist
                for j in range(data.shape[1]) :
                    if math.isnan(data[i][j]) :
                        data[i][j] = data[closest][j]




def normalize(X):
    #Thibaut
    min_max_scaler = preprocessing.MinMaxScaler()
    X_scaled = min_max_scaler.fit_transform(X)
    return X_scaled, min_max_scaler





def covariance(v1, v2):
    #Thibaut
    # returns the covariance between the 2 vectors
    mean_v1 = np.mean(v1)
    mean_v2 = np.mean(v2)
    return np.mean([(v1[i] - mean_v1) * (v2[i] - mean_v2) for i in range(len(v1))])


def correlation(v1, v2):
    return covariance(v1, v2) / (np.std(v1) * np.std(v2))


def correlation_input_output(X, y):
    # return the correlations between the input variables and the output variable
    #Thibaut
    return np.array([correlation(X[:, i], y) for i in range(X.shape[1])])


def select_features_with_highest_correlation(X, y, n_features=5, title=''):
    #Thibaut
    #c_X = cp.deepcopy(X)
    correlations = correlation_input_output(X, y)
    abs_corr = np.abs(correlations)
    perm_to_sort = abs_corr.argsort()  # get the permutation used to sort thearray
    perm_to_sort = perm_to_sort[::-1]  # reverse
    features_highest_cor = perm_to_sort[:n_features]
    # print(features_highest_cor)
    X_prim = X[:, features_highest_cor]
    if title != '':
        L = [np.sum(abs_corr[perm_to_sort][:i + 1]) for i in range(len(correlations) - 1)]
        L /= np.sum(abs_corr)
        plt.plot(L)
        plt.xlabel('nb of features')
        plt.ylabel('total correlation')
        plt.title(title)
        plt.show()
    return X_prim, features_highest_cor, abs_corr[perm_to_sort][:n_features]


def select_features_with_PCA(X, n_features=5):
    #Anthony
    #c_X = cp.deepcopy(X)
    pca = decomposition.PCA(n_components=n_features)
    pca.fit(X)
    return pca.transform(X), pca


def train_and_obtain_model(X, y):
    #Anthony
    model = LinearRegression()
    model.fit(X, y)
    return model

def L1(y_pred, y):
    return np.mean(abs(y_pred - y))


def test_model_with_cross_validation(X, y, n_spits=10):
    #Thibaut
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=n_spits)
    total_error = 0
    for train_idx, test_idx in kf.split(cp.deepcopy(X)):
        X_train, X_test, y_train, y_test = X[train_idx, :], X[test_idx, :], y[train_idx], y[test_idx]
        model = train_and_obtain_model(X_train, y_train)
        y_pred = model.predict(X_test)
        error = L1(y_pred, y_test)
        total_error += error
    return total_error / n_spits


def plot_error_against_nb_features(X, y, training_set_name, n_max=None, method='highest corr'):
    #Anthony
    if n_max is None:
        n_max = X.shape[1]
    errors = []
    n_features = [n for n in range(1, n_max)]
    for n in range(1, n_max):
        if method == 'highest corr':
            X_n, _, _ = select_features_with_highest_correlation(X, y, n)
        elif method == 'PCA':
            X_n, _ = select_features_with_PCA(X, n_features=n)
        else:
            raise Exception("Argument 'method' is unvalid")
        errors.append(test_model_with_cross_validation(X_n, y))
    plt.plot(n_features, errors)
    plt.xlabel('number of features')
    plt.ylabel('l1 error')
    plt.title(f'{training_set_name} data set, error against the number of features selected with method {method}')
    plt.show()


def choose_nb_features(X, y, alpha=0.99, n_max=None, method='highest corr'):
    #Anthony
    if n_max is None:
        n_max = X.shape[1]
    errors = [10 ** 15]
    for n in range(1, 4):
        if method == 'highest corr':
            X_n, _, _ = select_features_with_highest_correlation(X, y, n)
        elif method == 'PCA':
            X_n, _ = select_features_with_PCA(X, n_features=n)
        errors.append(test_model_with_cross_validation(X_n, y))
    n = 4
    while errors[n - 1] < alpha * errors[n - 4] and n < n_max:

        n += 1
        if method == 'highest corr':
            X_n, _, _ = select_features_with_highest_correlation(X, y, n)
        elif method == 'PCA':
            X_n, _ = select_features_with_PCA(X, n_features=n)
        errors.append(test_model_with_cross_validation(X_n, y))

    return n


def predict_new_data(model, X_new):
    return model.predict(X_new)


def main(training_set_path):
    X_train, y_train, X_test, y_test, attribute_names = get_raw_data(training_set_path, prop_test_data=0.2)
    X_train, min_max_scaler = normalize(X_train)
    X_test = min_max_scaler.fit_transform(X_test)
    for method in ['highest corr', 'PCA']:
        print('method', method)
        plot_error_against_nb_features(X_train, y_train, training_set_name=training_set_path,method=method)
        nb_features = choose_nb_features(X_train, y_train, method=method)
        print(f'number of features selected with method {method} : {nb_features}')
        if method == 'highest corr':
            title = f'total correlation against number of features for {training_set_path}'
            X_train_proj, features_highest_cor, abs_cor = select_features_with_highest_correlation(X_train, y_train,
                                                                                          n_features=nb_features,
                                                                                          title=title)
            print(f'feature selected are {attribute_names[features_highest_cor]}')
            print(f'corresponding absolute correlation with output variable are {abs_cor}')
            X_test_proj = X_test[:, features_highest_cor]
        else:
            X_train_proj, pca = select_features_with_PCA(X_train, n_features=nb_features)
            X_test_proj = pca.transform(X_test)

        model = train_and_obtain_model(X_train_proj, y_train)
        y_pred = model.predict(X_test_proj)
        error = L1(y_pred, y_test)
        std_dev = np.std(np.abs(y_pred-y_test))
        confidence_level = 0.95
        qt = norm.ppf((1-confidence_level)/2)
        delta = -qt*std_dev/np.sqrt(len(y_test))
        print(f'L1 error on test set ({len(y_pred)} data): ', error)
        print(f'confidence intervall of L1 error with confidence level {confidence_level} : [{error-delta}, {error+delta}]')
        print('*********')


for data_set in ['HousingData.csv', 'spnbmd.csv']:
    print(f'processing data set {data_set}')
    main(data_set)
    print('-----------------------------------')


























'''

#X = select_features_with_PCA(X, n_features=1)
error = test_model_with_cross_validation(X, y)
print('the average error (L1 norm) computed with cross validation is : ', error)
'''














