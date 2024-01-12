# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch import optim, nn

from data import HouseData
from data_process import BasicDataProcessor
from models import LinearRegressionModel, linearRegression
from trainer import Trainer


def load_data_from_csv():
    train = pd.read_csv("train.csv")  # Load train data (Write train.csv directory)
    # test = pd.read_csv("test.csv")  # Load test data (Write test.csv directory)
    #
    # all_data = pd.concat([train, test], ignore_index=True)  # Make train set and test set in the same data set
    all_data = train
    return all_data  # Visualize the DataFrame data


def show_features_more_than_1000_NULL(all_data):
    # Plot features with more than 1000 NULL values

    features = []
    null_values = []
    for i in all_data:
        if (all_data.isna().sum()[i]) > 1000 and i != 'SalePrice':
            features.append(i)
            null_values.append(all_data.isna().sum()[i])
    y_pos = np.arange(len(features))
    plt.bar(y_pos, null_values, align='center', alpha=0.5)
    plt.xticks(y_pos, features)
    plt.ylabel('NULL Values')
    plt.xlabel('Features')
    plt.title('Features with more than 1000 NULL values')
    plt.show()


def replace_null_values_with_mean(all_data):
    all_data = all_data.dropna(axis=1, thresh=1000)  # Drop columns that contain more than 1000 NULL values
    all_data = all_data.fillna(all_data.mean())  # Replace NULL values with mean values
    return all_data


def convert_str_to_int(all_data):
    # Convert string values to integer values
    # Dealing with NULL values
    all_data = pd.get_dummies(all_data)  # Convert string values to integer values
    return all_data


def convert_bool_to_int(all_data):
    # Convert boolean values to integer values
    # Dealing with NULL values
    all_data = all_data.astype(int)
    return all_data


def drop_correlated_features(all_data):
    # Drop features that are correlated to each other
    covariance_matrix = all_data.corr()
    list_of_features = [i for i in covariance_matrix]
    set_of_dropped_features = set()
    for i in range(len(list_of_features)):
        for j in range(i + 1, len(list_of_features)):  # Avoid repetitions
            feature1 = list_of_features[i]
            feature2 = list_of_features[j]
            if abs(covariance_matrix[feature1][feature2]) > 0.8:  # If the correlation between the features is > 0.8
                set_of_dropped_features.add(feature1)  # Add one of them to the set
    # I tried different values of threshold and 0.8 was the one that gave the best results

    all_data = all_data.drop(set_of_dropped_features, axis=1)
    return all_data


def drop_not_correlated_features(all_data):
    # Drop features that are not correlated with output('SalePrice')
    non_correlated_with_output = [column for column in data if abs(data[column].corr(data["SalePrice"])) < 0.045]
    # I tried different values of threshold and 0.045 was the one that gave the best results
    all_data = all_data.drop(non_correlated_with_output, axis=1)
    return all_data


def show_feature_with_outliers(all_data, feature):
    # Plot one of the features with outliers
    plt.plot(all_data[feature], all_data['SalePrice'], 'bo')
    plt.axvline(x=75000, color='r')
    plt.ylabel('SalePrice')
    plt.xlabel(feature)
    plt.title('SalePrice in function of ' + feature)
    plt.show()


def outliers_iqr(ys):
    quartile_1, quartile_3 = np.percentile(ys, [25, 75])  # Get 1st and 3rd quartiles (25% -> 75% of data will be kept)
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 1.5)  # Get lower bound
    upper_bound = quartile_3 + (iqr * 1.5)  # Get upper bound
    return np.where((ys > upper_bound) | (ys < lower_bound))  # Get outlier values


def drop_outliers(all_data):
    # First, we need to separate the data
    # (Because removing outliers ⇔ removing rows, and we don't want to remove rows from test set)
    new_train = all_data.iloc[:1460]
    new_test = all_data.iloc[1460:]

    # Third, we will drop the outlier values from the train set
    train_without_outliers = new_train  # We can't change train while running through it

    for column in new_train:
        outlier_values_list = np.ndarray.tolist(outliers_iqr(new_train[column])[0])  # outliers_iqr() returns an array
        train_without_outliers = new_train.drop(outlier_values_list)  # Drop outlier rows

    return train_without_outliers

    # trainWithoutOutliers = newTrain


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    data = BasicDataProcessor()
    train_features, test_features, train_labels = data.process_data()
    model = LinearRegressionModel(len(train_features[0]), 1)
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    trainer = Trainer(model, optimizer, loss_fn)
    trainer.fit(HouseData(train_features, train_labels))

    # data = load_data_from_csv()
    # # show_features_more_than_1000_NULL(data)
    # data = convert_str_to_int(data)
    # data = replace_null_values_with_mean(data)
    # data = convert_bool_to_int(data)
    # data = drop_correlated_features(data)
    # data = drop_not_correlated_features(data)
    # # show_feature_with_outliers(data, 'LotArea')
    # data = drop_outliers(data)
    # data.reset_index(drop=True, inplace=True)  # Reset indexes
    #
    # x = data.drop("SalePrice", axis=1)  # Remove SalePrice column
    # # print(x, "\n")
    # # y_hat = np.log1p(data["SalePrice"])  # Get SalePrice column {log1p(x) = log(x+1)}
    # # print(y_hat, "\n")
    # # y = data["SalePrice"]
    # # print(y)
    #
    # # train model
    # model = LinearRegressionModel(len(x.columns), 1)
    # optimizer = optim.SGD(model.parameters(), lr=0.001)
    # loss_fn = nn.MSELoss()
    #
    # # 假設 train_loader 是我們的數據加載器
    # train_loader = HouseData(data)
    #
    # trainer = Trainer(model, optimizer, loss_fn)
    # trainer.fit(train_loader)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
