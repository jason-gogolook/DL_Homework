import numpy as np
import pandas as pd


class BasicDataProcessor:
    def __init__(self):
        self.train_data = pd.read_csv('train.csv')  # change the name and the path as needed
        self.test_data = pd.read_csv('test.csv')

        # print(self.train_data.dtypes[self.train_data.dtypes != 'object'].index)
        # print(self.train_data.shape)
        #
        # print(self.test_data.dtypes[self.test_data.dtypes != 'object'].index)
        # print(self.test_data.shape)

        print(self.train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]],
              "\n==================================================")

    def process_data(self):
        # remove `ID`  and `SalePrice` from the features
        all_features = pd.concat((self.train_data.iloc[:, 1:-1], self.test_data.iloc[:, 1:]))
        print(all_features.dtypes[all_features.dtypes != 'object'].index)
        print('[all features shape]: ', all_features.shape,
              "\n==================================================")

        print('[Before Normalization] \n',
              all_features.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]],
              "\n==================================================")
        # Normalize the numerical features
        numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
        all_features[numeric_features] = all_features[numeric_features].apply(lambda x: (x - x.mean()) / (x.std()))

        # Set missing values to 0
        all_features = all_features.fillna(0)

        # Examine data after normalizing numeric features
        print('[After Normalization] \n',
              all_features.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])
        print('[all features shape]: ', all_features.shape,
              "\n==================================================")

        # Dummy_na=True refers to a missing value being a legal eigenvalue, and creates an indicative feature for it.
        all_features = pd.get_dummies(all_features, dummy_na=True, dtype=int)
        print(f'Shape of all training data: {all_features.shape}')
        print(f'Shape of non-processed training: {self.train_data.shape}')
        print(f'Shape of non-processed test: {self.test_data.shape}',
              "\n==================================================")

        # Examine data after one hot encoding
        print('[After One Hot Encoding] \n',
              all_features.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]],
              "\n==================================================")

        # all_features = all_features.map(lambda x: int(x) if isinstance(x, bool) else x)
        # print('[After transfer boolean] \n',
        #       all_features.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]],
        #       "\n==================================================")

        n_train = self.train_data.shape[0]

        # the train feature are in all_features[:n_train].values - need to convert them into a pytorch tensor??
        train_features = np.array(all_features[:n_train].values)
        print(train_features[0])

        # the test feature are in all_features[n_train:].values - need to convert them into a pytorch tensor??
        test_features = np.array(all_features[n_train:].values)

        # the train labels are in train_data.SalePrice.values - need to convert them into a pytorch tensor??
        train_labels = np.array(self.train_data.SalePrice.values).reshape((-1, 1))

        return train_features, test_features, train_labels
