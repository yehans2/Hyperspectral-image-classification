import os
import random

import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from tensorflow import keras
from keras.utils import normalize
import scipy.io as sio


class DataHolder:
    """Load in data and hold them. This class handles all tasks related to data.
       Including normalization, dimension reduction, and train test split.
    """

    def __init__(self, dataset: str = "indianpines"):
        # A dict used to store all the path to dataset.
        self._dict_dataset_path = {
            "indianpines": "./dataset/indian_pines/",
            "salinas": "./dataset/salinas/",
            "pavia": "./dataset/pavia_u/"
        }
        # This dict is used to guide the program to corrected find the key load from .m files
        self._dict_dataset_key = {
            "indianpines": {"data": 'indian_pines_corrected', "label": 'indian_pines_gt'},
            "salinas": {"data": 'salinas_corrected', "label": 'salinas_gt'},
            "pavia": {"data": "paviaU", "label": "paviaU_gt"}
        }
        self.dataset = dataset
        self.data_path = self._dict_dataset_path[self.dataset]

        # Record data's meta data.
        self.X_data = None
        self.y_data = None
        self.data_dim = {
            'n_samples': None,
            'n_bands': None
        }
        self.dr_algo = None
        # The mean and std of the raw data.
        self.origin_mean, self.origin_std = None, None
        # Output of the data processor
        self.status_data = dict()

    def main(self, command, data):
        """Execute the programm based on command and data

        Args:
            command (str): Available commands: load_data, dim_reduce, split, and get_data
            data (str or None): specify the DR algorithm.
        """
        dict_actions = {
            'load_data': self.load_data,
            'dim_reduce': self.dim_reduce,
            'split': self.train_test_split,
            'get_data': self.get_data
        }

        try:
            result = dict_actions[command](data)
        except KeyError:
            print(f"[{command}] is an unknown command!")
        else:
            if result:
                return result

    def load_data(self, dummy=None):
        # Get all the data paths in the folder
        data_in_folder = os.listdir(self.data_path)
        if len(data_in_folder) != 2:
            print(f"Please check the completeness of the data!")

        # Deal with two files in the folder
        for data in data_in_folder:
            cur_data_path = os.path.join(self.data_path, data)
            if data[0] == "X":
                self.X_data_all = sio.loadmat(cur_data_path)[self._dict_dataset_key[self.dataset]["data"]]
            else:
                self.y_data_all = sio.loadmat(cur_data_path)[self._dict_dataset_key[self.dataset]["label"]]
        
        self.data_dim["origin"] = self.X_data_all.shape
        # Flatten the data into the shape of [n_samples, n_bands]
        self.data_dim['n_bands'] = self.X_data_all.shape[-1]

        self.X_data_all = np.array(
            self.X_data_all).reshape(-1, self.data_dim['n_bands'])
        self.y_data_all = np.array(self.y_data_all).reshape(-1)

        self.X_data, self.y_data = self._ignore_background(
            self.X_data_all, self.y_data_all)

        self.y_data = np.subtract(self.y_data, 1)

        self.data_dim['n_samples'] = self.y_data.shape[0]
        self.data_dim['n_classes'] = len(np.unique(self.y_data))

        # Normalization
        scaler = StandardScaler()
        self.X_data_all = scaler.fit_transform(self.X_data_all)
        self.X_data = scaler.transform(self.X_data)
        self.origin_mean, self.origin_std = scaler.mean_, scaler.var_

        # Shuffling
        shuffled_idx = [i for i in range(len(self.X_data))]
        random.shuffle(shuffled_idx)
        self.X_data, self.y_data = self.X_data[shuffled_idx, :], \
            self.y_data[shuffled_idx]

        assert not np.any(np.isnan(self.X_data))

        print(f"Data loaded successfully!")

    def dim_reduce(self, data):
        if self.X_data is None:
            print(f"Please load data first!")
            return None

        dict_algo = {
            "pca": PCA,
            'ica': FastICA
        }

        self.dr_algo = data['dr_algo']
        if self.dr_algo == None:  # Don't perform DR
            return None

        n_components = data['n_components']
        transformer = dict_algo[self.dr_algo](n_components=n_components)

        # Perform DR
        self.X_data = transformer.fit_transform(self.X_data)
        self.X_data_all = transformer.fit_transform(self.X_data_all)
        self._update_data_dim()

    def train_test_split(self, data):
        test_ratio = data
        train_ratio = 1 - test_ratio

        n_train = round(len(self.X_data) * train_ratio)
        self.X_train, self.y_train = self.X_data[:n_train,
                                                 :], self.y_data[:n_train]
        self.X_test, self.y_test = self.X_data[n_train:,
                                               :], self.y_data[n_train:]

    def get_data(self, dummy=None):
        self.status_data = {
            'dataset': self.dataset,
            'X_all': self.X_data_all,  # All data with background
            'y_all': self.y_data_all,
            'X_train': self.X_train,
            'y_train': self.y_train,
            'X_test': self.X_test,
            'y_test': self.y_test,
            'data_dim': self.data_dim,
            'dr_method': self.dr_algo,
            'origin_mean': self.origin_mean,
            'origin_std': self.origin_std
        }

        print(f"Data shape: {self.data_dim}")

        return self.status_data

    def _ignore_background(self, X_data, y_data):
        no_background_data = []
        no_background_label = []

        for idx, data in enumerate(X_data):
            if y_data[idx] != 0:
                no_background_data.append(data)
                no_background_label.append(y_data[idx])

        new_x_data = np.array(no_background_data)
        new_y_data = np.array(no_background_label)

        return new_x_data, new_y_data

    def _update_data_dim(self):
        self.data_dim['n_samples'] = self.y_data.shape[0]
        self.data_dim['n_bands'] = self.X_data.shape[-1]


if __name__ == "__main__":
    dp = DataHolder("indian_pines")
    # Ask dp to load data, the other arg remains None since it's not used.
    dp.main("load_data", None)
    # Ask dp to perform dimension reduction by the given args.
    dp.main("dim_reduce", {'dr_algo': "ica", 'n_components': 10})
    print(dp.get_data())
