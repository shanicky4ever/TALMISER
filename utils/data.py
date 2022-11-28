import pandas as pd
import torch
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from utils.helper_function import save_obj, load_obj
from mutator import get_mutator
import datetime


class tabularDataHandler:
    def __init__(self, base_config, encode=False):
        self.base_config = base_config
        self._get_data()
        self._encode_data(encode)
        self._determine_label()
        self.mutator = get_mutator('tabular')(
            self.data, base_config['mutation_bound'])

    def _get_data(self):
        self.data = pd.read_csv(
            self.base_config['data_file'], na_filter=True, skipinitialspace=True)
        self.data.replace('?', None, inplace=True)
        self.__specific_change()
        for dr in self.base_config['drop_col']:
            self.data.drop(dr, axis=1, inplace=True)
        self.data.dropna(inplace=True)

    def __specific_change(self):
        if self.base_config['data_name'] == 'census':
            self.data.replace('<=50K', "LessThan50K", inplace=True)
            self.data.replace('>50K', "MoreThan50K", inplace=True)
        if self.base_config['data_name'] == 'compas':
            self.data = self.data[
                (self.data['days_b_screening_arrest'] <= 30) &
                (self.data['days_b_screening_arrest'] >= -30) &
                (self.data['is_recid'] != -1) &
                (self.data['c_charge_degree'] != "0") &
                (self.data['score_text'] != "N/A")
            ]
            in_custody = [datetime.datetime.strptime(s.split()[0], '%Y-%m-%d').date()
                          for s in self.data['c_jail_in']]
            out_custody = [datetime.datetime.strptime(s.split()[0], '%Y-%m-%d').date()
                           for s in self.data['c_jail_out']]
            custody_during = [(o-i).days
                              for i, o in zip(in_custody, out_custody)]
            self.data['custody_during'] = custody_during
            self.data.drop(['c_jail_in', 'c_jail_out'], axis=1, inplace=True)

    def _encode_data(self, encode):
        le = preprocessing.LabelEncoder()
        self.classes = load_obj(
            self.base_config['classes_file']) if not encode else {}
        for col in self.data.columns:
            if not pd.api.types.is_numeric_dtype(self.data[col]):
                if encode:
                    self.data[col] = le.fit_transform(self.data[col])
                    self.classes[col] = le.classes_.tolist()
                else:
                    le.classes_ = self.classes[col]
                    self.data[col] = le.transform(self.data[col])
        if encode:
            save_obj(self.classes, self.base_config['classes_file'])

    def get_decode_data(self, col_name, idx):
        if col_name not in self.classes:
            return str(idx)
        return self.classes[col_name][idx]

    def _determine_label(self):
        label_col = self.base_config['label_col']
        self.label = self.data[label_col]
        self.data.drop(label_col, axis=1, inplace=True)

    def get_label_name(self):
        return self.base_config['label_col']

    def get_label_unique(self):
        return self.label.unique()

    def get_attr_unique(self, attr_name):
        return self.data[attr_name].unique()

    def split_train_val(self):
        return train_test_split(
            self.data, self.label, test_size=self.base_config['val_ratio'], random_state=42)

    def get_input_dim(self):
        return self.data.shape[1]

    def _data_to_dataloader(self, x, y, batch_size, shuffle=False):
        return torch.utils.data.DataLoader(
            TabularDataset(x.values, y.values),
            batch_size=batch_size, shuffle=shuffle)

    def generate_dataloader_for_DNN(self, data=None, batch_size=32, split=True):
        if data is None:
            data = self.data
        if not split:
            return self._data_to_dataloader(data, self.label,
                                            batch_size=batch_size)

        X_train, X_val, y_train, y_val = self.split_train_val()
        train_loader = self._data_to_dataloader(
            X_train, y_train, batch_size=batch_size, shuffle=True)
        val_loader = self._data_to_dataloader(
            X_val, y_val, batch_size=batch_size)
        return train_loader, val_loader

    def get_attribute_index(self, attribute_name):
        if attribute_name in self.data.columns:
            return self.data.columns.get_loc(attribute_name)
        return None

    def get_data(self):
        return self.data

    def get_dtmc_folder(self):
        return self.base_config['dtmc_folder']

    def get_dataset_name(self):
        return self.base_config['data_name']


class TabularDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
