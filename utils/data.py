import pandas as pd
import torch
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from utils.helper_function import save_obj


class tabularDataHandler:
    def __init__(self, base_config):
        self.base_config = base_config
        self._get_data()
        self._encode_data()
        self._determine_label()
        self._split_train_val()

    def _get_data(self):
        self.data = pd.read_csv(
            self.base_config['data_file'], na_filter=True, skipinitialspace=True)
        self.data.replace('?', 'NaN', inplace=True)
        self.data.dropna(inplace=True)
        for dr in self.base_config['drop_col']:
            self.data.drop(dr, axis=1, inplace=True)

    def _encode_data(self):
        le = preprocessing.LabelEncoder()
        classes = {}
        for col in self.data.columns:
            if not pd.api.types.is_numeric_dtype(self.data[col]):
                self.data[col] = le.fit_transform(self.data[col])
                classes[col] = le.classes_
        save_obj(classes, self.base_config['classes_file'])

    def _determine_label(self):
        label_col = self.base_config['label_col']
        self.label = self.data[label_col]
        self.data.drop(label_col, axis=1, inplace=True)

    def _split_train_val(self):
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.data, self.label, test_size=self.base_config['val_ratio'], random_state=42)

    def get_input_dim(self):
        return self.data.shape[1]

    def generate_dataloader_for_DNN(self):
        train_data = TabularDataset(self.X_train.values, self.y_train.values)
        val_data = TabularDataset(self.X_val.values, self.y_val.values)

        train_loader = torch.utils.data.DataLoader(
            dataset=train_data, batch_size=32, shuffle=True)
        val_loader = torch.utils.data.DataLoader(
            dataset=val_data, batch_size=32)
        return train_loader, val_loader


class TabularDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
