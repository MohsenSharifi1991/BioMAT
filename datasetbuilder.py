import numpy as np
import torch
from torch.utils.data import Dataset
from preprocessing.transformation.transformation import Transformation
import torch.nn.functional as F
from sklearn import preprocessing
import torch


class DataSetBuilder(Dataset):
    def __init__(self, x, y, labels, transform_method=None, scaler=None):
        self.x = x
        self.y = y
        self.labels = labels
        self.y_label = []

        self.transform_method = transform_method
        self.scaler = scaler
        self._preprocess()
        self.n_sample = len(y)

        # x = np.transpose(self.x, (0, 2, 1))
        self.x = torch.from_numpy(x).double()
        self.y = torch.from_numpy(self.y).double()

    def _run_label_encoding(self):
        le = preprocessing.LabelEncoder()
        y_label = le.fit_transform(self.labels[:, 0, 3])
        y_label = torch.as_tensor(y_label)
        self.y_label = y_label.to(torch.int64)

    def _preprocess(self):
        if self.transform_method['data_transformer_method'] is not None:
            self._run_transform()

    def _run_transform(self):
        transform_handler = Transformation(method=self.transform_method['data_transformer_method'], by=self.transform_method['data_transformer_by'])
        if self.scaler is None:
            self.scaler, self.x = transform_handler.run_transform(train=self.x, scaler_fit=self.scaler)
        else:
            self.x = transform_handler.run_transform(val=self.x, scaler_fit=self.scaler)


    def __len__(self):
        return self.n_sample

    def __getitem__(self, item):
        return self.x[item], self.y[item]