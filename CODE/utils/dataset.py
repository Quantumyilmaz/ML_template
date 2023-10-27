import re

import pandas as pd
import numpy as np
from configs.ML_config import cfg

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

import torch
from torch.utils.data import Dataset

class DatasetManager:

    def __init__(self):
        
        # assigned in get_datasets
        self.__reset()
        self.datasets = self.get_datasets()

        self.columns_to_standardize = ...
    
    def next(self):
        self.X_train, self.y_train, self.X_test = self.datasets.__next__()
        return self.X_train, self.y_train, self.X_test

    def get_datasets(self): yield from map(self.__get_dataset,self.forecast_dates)

    def __get_dataset(self):
        
        self.__reset()
        
        X_train = ...
        y_train = ...
        X_test = ...

        len_X_train = len(X_train)
        X = pd.concat([X_train,X_test])

        if cfg.DATASET.SCALER is None: self.scaler_X = None ; self.scaler_y = None
        else:
            self.scaler_X, self.scaler_y = self.__get_scaler()
            X = self.scaler_X.fit_transform(X)
            assert isinstance(X, np.ndarray), "X must be a numpy array."

            y_train = self.scaler_y.fit_transform(y_train.values[:,None])
            
        if not cfg.DATASET.PCA: self.PCA = None
        else:
            self.PCA = PCA(n_components = cfg.DATASET.PCA, random_state = 42)
            X = self.PCA.fit_transform(X)
        
        X_train, X_test = X[:len_X_train], X[len_X_train:]

        return X_train, y_train, X_test
    

    def __get_scaler(self):

        if cfg.DATASET.SCALER.lower() == "standard":
            scaler = StandardScaler
            scaler_kwargs = {}
        elif cfg.DATASET.SCALER.lower() == "minmax":
            scaler = MinMaxScaler
            scaler_kwargs = {"feature_range": (-1,1)}
        else:
            raise NotImplementedError("Only 'standard' and 'minmax' are supported.")
        
        # define the pipeline to scale and encode
        pipeline_scaler = Pipeline(
                [("coltransformer", ColumnTransformer(
                    transformers = [
                        ("standardize", Pipeline([("scale", scaler(**scaler_kwargs))]), self.columns_to_standardize)
                ],
                    remainder = "passthrough"),
                )]
            )

        return pipeline_scaler, scaler(**scaler_kwargs)
    
    def __reset(self):

        # assigned in get_datasets
        self.scaler_X = None ; self.scaler_y = None ; self.PCA = None
        self.X_train, self.y_train, self.X_test = None, None, None
    
class TaskDataset(Dataset):
    
    def __init__(self, X, y):
        assert y is not None, "y is None."
        assert len(y.shape)==2, "Wrong shape of y."
        assert y.shape[1]==1, "y must be a column vector."
        
        self.X = torch.from_numpy(X) ; self.y = torch.from_numpy(y)

    def __getitem__(self, index):
        data_dict = {}

        data_dict['X'] = self.X[index]
        data_dict['y'] = self.y[index] 

        return data_dict

    def __len__(self):
        return len(self.y)