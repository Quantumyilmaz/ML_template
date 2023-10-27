from typing import Any, Mapping
import tensorflow as tf
import pickle
from configs.ML_config import cfg
from copy import deepcopy

import torch
from torch.nn import ReLU, LeakyReLU, Sigmoid, Tanh, ELU, ModuleList, BatchNorm1d, Dropout, Linear, Module
from torchsummary import summary
import os,sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


torch.manual_seed(42)


class FFNN(Module):
    def __init__(self,*args,**params):
        super().__init__()

        self.name = 'FFNN'

        input_size = 30
        output_size = 1
        self.layers = ModuleList()

        self.layer_dims = [input_size] + params['hidden'] + [output_size]
        self.activations = {'relu':ReLU,'sigmoid':Sigmoid,'tanh':Tanh,'elu':ELU}

        assert len(self.layer_dims) - 2 == len(params['activation']) == len(params['dropout'])\
            , "Number of hidden layers, activations and dropouts must be the same in the configuration!"
        assert isinstance(params['batch_norm'],bool), "Batch norm must be a boolean!"

        for i,k,m in zip(range(len(self.layer_dims)-2),params['activation'],params['dropout']):
            # layer
            self.layers.append(
                Linear(in_features=self.layer_dims[i],out_features=self.layer_dims[i+1],bias=True)
                )
            # activation
            self.layers.append(self.activations[k]())
            # batch norm
            if params['batch_norm']:
                self.layers.append(BatchNorm1d(self.layer_dims[i+1]))
            # dropout
            if m > 0:
                self.layers.append(Dropout(m))
        
        self.layers.append(Linear(in_features=self.layer_dims[-2],out_features=self.layer_dims[-1],bias=True))

        self.init_state_dict = deepcopy(self.state_dict())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def load_init_state_dict(self):
        self.load_state_dict(self.init_state_dict)

    def summary(self):
        summary(self.float(),input_size=(30,),device='cpu')


class PFNN(FFNN):
    """
        Partially trained FFNN. The last layer is trainable, the rest is frozen.
    """
    def __init__(self, *args, **params):
        super().__init__(*args, **params)

        for layer in self.layers[:-1]:
            if layer._get_name() == 'Linear':
                layer.requires_grad_(requires_grad=False)

