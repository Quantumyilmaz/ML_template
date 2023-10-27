from abc import ABC,abstractmethod
from typing import Optional
from datetime import datetime

import numpy as np

from configs.__config import cfg
from utils.misc import get_unique_filename, make_dir, logprint


date_format = "%Y_%m_%d_%H_%M_%S" #for saving models


class Model(ABC):
    def __init__(self, model_name, model, fname_fit='fit', fname_predict='predict', *args, **kwargs):
        self.model = model(*args, **kwargs)
        self.fname_fit = fname_fit
        self.fname_predict = fname_predict

        self._model_name = model_name
        assert self._model_name in cfg.SUPPORTED_MODELS, f"Model '{self._model_name}' not supported. Please choose one of {cfg.SUPPORTED_MODELS} or update __config.py."

        self.__save_dir = f"{cfg.PATH.OUTPUT}/models/{self._model_name}"
        make_dir(self.__save_dir)
        with open(f"{self.__save_dir}/save_history.txt", "w+") as f:
            f.write("")
    
    def fit(self,*args, **kwargs):
        return getattr(self.model, self.fname_fit)(*args, **kwargs)
    
    def predict(self, *args, **kwargs):
        return getattr(self.model, self.fname_predict)(*args, **kwargs)
    
    @abstractmethod
    def get_params(self):...

    @abstractmethod
    def set_params(self, params:dict):...

    def save(self, filename:Optional[str]=None):

        now = datetime.now()
        formatted_date = now.strftime(date_format)

        try:
            if filename is None:
                # filename = get_unique_filename(filename=f"{self._model_name}_1",dir=self.__save_dir)
                assert isinstance(cfg.MODEL.SAVE_WITH_DATE, bool), "cfg.MODEL.SAVE_WITH_DATE must be a boolean."
                filename = get_unique_filename(filename=self._model_name + cfg.MODEL.SAVE_WITH_DATE*f"_{formatted_date}",dir=self.__save_dir)

            np.savez(f"{self.__save_dir}/{filename}", **self.get_params())

            with open(f"{self.__save_dir}/save_history.txt", "a+") as f:
                f.write("\n" + filename)
        except:
            logprint("Could not save model.")

    def load(self, filename:Optional[str]=None):
        try:
            if filename is None:
                # load the last saved model
                with open(f"{self.__save_dir}/save_history.txt", "r") as f:
                    filename = f.readlines()[-1]

            self.set_params(np.load(filename))
        except:
            logprint("Could not load model.")