from sklearn.linear_model import LinearRegression,ElasticNet
from sklearn.pipeline import Pipeline
from models.__model import Model

class OLS(Model):
        def __init__(self):
                super().__init__(model_name='OLS',model=Pipeline, steps=[("model", LinearRegression())])

        def get_params(self):
                # TODO: Do we need Pipeline?
                return {'a':self.model.steps[0][1].coef_,'b':self.model.steps[0][1].intercept_}
        
        def set_params(self, params:dict):
                # TODO: Do we need Pipeline?
                self.model.steps[0][1].coef_ = params['a']
                self.model.steps[0][1].intercept_ = params['b']

class ElasticNET(ElasticNet):
        # TODO: add Model as parent class
        def __init__(self, *args,**kwargs) -> None:
                super().__init__(*args,**kwargs)