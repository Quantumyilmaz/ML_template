from sklearn.ensemble import RandomForestRegressor as RandomForest
from sklearn.ensemble import HistGradientBoostingRegressor as GradientBoost

class RF(RandomForest):
        # TODO: add Model as parent class
        def __init__(self, *args,**kwargs) -> None:
                super().__init__(*args,**kwargs)

class GB(GradientBoost):
        # TODO: add Model as parent class
        def __init__(self, *args,**kwargs) -> None:
                super().__init__(*args,**kwargs)