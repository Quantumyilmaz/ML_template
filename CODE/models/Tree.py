from sklearn.tree import DecisionTreeRegressor as DecisionTree

class Tree(DecisionTree):
    # TODO: add Model as parent class
    def __init__(self, *args,**kwargs) -> None:
            super().__init__(*args,**kwargs)
