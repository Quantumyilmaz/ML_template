from utils.tuner import OptunaTuner
from utils.dataset import DatasetManager


if __name__ == "__main__":
    # TODO: go over all datasets
    dataset_manager = DatasetManager()
    dataset_manager.next()
    tuner = OptunaTuner(X_train=dataset_manager.X_train,y_train=dataset_manager.y_train)
    tuner.tune()