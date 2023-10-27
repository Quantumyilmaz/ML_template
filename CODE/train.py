import pickle,os,sys,logging
from typing import Optional
import numpy as np

from sklearn.model_selection import train_test_split

from configs.ML_config import cfg
from utils.dataset import DatasetManager,TaskDataset
from utils.trainer import Trainer
from utils.misc import make_dir

from models.NN import FFNN, PFNN
from models.Linear import OLS, ElasticNET
from models.Ensemble import RF, GB

import torch
from torchsummary import summary
from torch.nn import DataParallel,MSELoss,L1Loss
from torch.optim import Adam,SGD
from torch.optim.lr_scheduler import StepLR,ReduceLROnPlateau
from torch.utils.data import DataLoader

NN_models = ['ffnn','pfnn']


def create_logger(log_path):
    log_format = '%(message)s'
    # logging.basicConfig(level=logging.DEBUG, format=log_format, filename=log_path)
    logging.basicConfig(level=logging.INFO, format=log_format, filename=log_path)
    console = logging.StreamHandler()
    # console.setLevel(logging.DEBUG)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(log_format))
    logging.getLogger(__name__).addHandler(console)
    return logging.getLogger(__name__)

def load_model(params_path:Optional[str]=None, logger=None,*args,**kwargs):

    if cfg.MODEL.BACKBONE.lower() in NN_models:

        if cfg.MODEL.BACKBONE.lower() == 'ffnn': model = FFNN
        elif cfg.MODEL.BACKBONE.lower() == 'pfnn': model = PFNN

        kwargs.update({
                        'hidden':kwargs.get('hidden',cfg.MODEL.H_DIMS),
                        'activation':kwargs.get('activation',cfg.MODEL.ACTIVATION),
                        'batch_norm':kwargs.get('batch_norm',cfg.MODEL.BATCH_NORM),
                        'dropout':kwargs.get('dropout',cfg.MODEL.DROPOUT)
                        }
                    )
        
    elif cfg.MODEL.BACKBONE.lower() == 'ols':
        model = OLS
    elif cfg.MODEL.BACKBONE.lower() == 'elasticnet':
        model = ElasticNET
    elif cfg.MODEL.BACKBONE.lower() == 'randomforest':
        model = RF
    elif cfg.MODEL.BACKBONE.lower() == 'gradientboost':
        model = GB
    else:
        raise NotImplementedError

    if params_path is not None:
        with open(params_path, 'rb') as f:
            return model(**pickle.load(f))

    return model(*args,**kwargs)

def load_optimizer(model):
    if cfg.OPTIMIZER.METHOD == 'adam':
        optimizer = Adam(model.parameters(), lr=cfg.OPTIMIZER.LR)
    elif cfg.OPTIMIZER.METHOD == 'sgd':
        optimizer = SGD(model.parameters(), lr=cfg.OPTIMIZER.LR)
    else:
        raise NotImplementedError
    return optimizer

def load_scheduler(optimizer,logger):
    if cfg.OPTIMIZER.LR_SCHEDULER.lower() == 'steplr':
        scheduler = StepLR(optimizer, step_size=cfg.OPTIMIZER.STEP_SIZE, gamma=cfg.OPTIMIZER.DECAY_FACTOR)
        logger.info('Using scheduler: StepLR')
    elif cfg.OPTIMIZER.LR_SCHEDULER.lower() =='reducelronplateau':
        scheduler = ReduceLROnPlateau(optimizer, factor=cfg.OPTIMIZER.DECAY_FACTOR,patience=cfg.OPTIMIZER.SCHEDULER_PATIENCE,min_lr=cfg.OPTIMIZER.MIN_LR,verbose=1)
        logger.info('Using scheduler: ReduceLROnPlateau')
    elif cfg.OPTIMIZER.LR_SCHEDULER == 'custom':...
        # scheduler = CustomLR
        # logger.info('Using scheduler: CustomLR')
    else:
        scheduler = None

    return scheduler

def load_criterion():  

    if cfg.MODEL.LOSS.lower() == 'mse':
        criterion = MSELoss()
    elif cfg.MODEL.LOSS.lower() == 'mae':
        criterion = L1Loss()
    else:
        raise NotImplementedError('Supported loss functions: MSE, MAE')
    return criterion

def load_data(X,y,test_size,shuffle,batch_size_train,batch_size_val,batch_shuffle_train,batch_shuffle_val):

    assert batch_size_train is not None and batch_size_val is not None, 'batch_size_train or batch_size_val must be specified.'
    
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, 
        test_size = test_size, 
        random_state = 42, 
        shuffle = shuffle)
    
    train_dataset = TaskDataset(X=X_train,y=y_train)
    val_dataset = TaskDataset(X=X_valid,y=y_valid)

    if batch_size_train == -1: batch_size_train = len(train_dataset)
    if batch_size_val == -1: batch_size_val = len(val_dataset)

    assert batch_size_train>0 and batch_size_val>0, 'batch_size_train and batch_size_val must be positive.'
    
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size_train, shuffle=batch_shuffle_train, pin_memory=bool(cfg.TRAIN.NGPUS))
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size_val, shuffle=batch_shuffle_val, pin_memory=bool(cfg.TRAIN.NGPUS))

    return train_dataloader , val_dataloader

def load_trainer(model,X_train,y_train,logger,is_tuning=False):
    if cfg.TRAIN.NGPUS > 0:
        model = model.cuda()
        gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] \
            if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
        # if logger is not None:
        #     logger.info('GPU training enabled. CUDA visible devices = %s' % gpu_list)
    if cfg.TRAIN.NGPUS > 1:
        model = DataParallel(model.cuda())

    optimizer = load_optimizer(model=model)
    scheduler = load_scheduler(optimizer=optimizer,logger=logger)
    criterion = load_criterion()

    output_dir = cfg.PATH.MODELS+f"/{model.name}/ckpts"
    make_dir(output_dir)

    train_dataloader, val_dataloader = load_data(X=X_train, 
                                                y=y_train, 
                                                test_size=cfg.TRAIN.VALIDATION_SPLIT if not is_tuning else cfg.TUNE.VALIDATION_SPLIT,
                                                shuffle=cfg.TRAIN.SHUFFLE,
                                                batch_size_train=cfg.TRAIN.BATCH_SIZE_TRAIN,
                                                batch_size_val=cfg.TRAIN.BATCH_SIZE_VAL,
                                                batch_shuffle_train=cfg.TRAIN.BATCH_SHUFFLE_TRAIN,
                                                batch_shuffle_val=cfg.TRAIN.BATCH_SHUFFLE_VAL)
    trainer = Trainer(model=model, 
                        optimizer=optimizer, 
                        scheduler=scheduler, 
                        criterion=criterion, 
                        train_dataloader=train_dataloader, 
                        val_dataloader=val_dataloader, 
                        ckpt_path=None, output_dir=output_dir, logger=logger,
                        )
    return trainer


def main_it(X_train,y_train):
    # TODO: set numpy seed
    np.random.seed = 42
    torch.manual_seed(42)

    model = load_model()


    if cfg.MODEL.BACKBONE.lower() in NN_models:

        summary(model.float(),input_size=(30,),device='cpu')
        model.double()

        log_dir = cfg.PATH.MODELS+f"/{model.name}"
        make_dir(log_dir)
        log_name = f"log_train"
        logger = create_logger(log_path=log_dir+f"/{log_name}.txt")

        trainer = load_trainer(model=model,
                                X_train=X_train,
                                y_train=y_train,
                                logger=logger,
                                is_tuning=False)

        trainer.train()
    
    else:
        # TODO: complete it for non neural network models
        pass

def main():
    # TODO: go over all datasets if multiple datasets are available
    dataset_manager = DatasetManager()
    dataset_manager.next()
    main_it(X_train=dataset_manager.X_train,y_train=dataset_manager.y_train)


if __name__ == "__main__":
    np.random.seed = 42
    torch.manual_seed(42)
    main()
