from configs.preprocess_config import cfg

### Dataset

cfg.DATASET.SCALER = "Standard" # "Standard" or "MinMax" (for neural nets) or None
cfg.DATASET.PCA = None # Set to 0 or None to disable PCA.

### Model
cfg.MODEL.BACKBONE = 'FFNN' # "OLS", "ElasticNet", "FFNN", "PFNN", "RandomForest", "GradientBoost"
cfg.MODEL.LOAD_FILENAME = None # name of the file to load the model from or None

### MODEL - NN
cfg.MODEL.LOSS = 'MSE'
cfg.MODEL.H_DIMS = [1000] # number of neurons in each hidden layer
cfg.MODEL.ACTIVATION = ['relu'] # activation function for each layer
cfg.MODEL.BATCH_NORM = False # whether to use batch normalization for each layer
cfg.MODEL.DROPOUT = [0] # dropout rate for each layer


### Tuning
cfg.TUNE.NHIDDEN_MIN_MAX = [1,3,1] # [min, max, step]
cfg.TUNE.NEURON_MIN_MAX = [32,1024,32] # [min, max, step]
cfg.TUNE.ACTIVATION = ["relu", "tanh", "sigmoid"]
cfg.TUNE.DROPOUT_MIN_MAX = [0, 0.5]
cfg.TUNE.NSTUDIES = 2
cfg.TUNE.NTRIALS = 1
cfg.TUNE.TIMEOUT = None
cfg.TUNE.TERMINATOR = False # EXPERIMENTAL. whether to use the terminator callback
cfg.TUNE.PRUNER = "hyperband" # "default" or "hyperband"
cfg.TUNE.SAMPLER = ... # TODO: right now only TPESampler is implemented by default

cfg.TUNE.VALIDATION_SPLIT = 0.2

### Train - Training
cfg.TRAIN.NMODELS = 1 # number of models to train. Intended for use of ensemble of NNs.
cfg.TRAIN.NGPUS = 0 # set to 0 to use CPU. Otherwise, set to the number of GPUs to use for NN training.
cfg.TRAIN.EPOCHS = 1
cfg.TRAIN.BATCH_SIZE_TRAIN = -1 # set to -1 to use the entire training set
cfg.TRAIN.SHUFFLE = False
cfg.TRAIN.BATCH_SHUFFLE_TRAIN = False
cfg.TRAIN.EARLY_STOPPING = None # set to None or 0 to disable. Otherwise, set to the number of validation epochs to wait before stopping.

### Train - Validation
cfg.TRAIN.VALIDATION_SPLIT = 0.1
cfg.TRAIN.BATCH_SIZE_VAL = -1 # set to -1 to use the entire validation set
cfg.TRAIN.BATCH_SHUFFLE_VAL = False
cfg.TRAIN.VAL_FREQUENCY = None # set to None or 0 to disable. Otherwise, set to the number of epochs to wait before validating the model.

### Train - Saving
cfg.TRAIN.SAVE_BEST_STATE = True # whether to save the best state of the model in terms of validation loss
cfg.TRAIN.SAVE_FREQUENCY = None # set to None or 0 to disable. Otherwise, set to the number of epochs to wait before saving the model.
cfg.TRAIN.SAVE_THRESHOLD = None # set to None or 0 to disable. Otherwise, set to the validation loss threshold to wait before saving the model.

### Optimizer
cfg.OPTIMIZER.METHOD = 'adam'
cfg.OPTIMIZER.LR = 0.003
cfg.OPTIMIZER.LR_SCHEDULER = 'ReduceLROnPlateau' # 'StepLR', 'ReduceLROnPlateau'
cfg.OPTIMIZER.SCHEDULER_PATIENCE = 50 # for ReduceLROnPlateau
cfg.OPTIMIZER.DECAY_FACTOR = 0.5 # for ReduceLROnPlateau and StepLR
cfg.OPTIMIZER.MIN_LR = 1e-8

### TEST/EVAL
cfg.TEST.STUFF = ...
