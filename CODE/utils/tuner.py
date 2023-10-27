import optuna,pickle,torch,os,logging,datetime
from optuna.samplers import TPESampler
from optuna.terminator import TerminatorCallback
from tqdm import trange
import numpy as np


from configs.ML_config import cfg
from utils.misc import get_unique_filename, make_dir
from train import load_model,load_trainer, create_logger


np.random.seed = 42
torch.manual_seed(42)

class OptunaTuner:
    def __init__(self,X_train,y_train) -> None:
        
        """
        Hyperparameter tuning for the FFNN model. Tunes
            - n_hidden: number of hidden layers
            - n_neurons_{i}: number of neurons in the i-th hidden layer
            - act_f_{i}: activation function of the i-th hidden layer
            - dropout_{i}: activation function of the i-th hidden layer
        """

        self.studies = []
        self.best_trials = {}
    
        if cfg.TUNE.PRUNER == 'hyperband':
            self.pruner = optuna.pruners.HyperbandPruner(min_resource = 2,
                                                    max_resource = cfg.TRAIN.EPOCHS,
                                                    reduction_factor = 2)
        elif cfg.TUNE.PRUNER == 'default':
            self.pruner = None
        else:
            raise NotImplementedError
        
        self.X_train, self.y_train = X_train, y_train
        
        self.trainer = None

        optuna.logging.enable_propagation()  # Propagate logs to the root logger.
        optuna.logging.disable_default_handler()  # Stop showing logs in sys.stderr.

        log_dir = cfg.PATH.MODELS+f"/{cfg.MODEL.BACKBONE}/optuna"
        make_dir(log_dir)
        log_name = f"log_tune"
        log_path = log_dir+f"/{log_name}.txt"
        logging.basicConfig(level=optuna.logging.DEBUG, format='%(message)s', filename=log_path)

        self.logger = create_logger(log_path=log_path)


    def tune(self):
        self.logger.info(f'Starting hyperparameter tuning... {datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")}')
        for i in trange(cfg.TUNE.NSTUDIES):
            self.optimize()
            self.logger.info(f"Episode {i} completed")
        directory = cfg.PATH.MODELS+f"/{self.trainer.model.name}/optuna"
        filename = get_unique_filename("best_trials",directory,save_with_dt=True)
        with open(f'{directory}/{filename}', 'wb') as f:
            pickle.dump(self.best_trials, f)
        self.logger.info("Tuning completed.")


    def optimize(self):
        """
        Creates a new study and optimizes the objective function wrt the hyperparameters.
        """
        study = optuna.create_study(direction = "minimize",
                                        sampler = TPESampler(seed = len(self.studies)),
                                        pruner = self.pruner)
        self.studies.append(study)

        callbacks = []
        if cfg.TUNE.TERMINATOR:
            callbacks.append(TerminatorCallback())

        self.studies[-1].optimize(self.objective,
                                    n_trials = cfg.TUNE.NTRIALS,
                                    timeout=cfg.TUNE.TIMEOUT,
                                    callbacks = callbacks)
        
        self.best_trials.update({self.studies[-1].best_trial.value:study.best_trial.params})



    def objective(self,trial):

        """
        Objective function for the hyperparameter tuning.
        """

        hyperparams = self.get_hyperparams(trial)
        model = load_model(**hyperparams)
        model.double()

        self.trainer = load_trainer(model=model,
                                    X_train=self.X_train,
                                    y_train=self.y_train,
                                    logger=self.logger,
                                    is_tuning=True)


        self.trainer.train(trial=trial)
        
        # # Load the best model state
        # self.logger.info(f'Best validation loss: {self.trainer.best_val}')
        # model.load_state_dict(self.trainer.best_state)

        # with torch.no_grad():
        #     model.eval().cpu()
        #     score = model(self.trainer.val_dataloader.dataset.X)
        #     score = self.trainer.criterion(score , self.trainer.val_dataloader.dataset.y).numpy()

        # self.logger.info(f"Validation score for tuning: {score}")
        
        return self.trainer.best_val
    
    
    def get_hyperparams(self,trial):
        """
        Returns a dictionary of suggested hyperparameters for the trial.
        """
        hyperparams_dict = {'hidden':[],'activation':[],'dropout':[]}

        hyperparams_dict.update({'batch_norm':trial.suggest_categorical("batch_norm", [True, False])})
        n_hidden = trial.suggest_int('n_hidden', *cfg.TUNE.NHIDDEN_MIN_MAX)
        
        for y in range(n_hidden):
            hyperparams_dict['hidden'].append(trial.suggest_int(f"n_neurons{y}", *cfg.TUNE.NEURON_MIN_MAX))
            hyperparams_dict['activation'].append(trial.suggest_categorical(f"act_f{y}", cfg.TUNE.ACTIVATION))
            hyperparams_dict['dropout'].append(trial.suggest_float(f"dropout{y}", *cfg.TUNE.DROPOUT_MIN_MAX, step=0.02))
        
        return hyperparams_dict