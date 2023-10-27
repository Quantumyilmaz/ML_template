import torch,optuna
from tqdm import tqdm
import numpy as np
import os,sys
from typing import Optional,Union
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from configs.ML_config import cfg
from datetime import datetime
from copy import deepcopy


np.random.seed = 42
torch.manual_seed(42)

class Trainer(object):  
    def __init__(self, 
                train_dataloader, 
                val_dataloader,
                model,
                optimizer, scheduler,
                criterion,
                ckpt_path,
                output_dir,
                logger,
                ):
        self.train_dataloader, self.val_dataloader, self.model, self.optimizer, self.scheduler,\
            self.criterion, self.ckpt_path, self.output_dir, self.logger = \
            train_dataloader, val_dataloader, model, optimizer, scheduler, \
            criterion, ckpt_path, output_dir, logger
        self.start_epoch = 1
        if self.ckpt_path is not None:
            self.load_ckpt()

        self.best_state = None
        self.best_val = None


    def train_it(self, batch):
        self.model.train()
        X , y = batch['X'], batch['y']
        if cfg.TRAIN.NGPUS > 0:
            X , y = X.cuda(), y.cuda()
        assert cfg.TRAIN.NGPUS == 0 or (X.is_cuda and y.is_cuda), 'GPU is not used for training.'
        self.optimizer.zero_grad()
        y_pred = self.model(X)
        loss = self.criterion(y_pred, y)
        loss.backward()
        self.optimizer.step()
        return loss

    def eval_epoch(self,trial=None,epoch=None):
        losses = []
        self.model.eval()
        for it, batch in tqdm(enumerate(self.val_dataloader), total=len(self.val_dataloader)):
            X_val, y_val = batch['X'], batch['y']
            if cfg.TRAIN.NGPUS > 0:
                X_val, y_val = X_val.cuda(), y_val.cuda()
            assert cfg.TRAIN.NGPUS == 0 or (X_val.is_cuda and y_val.is_cuda), 'GPU is not used for validation.'
            y_val_pred = self.model(X_val)
            loss = self.criterion(y_val_pred , y_val)
            if cfg.TRAIN.NGPUS > 0:
                loss = loss.cpu().numpy()
            losses.append(loss)
        loss = np.array(losses).mean()
        self.logger.info('Validation loss = {}.'.format(loss))

        if trial is not None:
            assert epoch is not None, 'Epoch must be specified for hyperparameter tuning.'
            trial.report(loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return loss
    
    def save_ckpt(self, epoch):
        ckpt = {'epoch': epoch,
                'model_state': self.model.state_dict(),
                'optimizer_state' : self.optimizer.state_dict()}
        ckpt_path = os.path.join(self.output_dir, 'ckpt_{}.pth'.format(epoch))
        torch.save(ckpt, ckpt_path)
        self.logger.info('Checkpoint saved to {}'.format(ckpt_path))

    def load_ckpt(self):
        ckpt = torch.load(self.ckpt_path)
        self.logger.info('Loading checkpoint from {}...'.format(self.ckpt_path))

        self.start_epoch = ckpt['epoch']+1 if 'epoch' in ckpt.keys() else 1
        self.logger.info('Starting epoch = {}.'.format(self.start_epoch))
        
        if self.model is not None and ckpt['model_state'] is not None:
            self.model.load_state_dict(ckpt['model_state'], strict=False)
            self.logger.info('Model state_dict loaded.')

        if self.optimizer is not None and ckpt['optimizer_state'] is not None:
            self.optimizer.load_state_dict(ckpt['optimizer_state'])
            self.logger.info('Optimizer state_dict loaded.')

    def train(self,trial=None):
        # TODO: set model params to the best ones in terms of validation loss
        self.logger.info(f'Starting training... {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}')

        assert isinstance(cfg.TRAIN.EPOCHS, int), 'Number of epochs must be an integer.'
        assert cfg.TRAIN.EPOCHS > 0, 'Number of epochs must be positive.'

        assert isinstance(cfg.TRAIN.VAL_FREQUENCY, Optional[int]), 'Validation frequency must be an integer.'
        assert cfg.TRAIN.VAL_FREQUENCY is None or cfg.TRAIN.VAL_FREQUENCY >= 0, 'Validation frequency must be non-negative.'

        assert isinstance(cfg.TRAIN.SAVE_FREQUENCY, Optional[int]), 'Save frequency must be an integer or None.'
        assert cfg.TRAIN.SAVE_FREQUENCY is None or cfg.TRAIN.SAVE_FREQUENCY >= 0, 'Save frequency must be non-negative.'

        assert isinstance(cfg.TRAIN.SAVE_THRESHOLD, Optional[Union[float,int]]), 'Validation threshold must be a float, integer or None.'

        assert isinstance(cfg.TRAIN.EARLY_STOPPING, Optional[int]), 'Early stopping must be an integer or None.'
        assert cfg.TRAIN.EARLY_STOPPING is None or cfg.TRAIN.EARLY_STOPPING >= 0, 'Early stopping must be non-negative.'
        
        # assert not self.tuning_mode or trial is not None, 'Trial must be specified for hyperparameter tuning.'

        loss=None
        eval_loss = None
        eval_threshold = cfg.TRAIN.SAVE_THRESHOLD
        early_stopping = cfg.TRAIN.EARLY_STOPPING if cfg.TRAIN.EARLY_STOPPING else -1

        learning_rate = self.optimizer.param_groups[0]['lr']
        for epoch in tqdm(range(self.start_epoch, cfg.TRAIN.EPOCHS+1)):
            if early_stopping == 0:
                self.logger.info('Early stopping.')
                break
            temp = []
            if learning_rate != self.optimizer.param_groups[0]['lr']:
                learning_rate = self.optimizer.param_groups[0]['lr']
                self.logger.info(f'Learning rate set to: {learning_rate}')
            for it, train_batch in tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader)):
                temp.append(self.train_it(train_batch))
                if loss is None or temp[-1] < loss:
                    loss = temp[-1]
                    self.logger.info('Batch loss improved = %0.5f.' %loss)
            training_loss = sum(temp)/len(temp)
            
            self.logger.info(f'Training loss: {training_loss}')
            # current_time = datetime.now().strftime("%H:%M:%S")
            # self.logger.info(f'Current Time = {current_time}')

            if trial is not None or cfg.TRAIN.VAL_FREQUENCY and (epoch % cfg.TRAIN.VAL_FREQUENCY) == 0:# and training_loss<cfg.TRAIN.SAVE_THRESHOLD:
                with torch.no_grad():
                    self.logger.info('Evaluation epoch {}...'.format(epoch))
                    temp_eval_loss = self.eval_epoch(trial=trial,epoch=epoch)
                    early_stopping = max(early_stopping,0) - 1
                    if eval_loss is None or temp_eval_loss < eval_loss:
                        eval_loss = temp_eval_loss
                        # self.logger.info('Validation loss improved = %0.5f.' %eval_loss)
                        if cfg.TRAIN.SAVE_BEST_STATE or trial is not None:
                            self.best_state = deepcopy(self.model.state_dict())
                            self.best_val = eval_loss
                        early_stopping = cfg.TRAIN.EARLY_STOPPING if cfg.TRAIN.EARLY_STOPPING else -1
                if eval_threshold is not None and temp_eval_loss<eval_threshold:
                    eval_threshold = temp_eval_loss
                    self.save_ckpt(epoch)
            if cfg.TRAIN.SAVE_FREQUENCY and (epoch % cfg.TRAIN.SAVE_FREQUENCY) == 0:
                self.save_ckpt(epoch)
            if cfg.OPTIMIZER.LR_SCHEDULER == 'StepLR':
                self.scheduler.step()
            elif cfg.OPTIMIZER.LR_SCHEDULER == 'ReduceLROnPlateau':
                self.scheduler.step(training_loss)
            elif cfg.OPTIMIZER.LR_SCHEDULER == 'custom':
                self.optimizer = self.scheduler(self.optimizer,training_loss,self.logger)

        if cfg.TRAIN.SAVE_BEST_STATE:
            self.logger.info('Saving best model with validation loss = {}.'.format(self.best_val))
            ckpt = {'epoch': None,
                    'model_state': self.best_state,
                    'optimizer_state' : None}
            ckpt_path = os.path.join(self.output_dir, 'best.pth')
            torch.save(ckpt, ckpt_path)
            self.logger.info('Best model saved to {}'.format(ckpt_path))
        self.logger.info('Training ended')
