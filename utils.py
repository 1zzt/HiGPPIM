# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Evaluation of model performance."""
# pylint: disable= no-member, arguments-differ, invalid-name

import datetime
import random
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
import os
from dataset import *
from torch_geometric.loader import DataLoader
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, mean_squared_error, mean_absolute_error, accuracy_score, matthews_corrcoef, f1_score 
from scipy.stats import pearsonr, kendalltau, spearmanr



class EvalMeter(object):
    
    def __init__(self):
        self.mask = []
        self.y_pred = []
        self.y_true = []

    def update(self, y_pred, y_true):

        self.y_pred.append(y_pred.detach().cpu())
        self.y_true.append(y_true.detach().cpu())

    def pearson_r2(self):
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)
        if(len(y_pred.shape) == 2):
            y_pred = y_pred.squeeze(-1)
            y_true = y_true.squeeze(-1)
        out_pcc,_ = pearsonr(y_true, y_pred)
        return out_pcc

    def kendall(self):
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)
        out_kendall,_ = kendalltau(y_true, y_pred)
        return out_kendall

    def spearman(self):
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)
        out_spearman,_ = spearmanr(y_true, y_pred)
        return out_spearman

    def mae(self):
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)

        return mean_absolute_error(y_true, y_pred)

    def rmse(self):
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)
        
        return mean_squared_error(y_true, y_pred, squared=False)

    def acc(self):
        y_pred = torch.cat(self.y_pred, dim=0)
        y_pred = np.array(y_pred) > 0.5
        y_true = torch.cat(self.y_true, dim=0)

        return accuracy_score(y_true, y_pred)

    def mcc(self):
        y_pred = torch.cat(self.y_pred, dim=0)
       
        y_pred = np.array(y_pred) > 0.5

        y_true = torch.cat(self.y_true, dim=0)
        
        # print(y_true)
        return matthews_corrcoef(y_true, y_pred)
    
    def f1(self):
        y_pred = torch.cat(self.y_pred, dim=0)
        y_pred = np.array(y_pred) > 0.5
        y_true = torch.cat(self.y_true, dim=0)

        return f1_score(y_true, y_pred)

    def roc_auc_score(self):
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)
        if len(y_true.unique()) == 1:
            print('Warning: Only one class {} present in y_true for a task. '
                      'ROC AUC score is not defined in that case.'.format(y_true[0]))
            return None
        else:
            return roc_auc_score(y_true.long().numpy(), torch.sigmoid(y_pred).numpy())

    def pr_auc_score(self):
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)
        if len(y_true.unique()) == 1:
            print('Warning: Only one class {} present in y_true for a task. '
                      'PR AUC score is not defined in that case.'.format(y_true[0]))
            return None
        else:
            precision, recall, _ = precision_recall_curve(
                    y_true.long().numpy(), torch.sigmoid(y_pred).numpy())
            return auc(recall, precision)

    def compute_metric(self, metric_name):
        

        if metric_name == 'pcc':
            return self.pearson_r2()
        elif metric_name == 'kendall':
            return self.kendall()
        elif metric_name == 'spearman':
            return self.spearman()
        elif metric_name == 'mae':
            return self.mae()
        elif metric_name == 'rmse':
            return self.rmse()

        elif metric_name == 'acc':
            return self.acc()
        elif metric_name == 'mcc':
            return self.mcc()
        elif metric_name == 'f1':
            return self.f1()

        elif metric_name == 'roc_auc_score':
            return self.roc_auc_score()

        elif metric_name == 'pr_auc_score':
            return self.pr_auc_score()
        else:
            raise ValueError('got {}'.format(metric_name))


def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_data(configs):
    
    data_path = configs['data_path']
    task_name = configs['task_name']
    dataset_name = configs['dataset_name']
    batch_size = configs['batch_size']

        
    train_df = pd.read_csv(os.path.join(data_path, task_name, dataset_name+'_train.csv'))
    val_df = pd.read_csv(os.path.join(data_path, task_name, dataset_name+'_val.csv'))
    test_df = pd.read_csv(os.path.join(data_path, task_name, dataset_name+'_test.csv'))

    train_smiles = train_df[train_df.columns[0]].values
    train_labels = train_df[train_df.columns[-1]].values

    val_smiles = val_df[val_df.columns[0]].values
    val_labels = val_df[val_df.columns[-1]].values

    test_smiles = test_df[test_df.columns[0]].values
    test_labels = test_df[test_df.columns[-1]].values

    
    
    train_data = MyDataset(root = None, smiles = train_smiles, labels = train_labels)
    val_data = MyDataset(root = None, smiles = val_smiles, labels = val_labels)
    test_data = MyDataset(root = None, smiles = test_smiles, labels = test_labels)

    print(len(train_data))

    train_loader = DataLoader(train_data, batch_size = batch_size, shuffle=True, follow_batch=['x_pw','x_hyg','x_fg'])
    val_loader = DataLoader(val_data, batch_size = batch_size, shuffle=True, follow_batch=['x_pw','x_hyg','x_fg'])
    test_loader = DataLoader(test_data, batch_size = batch_size, shuffle=False, follow_batch=['x_pw','x_hyg','x_fg'])

    return train_loader, val_loader, test_loader



class EarlyStopping(object):
    """Early stop tracker

    Save model checkpoint when observing a performance improvement on
    the validation set and early stop if improvement has not been
    observed for a particular number of epochs.

    Parameters
    ----------
    mode : str
        * 'higher': Higher metric suggests a better model
        * 'lower': Lower metric suggests a better model
        If ``metric`` is not None, then mode will be determined
        automatically from that.
    patience : int
        The early stopping will happen if we do not observe performance
        improvement for ``patience`` consecutive epochs.
    filename : str or None
        Filename for storing the model checkpoint. If not specified,
        we will automatically generate a file starting with ``early_stop``
        based on the current time.
    metric : str or None
        A metric name that can be used to identify if a higher value is
        better, or vice versa. Default to None. Valid options include:
        ``'r2'``, ``'mae'``, ``'rmse'``, ``'roc_auc_score'``.

    Examples
    --------
    Below gives a demo for a fake training process.

    >>> import torch
    >>> import torch.nn as nn
    >>> from torch.nn import MSELoss
    >>> from torch.optim import Adam
    >>> from dgllife.utils import EarlyStopping

    >>> model = nn.Linear(1, 1)
    >>> criterion = MSELoss()
    >>> # For MSE, the lower, the better
    >>> stopper = EarlyStopping(mode='lower', filename='test.pth')
    >>> optimizer = Adam(params=model.parameters(), lr=1e-3)

    >>> for epoch in range(1000):
    >>>     x = torch.randn(1, 1) # Fake input
    >>>     y = torch.randn(1, 1) # Fake label
    >>>     pred = model(x)
    >>>     loss = criterion(y, pred)
    >>>     optimizer.zero_grad()
    >>>     loss.backward()
    >>>     optimizer.step()
    >>>     early_stop = stopper.step(loss.detach().data, model)
    >>>     if early_stop:
    >>>         break

    >>> # Load the final parameters saved by the model
    >>> stopper.load_checkpoint(model)
    """
    def __init__(self, patience=10, filename=None, mode = 'higher'):
        if filename is None:
            dt = datetime.datetime.now()
            filename = 'early_stop_{}_{:02d}-{:02d}-{:02d}.pth'.format(
                dt.date(), dt.hour, dt.minute, dt.second)


        self.mode = mode
        self.patience = patience
        
        self.counter = 0

        self.timestep = 0
        self.filename = filename

        self.best_score_loss = None
        self.best_score_metric = None

        self.early_stop = False

    def _check_higher(self, score, prev_best_score):

        return score > prev_best_score

    def _check_lower(self, score, prev_best_score):

        return score < prev_best_score

    def step(self, scores, model):
        """Update based on a new score.

        The new score is typically model performance on the validation set
        for a new epoch.

        Parameters
        ----------
        score : float
            New score.
        model : nn.Module
            Model instance.

        Returns
        -------
        bool
            Whether an early stop should be performed.
        """
        is_save = False
        loss_score,  metric_score= [*scores]
        self.timestep += 1


        if self.best_score_loss is None and self.best_score_metric is None:
            self.best_score_loss = loss_score
            self.best_score_metric = metric_score
            self.save_checkpoint(model)

        else:
            
            if self.mode == 'higher':

                if self._check_lower(loss_score, self.best_score_loss):
                    self.best_score_loss = loss_score
                    self.counter = 0
                    self.save_checkpoint(model)
                    is_save = True
                
                if self._check_higher(metric_score, self.best_score_metric):
                    self.best_score_metric = metric_score
                    self.counter = 0
                    if is_save == False:
                        self.save_checkpoint(model)
                
                if (self._check_lower(loss_score, self.best_score_loss) == False) and (self._check_higher(metric_score, self.best_score_metric) == False):
                    self.counter += 1
                    print(
                        f'EarlyStopping counter: {self.counter} out of {self.patience}')
                    if self.counter >= self.patience:
                        self.early_stop = True
            
            else:
                
                if self._check_lower(loss_score, self.best_score_loss):
                    self.best_score_loss = loss_score
                    self.counter = 0
                    self.save_checkpoint(model)
                    is_save = True
                
                if self._check_lower(metric_score, self.best_score_metric):
                    self.best_score_metric = metric_score
                    self.counter = 0
                    if is_save == False:
                        self.save_checkpoint(model)
                
                if (self._check_lower(loss_score, self.best_score_loss) == False) and (self._check_lower(metric_score, self.best_score_metric) == False):
                    self.counter += 1
                    print(
                        f'EarlyStopping counter: {self.counter} out of {self.patience}')
                    if self.counter >= self.patience:
                        self.early_stop = True

        return self.early_stop

    def save_checkpoint(self, model):
        '''Saves model when the metric on the validation set gets improved.

        Parameters
        ----------
        model : nn.Module
            Model instance.
        '''
        torch.save({'model_state_dict': model.state_dict(),
                    'timestep': self.timestep}, self.filename)

    def load_checkpoint(self, model):
        '''Load the latest checkpoint

        Parameters
        ----------
        model : nn.Module
            Model instance.
        '''
        model.load_state_dict(torch.load(self.filename)['model_state_dict'])

        
def ptable_to_csv(table, filename, headers=True):
    """Save PrettyTable results to a CSV file.

    Adapted from @AdamSmith https://stackoverflow.com/questions/32128226

    :param PrettyTable table: Table object to get data from.
    :param str filename: Filepath for the output CSV.
    :param bool headers: Whether to include the header row in the CSV.
    :return: None
    """
    raw = table.get_string()
    data = [
        tuple(filter(None, map(str.strip, splitline)))
        for line in raw.splitlines()
        for splitline in [line.split('|')]
        if len(splitline) > 1
    ]
    if table.title is not None:
        data = data[1:]
    if not headers:
        data = data[1:]
    with open(filename, 'a') as f:
        for d in data:
            f.write('{}\n'.format(','.join(d)))

