import os
import argparse
import pandas as pd
from dataset import MyDataset
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
# from model import *
from model_combin import *
# from model_gnn_attention import Trans
from utils import *
import random
import numpy as np
from adan import Adan
from sklearn.model_selection import train_test_split
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from prettytable import PrettyTable

# from torch.utils.tensorboard import SummaryWriter

# tb_writer = SummaryWriter(log_dir="runs/HIGPPIM")

# os.environ["CUDA_VISIBLE_DEVICES"] = '1'




parser = argparse.ArgumentParser(description='Optional app description')
parser.add_argument('--dataset_name', type=str, required=False, default='bcl2_bak')
parser.add_argument('--task', type=str, required=False, default='classification')
parser.add_argument('--cuda', type=str, required=False, default='3')
parser.add_argument('--batch_size', type=int, required=False, default='32')
parser.add_argument('--lr', type=float, required=False, default='0.0001')
args = parser.parse_args()
dataset_name = args.dataset_name
cuda = args.cuda
task_name = args.task
batch_size = args.batch_size
lr = args.lr

# bcl2_bak bromodomain_histone cd4_gp120 ledgf_in lfa_icam mdm2_p53 ras_sos1 xiap_smac 


data_path = '/home/zitong/project/PPIM/Datasets/HIGPPIM/'  # pdCSM-PPI, SMMPPI-usesametrainset PPIMp
save_path = '/data/zhangzt/PPIM/hypergraph/infohub/' 

device = torch.device(("cuda:" + cuda) if torch.cuda.is_available() else "cpu") 


def run_a_train_epoch(model, data_loader):
    model.train()
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.96 ** (epoch))
    for _, data in enumerate(data_loader):

        data = data.to(device)
        y = data.y.to(device).float()
        pred = model(data).float()
        
        train_loss = loss_criterion(pred.view(-1), y.view(-1))

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        
    return train_loss.item()

def run_an_eval_epoch(model, data_loader):
    model.eval()
    eval_meter = EvalMeter()
    with torch.no_grad():
        for _, data in enumerate(data_loader):

            data = data.to(device)
            y = data.y.to(device).float()
            pred = model(data).float()

            val_loss = loss_criterion(pred.view(-1), y.view(-1))

            eval_meter.update(pred.view(-1), y.view(-1))

        eval_dict = {}
        for eval_item in metrics:
            eval_dict[eval_item] = round(eval_meter.compute_metric(eval_item), 3)
        
        
        
    return val_loss.item(), eval_dict


print(dataset_name)


configs = {
    'data_path': data_path,
    'task_name': task_name,
    'dataset_name': dataset_name,
    'batch_size': batch_size
}

train_loader, val_loader, test_loader = get_data(configs)
# optimizer = Adan(
#                 model.parameters(),
#                 # optimizer_grouped_parameters,
#                 lr =1e-3,                  # learning rate (can be much higher than Adam, up to 5-10x)
#                 betas = (0.02, 0.08, 0.001), # beta 1-2-3 as described in paper - author says most sensitive to beta3 tuning
#                 weight_decay = 0.02        # weight decay 0.02 is optimal per author
#             )

if task_name == 'classification':
    loss_criterion = nn.BCEWithLogitsLoss()
    metrics = ['mcc', 'f1', 'roc_auc_score','acc']
    t_tables = PrettyTable(['method', 'MCC', 'F1', 'AUC','ACC'])
else:
    loss_criterion = nn.MSELoss()
    metrics = ['pcc','kendall','spearman','rmse','mae']
    t_tables = PrettyTable(['method', 'pcc','kendall','spearman','rmse','mae'])

t_tables.float_format = '.3'   



model = HGA(in_features = 57, hidden_size = 256, gat_pw_headers = 4, gat_pw_edge_dim = 11, 
            hyg_headers = 4, hyg_edge_dim = 73, gat_fg_headers = 4).to(device)

# model(batch)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

folder_path = os.path.join(save_path, 'pth', dataset_name)
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    print(f"Folder {folder_path} has been created.")

stoper_mode = 'higher'
stopperfile = os.path.join(folder_path, 'seed' +'-'+ stoper_mode +'stopper.pth')

# patience=30 100
stopper = EarlyStopping(patience=100, filename=stopperfile, mode = stoper_mode)



for epoch in range(4000):

    model.train()
    train_loss = run_a_train_epoch(model, train_loader)
    # print('train_loss: ' + str(train_loss))

    val_loss, eval_dict_val = run_an_eval_epoch(model, val_loader)
    # print('val_loss: ' + str(val_loss))
    if task_name == 'classification':
        val_metric = eval_dict_val['mcc']
    else:
        if stoper_mode == 'higher':
            val_metric = round((eval_dict_val['pcc'] + eval_dict_val['kendall'] + eval_dict_val['spearman'])/3, 3) - eval_dict_val['rmse'] - eval_dict_val['mae']
        else:
            val_metric = eval_dict_val['rmse'] + eval_dict_val['mae']
    # print('valid result: ')
    # print(eval_dict_val)
    
    if epoch > 10:
        early_stop = stopper.step([val_loss, val_metric], model)
        if early_stop:
            print(epoch)
            break

stopper.load_checkpoint(model)
test_loss, eval_dict_test = run_an_eval_epoch(model, test_loader)
# print('test_loss: ' + str(test_loss))
# print('test result: ')
# print(eval_dict_test)

# result = pd.DataFrame([str(epoch), ('train_loss: ' + str(train_loss)), ('val_loss: ' + str(val_loss))])
# result.to_csv(results_filename,mode = 'a', index=None, header=None)

if task_name == 'classification':
    mcc, f1, auc, acc  = eval_dict_test['mcc'], eval_dict_test['f1'], eval_dict_test['roc_auc_score'], eval_dict_test['acc']
    row = ['test', mcc, f1, auc, acc]
else:
    pcc, kendall,spearman, rmse, mae = eval_dict_test['pcc'], eval_dict_test['kendall'], eval_dict_test['spearman'], eval_dict_test['rmse'], eval_dict_test['mae']
    row = ['test', pcc, kendall,spearman, rmse, mae]
t_tables.add_row(row)
print(t_tables)
