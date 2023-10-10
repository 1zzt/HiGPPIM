import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,GATConv,SAGPooling,TransformerConv, GINConv
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool

from torch_geometric.nn import SAGEConv

# from torch_geometric.nn import HypergraphConv
from layers import *
from gatconv import GATConv

class RMSNorm(nn.Module):
    def __init__(self, dim, eps = 1e-6):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.norm(x, dim = -1, keepdim = True) * self.scale
        return x / norm.clamp(min = self.eps) * self.g


class HGA(nn.Module):
    def __init__(self, in_features, hidden_size, gat_pw_headers = 1, gat_pw_edge_dim = 11, 
                 hyg_headers = 1, hyg_edge_dim = 73, gat_fg_headers = 1, 
                 res_out_channels = 32, padding_max = 100, sa_headers = 1) -> None:
        super().__init__()
        self.in_features = in_features
        self.hidden_size = hidden_size

        self.gat_pw_headers = gat_pw_headers
        self.gat_pw_edge_dim = gat_pw_edge_dim

        self.hyg_headers = hyg_headers
        self.hyg_edge_dim = hyg_edge_dim    # 官能团（超边）初始特征维度

        self.gat_fg_headers = gat_fg_headers

        self.sa_headers = sa_headers

        self.res_out_channels = res_out_channels

        self.padding_max = padding_max

        self.alpha_pw = nn.Parameter(torch.tensor([0.5]))
        self.alpha_fg = nn.Parameter(torch.tensor([0.5]))


        # self.conv1 = GATConv(in_features, hidden_size)
        # self.conv2 = GATConv(hidden_size, hidden_size)
        # self.conv3 = GATConv(hidden_size, hidden_size)
        
        self.gat1 = GATConv(in_features, hidden_size // gat_pw_headers, gat_pw_headers, edge_dim = gat_pw_edge_dim)
        self.gat2 = GATConv(hidden_size, hidden_size // gat_pw_headers, gat_pw_headers, edge_dim = gat_pw_edge_dim)
        self.gat3 = GATConv(hidden_size, hidden_size // gat_pw_headers, gat_pw_headers, edge_dim = gat_pw_edge_dim)
        
        self.hyg_conv = HypergraphConv(in_features, hidden_size)

        self.hyg_conv1 = HypergraphConv(hidden_size, hidden_size // hyg_headers,  heads = hyg_headers, hyg_edge_dim = hidden_size, mode='atom')
        self.hyg_conv2 = HypergraphConv(hidden_size, hidden_size // hyg_headers, heads = hyg_headers, hyg_edge_dim = hidden_size, mode='fg')
        self.hyg_conv3 = HypergraphConv(hidden_size, hidden_size // hyg_headers,  heads = hyg_headers, hyg_edge_dim = hidden_size, mode='atom')
        self.hyg_conv4 = HypergraphConv(hidden_size, hidden_size // hyg_headers, heads = hyg_headers, hyg_edge_dim = hidden_size, mode='fg')
        self.hyg_conv5 = HypergraphConv(hidden_size, hidden_size // hyg_headers,  heads = hyg_headers, hyg_edge_dim = hidden_size, mode='atom')
        self.hyg_conv6 = HypergraphConv(hidden_size, hidden_size // hyg_headers, heads = hyg_headers, hyg_edge_dim = hidden_size, mode='fg')


        self.gat4 = GATConv(hyg_edge_dim, hidden_size // gat_fg_headers, gat_fg_headers)
        self.gat5 = GATConv(hidden_size, hidden_size // gat_fg_headers, gat_fg_headers)
        self.gat6 = GATConv(hidden_size, hidden_size // gat_fg_headers, gat_fg_headers)


        self.resb1 = ResidualBlock(1, res_out_channels, padding_max, hidden_size)
        self.resb2 = ResidualBlock_resize(3, res_out_channels, hidden_size)
        self.resb3 = ResidualBlock_resize3(3, res_out_channels, hidden_size)

        self.resb4 = ResidualBlock_atom_fg1(6, res_out_channels, hidden_size, use_1x1conv = True)
        self.resb5 = ResidualBlock_atom_fg2(3, res_out_channels, hidden_size, use_1x1conv = True)

        self.recalb = RecalibrateBlock(hidden_size)
        self.recalb2 = RecalibrateBlock2(hidden_size)
        self.recalb3 = RecalibrateBlock_x3(hidden_size)
        self.recalb4 = RecalibrateBlock_x3_3(hidden_size)

        self.bn1 = RMSNorm(hidden_size)
        self.bn2 = RMSNorm(hidden_size)
        self.bn3 = RMSNorm(hidden_size)     
        self.bn5 = nn.BatchNorm1d(hidden_size)
        self.bn6 = nn.BatchNorm1d(hidden_size)

        self.attention = ScaledDotProductAttention(hidden_size, hidden_size, hidden_size, sa_headers)

        self.readout = SAGPooling(hidden_size, min_score=None, ratio = 0.8)

        self.pred = nn.Sequential(nn.Linear(hidden_size, hidden_size // 2),
                    nn.BatchNorm1d(hidden_size // 2),
                    nn.ReLU(),
                    
                    nn.Linear(hidden_size // 2, 1))
        
        self.pred2 = nn.Sequential(nn.Linear(hidden_size * sa_headers * 3, hidden_size * sa_headers * 3 // 2),
                    nn.BatchNorm1d(hidden_size * sa_headers * 3 // 2),
                    nn.ReLU(),
                    
                    nn.Linear(hidden_size * sa_headers * 3 // 2, 1))
        

        self.pred_resize = nn.Sequential(nn.Linear(hidden_size * res_out_channels, hidden_size * res_out_channels // 2),
            nn.BatchNorm1d(hidden_size * res_out_channels // 2),                             
            nn.ReLU(),
            nn.Linear(hidden_size * res_out_channels // 2, 1))
        
        # self.bn1 = RMSNorm(hidden_size)
    def forward(self, data):

        # 方案1 层次
        # data.x_pw =  (F.leaky_relu(self.gat1(data.x_pw, data.edge_index_pw, data.edge_attr)))   # 原子数量 x 57 -> 原子数量 x 128
        # data.x_fg =  (F.leaky_relu(self.gat4(data.x_fg, data.edge_index_fg)))

        # data.x_fg_1 =  F.leaky_relu(self.hyg_conv1(data.x_pw, data.edge_index_hyg, hyperedge_attr = data.x_fg))
        # data.x_pw_1 =  F.leaky_relu(self.hyg_conv2(data.x_pw, data.edge_index_hyg, hyperedge_attr = data.x_fg))


        # data.x_pw =  (F.leaky_relu(self.gat2(data.x_pw_1, data.edge_index_pw, data.edge_attr)))   # 原子数量 x 57 -> 原子数量 x 128
        # data.x_fg =  (F.leaky_relu(self.gat5(data.x_fg_1, data.edge_index_fg)))

        # data.x_fg_2 =  F.leaky_relu(self.hyg_conv3(data.x_pw, data.edge_index_hyg, hyperedge_attr = data.x_fg))
        # data.x_pw_2 =  F.leaky_relu(self.hyg_conv4(data.x_pw, data.edge_index_hyg, hyperedge_attr = data.x_fg))

        
        # data.x_pw =  (F.leaky_relu(self.gat3(data.x_pw_2, data.edge_index_pw, data.edge_attr)))   # 原子数量 x 57 -> 原子数量 x 128
        # data.x_fg =  (F.leaky_relu(self.gat6(data.x_fg_2, data.edge_index_fg)))

        # 方案2 枢纽
        data.x_pw =  (F.leaky_relu(self.gat1(data.x_pw, data.edge_index_pw, data.edge_attr)))   # 原子数量 x 57 -> 原子数量 x 128
        data.x_fg =  (F.leaky_relu(self.gat4(data.x_fg, data.edge_index_fg)))

        data.x_fg_1 =  F.leaky_relu(self.hyg_conv1(data.x_pw, data.edge_index_hyg, hyperedge_attr = data.x_fg))
        data.x_pw_1 =  F.leaky_relu(self.hyg_conv2(data.x_pw, data.edge_index_hyg, hyperedge_attr = data.x_fg))


        data.x_pw =  (F.leaky_relu(self.gat2((data.x_pw + data.x_pw_1), data.edge_index_pw, data.edge_attr)))   # 原子数量 x 57 -> 原子数量 x 128
        data.x_fg =  (F.leaky_relu(self.gat5((data.x_fg + data.x_fg_1), data.edge_index_fg)))

        data.x_fg_2 =  F.leaky_relu(self.hyg_conv3(data.x_pw, data.edge_index_hyg, hyperedge_attr = data.x_fg))
        data.x_pw_2 =  F.leaky_relu(self.hyg_conv4(data.x_pw, data.edge_index_hyg, hyperedge_attr = data.x_fg))

        
        data.x_pw =  (F.leaky_relu(self.gat3((data.x_pw + data.x_pw_2), data.edge_index_pw, data.edge_attr)))   # 原子数量 x 57 -> 原子数量 x 128
        data.x_fg =  (F.leaky_relu(self.gat6((data.x_fg + data.x_fg_2), data.edge_index_fg)))


        # 方案3 枢纽按比例


        # data.x_pw =  (F.leaky_relu(self.gat1(data.x_pw, data.edge_index_pw, data.edge_attr)))   # 原子数量 x 57 -> 原子数量 x 128
        # data.x_fg =  (F.leaky_relu(self.gat4(data.x_fg, data.edge_index_fg)))

        # data.x_fg_1 =  F.leaky_relu(self.hyg_conv1(data.x_pw, data.edge_index_hyg, hyperedge_attr = data.x_fg))
        # data.x_pw_1 =  F.leaky_relu(self.hyg_conv2(data.x_pw, data.edge_index_hyg, hyperedge_attr = data.x_fg))


        # data.x_pw =  (F.leaky_relu(self.gat2((data.x_pw * self.alpha_pw + data.x_pw_1 * (1 - self.alpha_pw)), data.edge_index_pw, data.edge_attr)))   # 原子数量 x 57 -> 原子数量 x 128
        # data.x_fg =  (F.leaky_relu(self.gat5((data.x_fg * self.alpha_fg + data.x_fg_1 * (1 - self.alpha_fg)), data.edge_index_fg)))

        # data.x_fg_2 =  F.leaky_relu(self.hyg_conv3(data.x_pw, data.edge_index_hyg, hyperedge_attr = data.x_fg))
        # data.x_pw_2 =  F.leaky_relu(self.hyg_conv4(data.x_pw, data.edge_index_hyg, hyperedge_attr = data.x_fg))

        
        # data.x_pw =  (F.leaky_relu(self.gat3((data.x_pw * self.alpha_pw + data.x_pw_2 * (1 - self.alpha_pw)), data.edge_index_pw, data.edge_attr)))   # 原子数量 x 57 -> 原子数量 x 128
        # data.x_fg =  (F.leaky_relu(self.gat6((data.x_fg * self.alpha_fg + data.x_fg_2 * (1 - self.alpha_fg)), data.edge_index_fg)))

    
        # print(self.alpha_pw)
        # print(self.alpha_fg)

        # -----------------------------3. residual net --resize ：256 = 16x16 ---------------------------

        global_pw_add = global_add_pool(data.x_pw, data.x_pw_batch).view(-1, 16, 16)
        global_pw_mean = global_mean_pool(data.x_pw, data.x_pw_batch).view(-1, 16, 16)
        global_pw_max = global_max_pool(data.x_pw, data.x_pw_batch).view(-1, 16, 16)
        

        global_pw = torch.stack([global_pw_add, global_pw_mean, global_pw_max], dim=1)

        # global_hyg_add = global_add_pool(data.x_hyg, data.x_hyg_batch).view(-1, 16, 16)
        # global_hyg_mean = global_mean_pool(data.x_hyg, data.x_hyg_batch).view(-1, 16, 16)
        # global_hyg_max = global_max_pool(data.x_hyg, data.x_hyg_batch).view(-1, 16, 16)

        # global_hyg = torch.stack([global_hyg_add, global_hyg_mean, global_hyg_max], dim=1)

        global_fg_add = global_add_pool(data.x_fg, data.x_fg_batch).view(-1, 16, 16)
        global_fg_mean = global_mean_pool(data.x_fg, data.x_fg_batch).view(-1, 16, 16)
        global_fg_max = global_max_pool(data.x_fg, data.x_fg_batch).view(-1, 16, 16)

        global_fg = torch.stack([global_fg_add, global_fg_mean, global_fg_max], dim=1)

        # global_fg = global_add_pool(data.x_fg, data.x_fg_batch)

        # res_x_pw, res_x_hyg = self.resb(global_pw, global_hyg)

        # 1. 不校准x3 resnet直接融合x1,2,3
        # res_x_pw, res_x_hyg, res_x_fg = self.resb3(global_pw, global_hyg, global_fg)
        # global_fea = res_x_pw * res_x_hyg * res_x_fg

        # pred = self.pred(global_fea)

        # 2. 仅校准x3， resnet后的x3
        # res_x_pw, res_x_hyg, res_x_fg = self.resb3(global_pw, global_hyg, global_fg)
        
        # global_fea = self.recalb3(res_x_pw * res_x_hyg, res_x_fg)

        # pred = self.pred(global_fea)

        # 3. 仅校准x3，没有输入到resnet的x3
        
        # global_fg_add = global_add_pool(data.x_fg, data.x_fg_batch)
        # global_fg_mean = global_mean_pool(data.x_fg, data.x_fg_batch)
        # global_fg_max = global_max_pool(data.x_fg, data.x_fg_batch)

        # res_x_pw, res_x_hyg = self.resb2(global_pw, global_hyg)
        
        # global_fea = self.recalb4(res_x_pw * res_x_hyg, global_fg_add, global_fg_mean, global_fg_max)

        # pred = self.pred(global_fea)

        # 4. x1, x3融合--先拼接后融合
        # global_fea = self.resb4(global_pw, global_fg)
        # pred = self.pred(global_fea)

        # 4. x1, x3融合--分别卷积再*
        global_fea = self.resb5(global_pw, global_fg)
        pred = self.pred(global_fea)



        # global_fea = self.recalb(res_x_pw, global_pw, res_x_hyg, global_hyg, global_fg)
        # global_fea = self.recalb2(res_x_pw, global_pw, res_x_hyg, global_hyg) * global_fg
        # global_fea = res_x_pw * res_x_hyg * global_fg

        # -----------------------------3. residual net --resize ：256 = 16x16 ----------------------------


        # global_fea = torch.concat((global_pw, global_hyg), dim=1)
        # global_fea = global_pw * global_hyg * global_fg
        # pred = self.pred_resize(global_fea)
        # print('hhhh')
        return pred
        