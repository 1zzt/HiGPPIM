from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor
import torch.nn as nn
from torch.nn import Parameter
from torch_scatter import scatter_add

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import softmax

import numpy as np

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, padding_max, hidden_size):
        super(ResidualBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.padding_max = padding_max
 
        self.conv11 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv12 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv13 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn11 = nn.BatchNorm2d(out_channels)
        self.bn12 = nn.BatchNorm2d(out_channels)
        self.bn13 = nn.BatchNorm2d(out_channels)
        self.lin1 = nn.Linear(out_channels * padding_max * hidden_size, hidden_size)


        self.conv21 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv22 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv23 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn21 = nn.BatchNorm2d(out_channels)
        self.bn22 = nn.BatchNorm2d(out_channels)
        self.bn23 = nn.BatchNorm2d(out_channels)
        self.lin2 = nn.Linear(out_channels * padding_max * hidden_size, hidden_size)


 
    def forward(self, x1, x2):
        # 没有share weight
        x1_out = F.relu(self.bn11(self.conv11(x1)))
        x1_out = F.relu(self.bn12(self.conv12(x1_out)))
        x1_out = self.bn13(self.conv13(x1_out))
        x1_out = F.relu(x1 + x1_out)

        x2_out = F.relu(self.bn21(self.conv21(x2)))
        x2_out = F.relu(self.bn22(self.conv22(x2_out)))
        x2_out = self.bn23(self.conv23(x2_out))
        x2_out = F.relu(x2 + x2_out)


        x1_out = torch.flatten(x1_out,1)
        x1_out = self.lin1(x1_out)

        x2_out = torch.flatten(x2_out,1)
        x2_out = self.lin2(x2_out)

        return x1_out, x2_out


class ResidualBlock_resize(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_size):
        super(ResidualBlock_resize, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size
 
        self.conv11 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv12 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv13 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn11 = nn.BatchNorm2d(out_channels)
        self.bn12 = nn.BatchNorm2d(out_channels)
        self.bn13 = nn.BatchNorm2d(out_channels)
        self.lin1 = nn.Linear(out_channels * hidden_size, hidden_size)


        self.conv21 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv22 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv23 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn21 = nn.BatchNorm2d(out_channels)
        self.bn22 = nn.BatchNorm2d(out_channels)
        self.bn23 = nn.BatchNorm2d(out_channels)
        self.lin2 = nn.Linear(out_channels * hidden_size, hidden_size)


 
    def forward(self, x1, x2):
        # 没有share weight
        x1_out = F.relu(self.bn11(self.conv11(x1)))
        x1_out = F.relu(self.bn12(self.conv12(x1_out)))
        x1_out = self.bn13(self.conv13(x1_out))
        x1_out = F.relu(x1 + x1_out)

        x2_out = F.relu(self.bn21(self.conv21(x2)))
        x2_out = F.relu(self.bn22(self.conv22(x2_out)))
        x2_out = self.bn23(self.conv23(x2_out))
        x2_out = F.relu(x2 + x2_out)

        x1_out = torch.flatten(x1_out,1)
        x1_out = self.lin1(x1_out)

        x2_out = torch.flatten(x2_out,1)
        x2_out = self.lin2(x2_out)

        return x1_out, x2_out

class ResidualBlock_atom_fg1(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_size, use_1x1conv = True):
        super(ResidualBlock_atom_fg1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size
 
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.lin1 = nn.Linear(out_channels * hidden_size, hidden_size)

        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
 
    def forward(self, x1, x3):

        x = torch.cat((x1, x3), dim=1)
        
        x_out = F.relu(self.bn1(self.conv1(x)))
        x_out = F.relu(self.bn2(self.conv2(x_out)))
        x_out = self.bn3(self.conv3(x_out))

        x = self.conv1x1(x)
        x_out = F.relu(x + x_out)

        # 展平成向量 [b, hidden_size]
        
        x_out = torch.flatten(x_out,1)
        x_out = self.lin1(x_out)

        return x_out


class  ResidualBlock_atom_fg2(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_size, use_1x1conv = True):
        super(ResidualBlock_atom_fg2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.use_1x1conv = use_1x1conv
 
        self.conv11 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv12 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv13 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn11 = nn.BatchNorm2d(out_channels)
        self.bn12 = nn.BatchNorm2d(out_channels)
        self.bn13 = nn.BatchNorm2d(out_channels)
        self.lin1 = nn.Linear(out_channels * hidden_size, hidden_size)

        self.conv1x1_1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.conv21 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv22 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv23 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn21 = nn.BatchNorm2d(out_channels)
        self.bn22 = nn.BatchNorm2d(out_channels)
        self.bn23 = nn.BatchNorm2d(out_channels)
        self.lin2 = nn.Linear(out_channels * hidden_size, hidden_size)

        self.conv1x1_2 = nn.Conv2d(in_channels, out_channels, kernel_size=1)


 
    def forward(self, x1, x3):

        x1_out = F.relu(self.bn11(self.conv11(x1)))
        x1_out = F.relu(self.bn12(self.conv12(x1_out)))
        x1_out = self.bn13(self.conv13(x1_out))
        
        if self.use_1x1conv:
            x1 = self.conv1x1_1(x1)
        x1_out = F.relu(x1 + x1_out)


        x3_out = F.relu(self.bn21(self.conv21(x3)))
        x3_out = F.relu(self.bn22(self.conv22(x3_out)))
        x3_out = self.bn23(self.conv23(x3_out))

        if self.use_1x1conv:
            x3 = self.conv1x1_2(x3)
        x3_out = F.relu(x3 + x3_out)

        # 展平成向量 [b, hidden_size]
        x1_out = torch.flatten(x1_out,1)
        x1_out = self.lin1(x1_out)

        x3_out = torch.flatten(x3_out,1)
        x3_out = self.lin2(x3_out)

        return x1_out * x3_out


class ResidualBlock_resize3(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_size):
        super(ResidualBlock_resize3, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size
 
        self.conv11 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv12 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv13 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn11 = nn.BatchNorm2d(out_channels)
        self.bn12 = nn.BatchNorm2d(out_channels)
        self.bn13 = nn.BatchNorm2d(out_channels)
        self.lin1 = nn.Linear(out_channels * hidden_size, hidden_size)


        self.conv21 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv22 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv23 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn21 = nn.BatchNorm2d(out_channels)
        self.bn22 = nn.BatchNorm2d(out_channels)
        self.bn23 = nn.BatchNorm2d(out_channels)
        self.lin2 = nn.Linear(out_channels * hidden_size, hidden_size)

        self.conv31 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv32 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv33 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn31 = nn.BatchNorm2d(out_channels)
        self.bn32 = nn.BatchNorm2d(out_channels)
        self.bn33 = nn.BatchNorm2d(out_channels)
        self.lin3 = nn.Linear(out_channels * hidden_size, hidden_size)
 
    def forward(self, x1, x2, x3):

        x1_out = F.relu(self.bn11(self.conv11(x1)))
        x1_out = F.relu(self.bn12(self.conv12(x1_out)))
        x1_out = self.bn13(self.conv13(x1_out))
        x1_out = F.relu(x1 + x1_out)

        x2_out = F.relu(self.bn21(self.conv21(x2)))
        x2_out = F.relu(self.bn22(self.conv22(x2_out)))
        x2_out = self.bn23(self.conv23(x2_out))
        x2_out = F.relu(x2 + x2_out)

        x3_out = F.relu(self.bn31(self.conv31(x3)))
        x3_out = F.relu(self.bn32(self.conv32(x3_out)))
        x3_out = self.bn33(self.conv33(x3_out))
        x3_out = F.relu(x3 + x3_out)


        x1_out = torch.flatten(x1_out,1)
        x1_out = self.lin1(x1_out)

        x2_out = torch.flatten(x2_out,1)
        x2_out = self.lin2(x2_out)

        x3_out = torch.flatten(x3_out,1)
        x3_out = self.lin3(x3_out)

        return x1_out, x2_out, x3_out


class RecalibrateBlock(nn.Module):
    def __init__(self, hidden_size):
        super(RecalibrateBlock, self).__init__()
        self.hidden_size = hidden_size

        self.mlp1 = nn.Sequential(
            Linear(hidden_size, hidden_size, bias=True),
            nn.ReLU(inplace=True),
            Linear(hidden_size, hidden_size, bias=True),
        )

        self.mlp2 = nn.Sequential(
            Linear(hidden_size, hidden_size, bias=True),
            nn.ReLU(inplace=True),
            Linear(hidden_size, hidden_size, bias=True),
        )

        self.mlp3 = nn.Sequential(
            Linear(hidden_size, hidden_size, bias=True),
            nn.ReLU(inplace=True),
            Linear(hidden_size, hidden_size, bias=True),
        )

    def forward(self, x1, x1_g, x2, x2_g, xfg):

        x1_g = self.mlp1(x1_g)
        x2_g = self.mlp2(x2_g)
        xfg = self.mlp3(xfg)
        

        x1_out = torch.sigmoid(x1_g) * x1
        x2_out = torch.sigmoid(x2_g) * x2

        x = torch.sigmoid(xfg) * (x1_out + x2_out)

        return x


class RecalibrateBlock2(nn.Module):
    def __init__(self, hidden_size):
        super(RecalibrateBlock2, self).__init__()
        self.hidden_size = hidden_size

        self.mlp1 = nn.Sequential(
            Linear(hidden_size, hidden_size, bias=True),
            nn.ReLU(inplace=True),
            Linear(hidden_size, hidden_size, bias=True),
        )

        self.mlp2 = nn.Sequential(
            Linear(hidden_size, hidden_size, bias=True),
            nn.ReLU(inplace=True),
            Linear(hidden_size, hidden_size, bias=True),
        )

        self.mlp3 = nn.Sequential(
            Linear(hidden_size, hidden_size, bias=True),
            nn.ReLU(inplace=True),
            Linear(hidden_size, hidden_size, bias=True),
        )

    def forward(self, x1, x1_g, x2, x2_g):

        x1_g = self.mlp1(x1_g)
        x2_g = self.mlp2(x2_g)
        

        x1_out = torch.sigmoid(x1_g) * x1
        x2_out = torch.sigmoid(x2_g) * x2

        x = (x1_out + x2_out) 

        return x       


class RecalibrateBlock_x3(nn.Module):
    def __init__(self, hidden_size):
        super(RecalibrateBlock_x3, self).__init__()
        self.hidden_size = hidden_size

        self.mlp1 = nn.Sequential(
            Linear(hidden_size, hidden_size, bias=True),
            nn.ReLU(inplace=True),
            Linear(hidden_size, hidden_size, bias=True),
        )

    def forward(self, x12, x3):
        x3 = self.mlp1(x3)

        out = torch.sigmoid(x3) * x12

        return out 
    

class RecalibrateBlock_x3_3(nn.Module):
    def __init__(self, hidden_size):
        super(RecalibrateBlock_x3_3, self).__init__()
        self.hidden_size = hidden_size

        self.mlp1 = nn.Sequential(
            Linear(hidden_size, hidden_size, bias=True),
            nn.ReLU(inplace=True),
            Linear(hidden_size, hidden_size, bias=True),
        )
        self.mlp2 = nn.Sequential(
            Linear(hidden_size, hidden_size, bias=True),
            nn.ReLU(inplace=True),
            Linear(hidden_size, hidden_size, bias=True),
        )
        self.mlp3 = nn.Sequential(
            Linear(hidden_size, hidden_size, bias=True),
            nn.ReLU(inplace=True),
            Linear(hidden_size, hidden_size, bias=True),
        )

    def forward(self, x12, x3_add, x3_mean, x3_max):
        x3_add = self.mlp1(x3_add)
        x3_mean = self.mlp2(x3_mean)
        x3_max = self.mlp3(x3_max)

        x3 = x3_add + x3_mean + x3_max

        out = torch.sigmoid(x3) * x12

        return out 


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, headers, dropout=.0):

        super(ScaledDotProductAttention, self).__init__()
        self.fc_q = nn.Linear(d_model, headers * d_k)
        self.fc_k = nn.Linear(d_model, headers * d_k)
        self.fc_v = nn.Linear(d_model, headers * d_v)
        self.fc_o = nn.Linear(headers * d_v, d_model)
        self.dropout=nn.Dropout(dropout)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.headers = headers

        # self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


    def forward(self, query, key, value, attention_mask=None, attention_scale=None):

        b_s, nq = query.shape[:2]   
        nk = key.shape[1]

        q = self.fc_q(query).view(b_s, nq, self.headers, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(key).view(b_s, nk, self.headers, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(value).view(b_s, nk, self.headers, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)
        # print(q.shape)
        # print(k.shape)
        # print(v.shape)
        # print(torch.matmul(q, k).shape)

        if attention_scale is not None:
            att = att * attention_scale
        att = torch.softmax(att, -1)
        # print(att)

        att=self.dropout(att)

        # print(att.shape)
        # print(v.shape)

        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.headers * self.d_v)  # (b_s, nq, h*d_v)

        return out

        
class HypergraphConv(MessagePassing):
   
    def __init__(self, in_channels, out_channels, use_attention=True, heads = 1, hyg_edge_dim = 73,
                 concat=True, negative_slope=0.2, dropout=0, bias=True, mode = 'atom',
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(flow='source_to_target', node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_attention = use_attention
        self.hyg_edge_dim = hyg_edge_dim
        self.mode = mode

        if self.use_attention:
            self.heads = heads
            self.concat = concat
            self.negative_slope = negative_slope
            self.dropout = dropout
            self.lin = Linear(in_channels, heads * out_channels, bias=False,
                              weight_initializer='glorot')
            
            self.lin_hyperedge = Linear(hyg_edge_dim, heads * out_channels, bias=False,
                    weight_initializer='glorot')
            
            self.att = Parameter(torch.Tensor(1, heads, 2 * out_channels))
        else:
            self.heads = 1
            self.concat = True
            self.lin = Linear(in_channels, out_channels, bias=False,
                              weight_initializer='glorot')
        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        if self.use_attention:
            glorot(self.att)
        zeros(self.bias)

    def forward(self, x: Tensor, hyperedge_index: Tensor,
                hyperedge_weight: Optional[Tensor] = None,
                hyperedge_attr: Optional[Tensor] = None) -> Tensor:
        num_nodes, num_edges = x.size(0), 0

        if hyperedge_index.numel() > 0:
            num_edges = int(hyperedge_index[1].max()) + 1

        if hyperedge_weight is None:
            hyperedge_weight = x.new_ones(num_edges)

        x = self.lin(x)

        alpha = None
        if self.use_attention:
            
            assert hyperedge_attr is not None
            x = x.view(-1, self.heads, self.out_channels)
            hyperedge_attr = self.lin_hyperedge(hyperedge_attr)
            hyperedge_attr = hyperedge_attr.view(-1, self.heads,
                                                 self.out_channels)
            x_i = x[hyperedge_index[0]]
            x_j = hyperedge_attr[hyperedge_index[1]]
            alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
            alpha = F.leaky_relu(alpha, self.negative_slope)
            if self.mode == 'atom':

                alpha = softmax(alpha, hyperedge_index[1], num_nodes=x.size(0))
            else:
                alpha = softmax(alpha, hyperedge_index[0], num_nodes=x.size(0))
            # alpha = softmax(alpha, hyperedge_index[0], num_nodes=x.size(0))
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        D = scatter_add(hyperedge_weight[hyperedge_index[1]],
                        hyperedge_index[0], dim=0, dim_size=num_nodes)
        D = 1.0 / D
        D[D == float("inf")] = 0

        B = scatter_add(x.new_ones(hyperedge_index.size(1)),
                        hyperedge_index[1], dim=0, dim_size=num_edges)
        B = 1.0 / B
        B[B == float("inf")] = 0

        if self.mode == 'atom':
            out = self.propagate(hyperedge_index, x=x, norm=B, alpha=alpha,
                             size=(num_nodes, num_edges))
        else:
            out = self.propagate(hyperedge_index.flip([0]), x=hyperedge_attr, norm=D,
                             alpha=alpha, size=(num_edges, num_nodes))
            

        if self.concat is True:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)
        if self.bias is not None:
            out = out + self.bias

        return out

    def message(self, x_j: Tensor, norm_i: Tensor, alpha: Tensor) -> Tensor:
        
        H, F = self.heads, self.out_channels

        out = norm_i.view(-1, 1, 1) * x_j.view(-1, H, F)

        if alpha is not None:
            out = alpha.view(-1, self.heads, 1) * out

        return out
