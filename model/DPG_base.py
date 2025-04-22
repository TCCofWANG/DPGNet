import torch
from torch_utils.graph_process import *
import torch.nn as nn
from torch.nn import BatchNorm2d, Conv1d, Conv2d, ModuleList, Parameter
import torch.nn.functional as F
import numpy as np
import math
import matplotlib.pyplot as plt
import os
import seaborn as sns



class DPG_Predictor(nn.Module):
    def __init__(self, num_nodes, in_dim, out_dim,d_model,d_ff,pred_len,
                 dropout=0.3, kernel_size=2, blocks=4, layers=1,args=None,**kwargs):
        super().__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.args=args

        residual_channels = d_model
        dilation_channels = d_model
        skip_channels = d_ff
        end_channels = d_ff

        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                        out_channels=residual_channels,
                                        kernel_size=(1, 1))

        receptive_field = 1

        self.supports_len = 2

        depth = list(range(blocks * layers))

        # 1x1 convolution for residual and skip connections (slightly different see docstring)
        self.residual_convs = ModuleList([Conv2d(dilation_channels, residual_channels, (1, 1)) for _ in depth]) # 层内
        self.skip_convs = ModuleList([Conv2d(dilation_channels, skip_channels, (1, 1)) for _ in depth]) # 层间
        self.bn = ModuleList([BatchNorm2d(residual_channels) for _ in depth])
        self.graph_convs = ModuleList([GraphConvNet(dilation_channels, residual_channels, dropout, support_len=self.supports_len)
                                              for _ in depth])

        self.filter_convs = ModuleList()
        self.gate_convs = ModuleList()
        for b in range(blocks):
            additional_scope = kernel_size - 1
            D = 1 # dilation
            for i in range(layers):
                # dilated convolutions，TCN-a,TCN-b
                self.filter_convs.append(Conv2d(residual_channels, dilation_channels, (1, kernel_size), dilation=D))
                self.gate_convs.append(Conv2d(residual_channels, dilation_channels, (1, kernel_size), dilation=D))
                D *= 2 # 膨胀系数d以2的指数形式增长
                receptive_field += additional_scope
                additional_scope *= 2 #由于kernel=2，因此其实每一次加上膨胀系数d即可
        self.receptive_field = receptive_field

        self.end_conv = nn.Sequential(nn.Conv2d(skip_channels, end_channels, (1, 1), bias=True),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(end_channels, out_dim * pred_len, (1, 1), bias=True))
        self.pred_len=pred_len

    def forward(self, x,adj,**kargs):
        # adj = [adj_dynamic_with_explicit,adj_dynamic_with_implicit]
        mode = kargs.get('mode')
        x = self.start_conv(x)  # 特征维度升维
        adjacency_matrices=adj

        in_len = x.size(3)
        if in_len < self.receptive_field:
            x = nn.functional.pad(x, (self.receptive_field - in_len, 0, 0, 0))
        skip = 0

        # WaveNet layers
        for i in range(self.blocks * self.layers):
            # EACH BLOCK
            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |   |-dil_conv -- tanh --|                |
            #         ---|                  * ----|-- 1x1 -- + -->	*x_in*
            #                |-dil_conv -- sigm --|    |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*
            residual = x
            # dilated convolution -->输入序列的长度会不断变小
            filter = torch.tanh(self.filter_convs[i](residual))
            gate = torch.sigmoid(self.gate_convs[i](residual))
            x = filter * gate
            # parametrized skip connection
            s = self.skip_convs[i](x)  # what are we skipping??
            try:  # if i > 0 this works
                skip = skip[:, :, :,  -s.size(3):]
            except:
                skip = 0
            skip = s + skip

            if i == (self.blocks * self.layers - 1):  # last X getting ignored anyway
                break

            graph_out = self.graph_convs[i](x, adjacency_matrices)
            x = graph_out

            x = x + residual[:, :, :, -x.size(3):]
            x = self.bn[i](x)

        x = F.relu(skip)  # ignore last X?
        x=self.end_conv(x)
        B, _, N, _ = x.shape
        x = x.transpose(1,-1)
        return x


def nconv(x, A):
    """Multiply x by adjacency matrix along source node axis"""
    return torch.einsum('ncvl,nvw->ncwl', (x, A.to(x.device))).contiguous()


class GraphConvNet(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super().__init__()
        c_in = (order * support_len + 1) * c_in
        self.final_conv = Conv2d(c_in, c_out, (1, 1), padding=(0, 0), stride=(1, 1), bias=True)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support: list):
        out = [x]
        for a in support:
            x1 = nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = nconv(x1, a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out, dim=1)
        h = self.final_conv(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h