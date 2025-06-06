import torch
from torch_utils.graph_process import *
import torch.nn as nn
from torch.nn import BatchNorm2d, Conv1d, Conv2d, ModuleList, Parameter
import torch.nn.functional as F
import numpy as np
device=torch.device('cuda'if torch.cuda.is_available() else 'cpu')
def nconv(x, A):
    """Multiply x by adjacency matrix along source node axis"""
    return torch.einsum('ncvl,vw->ncwl', (x, A.to(x.device))).contiguous()

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


class GWNet_official(nn.Module):
    def __init__(self, num_nodes, dropout=0.3, supports=None, do_graph_conv=True,
                 addaptadj=True, aptinit=None, in_dim=2, out_dim=12,n_hidden=32,pred_len=12,
                cat_feat_gc=False,kernel_size=2, blocks=4, layers=1,apt_size=10):
        super().__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.do_graph_conv = do_graph_conv
        self.cat_feat_gc = False
        self.addaptadj = addaptadj
        residual_channels = n_hidden
        dilation_channels = n_hidden
        skip_channels = n_hidden * 8
        end_channels = n_hidden * 16


        if self.cat_feat_gc:
            self.start_conv = nn.Conv2d(in_channels=1,  # hard code to avoid errors
                                        out_channels=residual_channels,
                                        kernel_size=(1, 1))
            self.cat_feature_conv = nn.Conv2d(in_channels=in_dim - 1,
                                              out_channels=residual_channels,
                                              kernel_size=(1, 1))
        else:
            self.start_conv = nn.Conv2d(in_channels=in_dim,
                                        out_channels=residual_channels,
                                        kernel_size=(1, 1))
        supports = transition_matrix(supports)
        self.fixed_supports = [nn.Parameter(supports,requires_grad=False)]
        receptive_field = 1

        self.supports_len = len(self.fixed_supports)
        if do_graph_conv and addaptadj:
            if aptinit is None:
                nodevecs = torch.randn(num_nodes, apt_size), torch.randn(apt_size, num_nodes)
            
            else:
                nodevecs = self.svd_init(apt_size, aptinit)
            self.supports_len += 1
            self.nodevec1, self.nodevec2 = [Parameter(n.to(device), requires_grad=True) for n in nodevecs]

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
                D *= 2 
                receptive_field += additional_scope
                additional_scope *= 2 
        self.receptive_field = receptive_field

        self.end_conv = nn.Sequential(nn.Conv2d(skip_channels, end_channels, (1, 1), bias=True),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(end_channels, out_dim * pred_len, (1, 1), bias=True))
        self.pred_len=pred_len

    @staticmethod
    def svd_init(apt_size, aptinit):
        m, p, n = torch.svd(aptinit)
        nodevec1 = torch.mm(m[:, :apt_size], torch.diag(p[:apt_size] ** 0.5))
        nodevec2 = torch.mm(torch.diag(p[:apt_size] ** 0.5), n[:, :apt_size].t())
        return nodevec1, nodevec2

    def forward(self, x,adj,**kargs):
        # Input shape is (bs, features, n_nodes, n_timesteps)
        in_len = x.size(3)
        if in_len < self.receptive_field:
            x = nn.functional.pad(x, (self.receptive_field - in_len, 0, 0, 0))
        if self.cat_feat_gc:
            f1, f2 = x[:, [0]], x[:, 1:]
            x1 = self.start_conv(f1)
            x2 = F.leaky_relu(self.cat_feature_conv(f2))
            x = x1 + x2
        else: 
            x = self.start_conv(x) 
        skip = 0
        adjacency_matrices = self.fixed_supports
        # calculate the current adaptive adj matrix once per iteration
        if self.addaptadj:
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            # adp = torch.tanh(self.adaptive_mx) / self.adaptive_mx.shape[0]  # 归一化类似的操作

            adjacency_matrices = list(self.fixed_supports) + [adp]
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
            if self.do_graph_conv:
                graph_out = self.graph_convs[i](x, adjacency_matrices)
                x = x + graph_out if self.cat_feat_gc else graph_out
            else:
                x = self.residual_convs[i](x)
            x = x + residual[:, :, :, -x.size(3):]
            x = self.bn[i](x)

        x = F.relu(skip)  # ignore last X?
        x=self.end_conv(x)
        B, _, N, _ = x.shape
        x = x.transpose(1,-1)
        return x
