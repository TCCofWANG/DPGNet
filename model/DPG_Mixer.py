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
from layers.DPG_Mixer_base import DPG_Mixer_Predictor

device=torch.device('cuda'if torch.cuda.is_available() else 'cpu')


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)



class DPG_Mixer(nn.Module):
    def __init__(self, num_nodes, in_dim, out_dim,d_model,d_ff,pred_len,
                 dropout=0.3, supports=None,kernel_size=2, blocks=4, layers=1,args=None,**kwargs):
        super().__init__()
        # 超参设置
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.args=args
        residual_channels = d_model
        self.pred_n = 1
        self.iter_pred_len = pred_len
        self.supports_len = 3
        supports = transition_matrix(supports)
        self.fixed_supports = nn.Parameter(supports,requires_grad=False)
        self.flag = True

        self.init_fc = Conv2d(in_dim, residual_channels, (1, 1), padding=(0, 0), stride=(1, 1), bias=True)

        # TODO 动态图
        self.xlstm = XLSTM_dynamic_graph(in_feature=residual_channels,d_model=d_model,save_path=args.output_dir,num_nodes=num_nodes,pred_n=self.pred_n)

        # TODO 输出头
        self.predictor = DPG_Mixer_Predictor(num_nodes,d_model,out_dim,d_model,d_ff,self.iter_pred_len,self.supports_len,dropout,kernel_size,blocks,layers,configs=args)

        # TODO embedding
        self.embedding_use = args.embedding_use
        self.num_layer = 2
        self.num_nodes = args.num_nodes
        self.day_of_week_size = 7  # 选取一周的n天
        self.time_of_day_size = args.points_per_hour * 24  # 一天有几个时间步记录
        self.time_in_day_emb = nn.Parameter(torch.empty(self.time_of_day_size, self.args.d_model))
        nn.init.xavier_uniform_(self.time_in_day_emb)
        self.day_in_week_emb = nn.Parameter(torch.empty(self.day_of_week_size, self.args.d_model))
        nn.init.xavier_uniform_(self.day_in_week_emb)
        self.node_emb = nn.Parameter(torch.empty(args.num_nodes, self.args.d_model))
        nn.init.xavier_uniform_(self.node_emb)
        self.time_series_emb_layer = nn.Conv2d(
            in_channels=self.args.num_features * self.args.seq_len, out_channels=self.args.d_model, kernel_size=(1, 1),
            bias=True)
        self.hidden_dim = self.args.d_model + self.args.d_model + self.args.d_model + self.args.d_model
        self.encoder = nn.Sequential(
            *[MultiLayerPerceptron(self.hidden_dim, self.hidden_dim) for _ in range(self.num_layer)])
        self.regression_layer = nn.Conv2d(in_channels=self.hidden_dim,
                                          out_channels=self.args.seq_len * self.args.num_features, kernel_size=(1, 1),
                                          bias=True)
        self.bn_hidden = nn.BatchNorm2d(in_dim, affine=True)

        # TODO norm
        self.norm_use = args.norm_use
        self.t_norm = TNorm(num_nodes, self.args.d_model)
        self.s_norm = SNorm(self.args.d_model)
        self.conv_norm = nn.Conv2d(3 * self.args.d_model, self.args.d_model, kernel_size=1)
        self.conv_norm2 = nn.Conv2d(self.args.d_model, self.args.num_features, kernel_size=1)

        # test_vision
        self.test_vision = False


    @staticmethod
    def svd_init(apt_size, aptinit):
        m, p, n = torch.svd(aptinit)
        nodevec1 = torch.mm(m[:, :apt_size], torch.diag(p[:apt_size] ** 0.5))
        nodevec2 = torch.mm(torch.diag(p[:apt_size] ** 0.5), n[:, :apt_size].t())
        return nodevec1, nodevec2

    def visual_Attention(self, A, save_path):
        # 这一版里面输入的A是[support_explicit,support_implicit]
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        tmp = A[0][0, :].clone()
        plt.figure()
        sns.heatmap(tmp.detach().cpu().numpy(), annot=False, cmap='coolwarm',vmin=0,vmax=1)
        plt.title('Attention_explicit')
        # plt.legend()
        plt.savefig(os.path.join(save_path, 'Attention_explicit'))
        plt.close()

        tmp = A[1][0, :].clone()
        plt.figure()
        sns.heatmap(tmp.detach().cpu().numpy(), annot=False, cmap='coolwarm',vmin=0,vmax=1)
        plt.title('Attention_implicit')
        # plt.legend()
        plt.savefig(os.path.join(save_path, 'Attention_implicit'))
        plt.close()
        return

    def visual_Ori_Attention(self, A, save_path):
        # 可视化原始的输入Adj
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        tmp = A[:, :].clone()
        plt.figure()
        sns.heatmap(tmp.detach().cpu().numpy(), annot=False, cmap='coolwarm', vmin=0, vmax=1)
        plt.title('Attention_Original')
        # plt.legend()
        plt.savefig(os.path.join(save_path, 'Attention_Original'))
        plt.close()
        return

    def Embedding(self, input_data, seq_time):
        hour = (seq_time[:, -2:-1, ...] + 0.5) * 23  # 得到第几个小时
        min = (seq_time[:, -1:, ...] + 0.5) * 59  # 得到第几分钟
        hour_index = (hour * 60 + min) / (60 / self.args.points_per_hour)
        time_in_day_emb = self.time_in_day_emb[
            hour_index[..., -1].squeeze(1).repeat(1, self.args.num_nodes).type(torch.LongTensor)]  # (B,N,D)
        day = (seq_time[:, 2:3, ...] + 0.5) * (6 - 0)
        day_in_week_emb = self.day_in_week_emb[
            day[..., -1].squeeze(1).repeat(1, self.args.num_nodes).type(torch.LongTensor)]
        # time series embedding
        batch_size, _, num_nodes, _ = input_data.shape
        input_data = input_data.transpose(1, 2).contiguous()
        input_data = input_data.view(batch_size, num_nodes, -1).transpose(1, 2).unsqueeze(-1)
        time_series_emb = self.time_series_emb_layer(input_data)
        node_emb = []
        node_emb.append(self.node_emb.unsqueeze(0).expand(batch_size, -1, -1).transpose(1, 2).unsqueeze(-1))
        # temporal embeddings
        tem_emb = []
        tem_emb.append(time_in_day_emb.transpose(1, 2).unsqueeze(-1))
        tem_emb.append(day_in_week_emb.transpose(1, 2).unsqueeze(-1))
        # concate all embeddings
        hidden = torch.cat([time_series_emb] + node_emb + tem_emb, dim=1)
        # encoding
        hidden = self.encoder(hidden)
        # regression
        hidden = self.regression_layer(hidden).squeeze(-1)
        B, D, N = hidden.shape
        hidden = hidden.reshape(B, self.args.num_features, self.args.seq_len, N)
        hidden = hidden.permute(0, 1, -1, -2)  # (B,C,N,L)
        hidden = self.bn_hidden(hidden)
        return hidden  # (B,C,N,L)


    def forward(self, x,adj,**kwargs):
        # Input shape is (bs, features, n_nodes, n_timesteps)
        mode = kwargs.get('mode')
        targets = kwargs.get('targets')
        seq_time = kwargs.get('seqs_time')
        targets_time = kwargs.get('targets_time')
        test_vision = self.test_vision

        original_adj = torch.tensor(adj).unsqueeze(0).repeat(x.shape[0], 1, 1).to(x.device)
        iter_pred_list = []
        B,D,N,L = x.shape

        # 每一次的动态图输入为上一次的输出，而不是原始的adj，目的是为了解决长期预测情况下精度不好的问题
        adj_dynamic = self.fixed_supports.unsqueeze(0).repeat(B, 1, 1)

        input_x = x
        time=seq_time

        # TODO embedding
        if self.embedding_use:
            emb = self.Embedding(input_data=input_x.clone(), seq_time=time)
            input_x_ = self.init_fc(input_x + emb)  # 特征维度升维

        else:
            input_x_ = self.init_fc(input_x)

        # TODO norm
        if self.norm_use:
            tnorm = self.t_norm(input_x_)
            snorm = self.s_norm(input_x_)
            input_x_ = torch.cat([input_x_, snorm, tnorm], dim=1)
            input_x_ = self.conv_norm(input_x_)

        # TODO adj_dynamic
        adj_dynamic = self.xlstm(input_x_, cell_past=adj_dynamic,mode=kwargs.get('mode'))  # 为了避免衰减太快，这里每次传入的还是初始的图
        adj_dynamic_with_explicit = torch.where(original_adj>0,torch.tanh(adj_dynamic),0)
        adj_dynamic_with_implicit = torch.where(original_adj==0,torch.tanh(adj_dynamic),0)
        adjacency_matrices=[adj_dynamic_with_explicit,adj_dynamic_with_implicit]


        adjacency_matrices = adjacency_matrices
        # output dimension[B,D=1,N,L_iter]

        # TODO Predictor
        iter_pred_output,season_graphs,trend_graphs,seasons = self.predictor(input_x_,adjacency_matrices,test_vision=test_vision)
        iter_pred_list.append(iter_pred_output.permute(0,3,1,2))

        preds = torch.cat(iter_pred_list,dim=-1)
        if test_vision:
            return preds,season_graphs,trend_graphs,seasons
        else:
            return preds



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

'''利用XLSTM的思想生成动态图'''
class XLSTM_dynamic_graph(nn.Module):
    def __init__(self,in_feature,d_model,save_path,num_nodes,pred_n,**kwargs):
        super().__init__()
        self.patch_len = 3
        self.stride = 3
        self.num_nodes = num_nodes
        self.pred_n=pred_n

        self.weight_pool_k = nn.Parameter(torch.FloatTensor(num_nodes, d_model, d_model),requires_grad=True)
        self.weight_pool_v = nn.Parameter(torch.FloatTensor(num_nodes, d_model, d_model),requires_grad=True)

        self.bias_pool_k = nn.Parameter(torch.FloatTensor(num_nodes, d_model),requires_grad=True)
        self.bias_pool_v = nn.Parameter(torch.FloatTensor(num_nodes, d_model),requires_grad=True)

        nn.init.xavier_normal_(self.weight_pool_k)
        nn.init.xavier_normal_(self.weight_pool_v)

        nn.init.xavier_normal_(self.bias_pool_k)
        nn.init.xavier_normal_(self.bias_pool_v)


        self.init_fc = Conv2d(in_feature*self.patch_len, d_model, (1, 1), padding=(0, 0), stride=(1, 1), bias=True)
        self.bn1 = nn.BatchNorm2d(d_model)

        self.input=nn.Linear(d_model,1)
        self.forget = nn.Linear(d_model, 1)
        # 以下是两个状态空间

        #FIXME 测试softplus约束输入门的输入
        self.q_k_activation = nn.Softplus()

        # weight pool 的后续层
        self.v_ln =nn.LayerNorm(d_model,elementwise_affine=True)
        self.v_d = nn.Dropout(0.1)

        self.k_ln = nn.LayerNorm(d_model,elementwise_affine=True)
        self.k_d = nn.Dropout(0.1)

        self.now_ln = nn.LayerNorm(num_nodes,elementwise_affine=True)

        self.save_path=save_path
        self.flag=True

        self.padding_patch = 'end'

        if self.padding_patch == 'end': # can be modified to general case
            self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))

    def visual_cell(self,A, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        tmp = A[0,:,:].clone()
        plt.figure()
        sns.heatmap(tmp.detach().cpu().numpy(), annot=False, cmap='coolwarm',vmin=0,vmax=1)
        plt.title('cell')
        # plt.legend()
        plt.savefig(os.path.join(save_path, f'cell_{self.flag}'))
        plt.close()
        return

    def forward(self,x,cell_past=None,normalize_past=None,m_past=None,**kwargs):
        mode=kwargs.get('mode')
        B,C,N,L=x.shape
        h_list=[]
        if cell_past!=None:
            cell_list = [cell_past]
        else:
            cell_list=[torch.zeros((B,N,N),device=device)]
        if normalize_past!=None:
            normalize_list=[normalize_past]
        else:
            normalize_list=[torch.zeros((B,N,1),device=device)]
        if m_past!=None:
            m_list=[m_past]
        else:
            m_list=[torch.zeros((B,N,1),device=device)]

        if self.padding_patch == 'end':
            z = x.view(B,-1,L)# True
            z = self.padding_patch_layer(z)
            z = z.view(B,C,N,-1)# 序列最后填充  序列最后一个值重复stride次，即边缘重复stride次

        z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        B,C,N,n,p = z.shape# z: [bs,dim,patch_num,patch_len]  类似卷积的切片操作
        z = z.view(B,C*p,N,n)

        z = self.init_fc(z)
        z = self.bn1(z)
        for i in range(z.shape[-1]):
            h,cell,normalize_now,m_now=self.step(z[...,i],cell_past=cell_list[-1],normalize_past=normalize_list[-1],m_past=m_list[-1])
            h_list.append(h);cell_list.append(cell);normalize_list.append(normalize_now);m_list.append(m_now)
            if mode == 'test' and self.flag:
                self.visual_cell(h,save_path=self.save_path)
                self.flag = False

        return h_list[-1]

    def step(self,xt,cell_past,normalize_past,m_past):
        '''xt (B,C,N);cell_past(B,N,N),normalize_past(B,N,1)'''
        xt=xt.transpose(-1,-2) #(B,N,C)
        I_gate=torch.sigmoid(self.input(xt))#(B,N,1)
        F_gate = 1 - I_gate# (B,N,1)

        m_now=m_past
        B,N,C=xt.shape

        # weight pool 映射
        # ----------------------------------------------
        key = torch.einsum('bnd,ndo->bno', xt, self.weight_pool_k) + self.bias_pool_k
        key = self.q_k_activation(self.k_d(self.k_ln(key)))
        key = key / math.sqrt(key.shape[-1])

        value = torch.einsum('bnd,ndo->bno', xt, self.weight_pool_v) + self.bias_pool_v
        value = self.q_k_activation(self.v_d(self.v_ln(value)))
        # ----------------------------------------------


        # -1测试过了 torch.sum(now[0,0,:])=1  F.softmax的dim和正常的切片索引不一样
        now = F.relu(self.now_ln(torch.matmul(key,value.transpose(-1,-2))))
        normal_key=torch.max(torch.abs(value),dim=-1,keepdim=True)[0] # 将H维度融掉
        normalize_now=torch.multiply(F_gate,normalize_past)+torch.multiply(I_gate,normal_key)
        tmp=torch.relu(normalize_now-1)+1 # 把小于1的都变为了1

        # 沿着行的方向计算最小值和最大值
        min_vals, _ = torch.min(now, dim=-1, keepdim=True)
        max_vals, _ = torch.max(now, dim=-1, keepdim=True)

        # 最小-最大缩放，将x的范围缩放到[0, 1]
        now_min_max = (now - min_vals) / (max_vals - min_vals)
        now_min_max = now_min_max/tmp

        # 二值化，降低小噪声的影响，稀疏化 可以改进为topk
        now_min_max = torch.where(now_min_max>=0.5,1,0)
        cell=torch.multiply(F_gate,cell_past)+torch.multiply(I_gate,now_min_max)#(B,N,N)
        h=cell

        return h,cell,normalize_now,m_now




class MultiLayerPerceptron(nn.Module):
    """Multi-Layer Perceptron with residual links."""

    def __init__(self, input_dim, hidden_dim) -> None:
        super().__init__()
        self.fc1 = nn.Conv2d(
            in_channels=input_dim,  out_channels=hidden_dim, kernel_size=(1, 1), bias=True)
        self.fc2 = nn.Conv2d(
            in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=(1, 1), bias=True)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=0.15)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """Feed forward of MLP.

        Args:
            input_data (torch.Tensor): input data with shape [B, D, N]

        Returns:
            torch.Tensor: latent repr
        """

        hidden = self.fc2(self.drop(self.act(self.fc1(input_data))))      # MLP
        hidden = hidden + input_data                           # residual
        return hidden


class SNorm(nn.Module):
    def __init__(self,  channels):
        super(SNorm, self).__init__()
        self.beta = nn.Parameter(torch.zeros(channels))
        self.gamma = nn.Parameter(torch.ones(channels))

    def forward(self, x):
        # input(B,C,N,T)
        x_norm = (x - x.mean(2, keepdims=True)) / (x.var(2, keepdims=True, unbiased=True) + 0.00001) ** 0.5 #这个是求节点的均值和方差

        out = x_norm * self.gamma.view(1, -1, 1, 1) + self.beta.view(1, -1, 1, 1)
        return out #(B,C,N,T)


# TODO 得到的是高频分量
class TNorm(nn.Module):
    def __init__(self,  num_nodes, channels, track_running_stats=True, momentum=0.1):
        super(TNorm, self).__init__()
        self.track_running_stats = track_running_stats
        self.beta = nn.Parameter(torch.zeros(1, channels, num_nodes, 1))
        self.gamma = nn.Parameter(torch.ones(1, channels, num_nodes, 1))
        self.register_buffer('running_mean', torch.zeros(1, channels, num_nodes, 1))
        self.register_buffer('running_var', torch.ones(1, channels, num_nodes, 1))
        self.momentum = momentum

    def forward(self, x):
        # input:x(B,C,N,T)
        if self.track_running_stats:#在batch_size维度和时间维度
            mean = x.mean((0, 3), keepdims=True)
            var = x.var((0, 3), keepdims=True, unbiased=False)
            if self.training:
                n = x.shape[3] * x.shape[0]
                # 以下的是指数加权平均求期望E，因为这个是整一个完整(数据集)的时间步上的均值和方差,因此这里采用指数加权平均的方法来近似得到
                with torch.no_grad():
                    self.running_mean = self.momentum * mean + (1 - self.momentum) * self.running_mean
                    self.running_var = self.momentum * var * n / (n - 1) + (1 - self.momentum) * self.running_var
            else:
                mean = self.running_mean
                var = self.running_var
        else:
            mean = x.mean((3), keepdims=True)
            var = x.var((3), keepdims=True, unbiased=True)
        x_norm = (x - mean) / (var + 0.00001) ** 0.5
        out = x_norm * self.gamma + self.beta
        return out # (B,C,N,T)





