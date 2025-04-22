from torch import nn
import torch
from layers.STIDGCN_related import TemporalEmbedding, GLU
import math
import torch.nn.functional as F
from torch_utils.graph_process import *
import torch
import matplotlib.pyplot as plt
import os
from torch.nn import BatchNorm2d, Conv1d, Conv2d, ModuleList, Parameter
import seaborn as sns
import torch.nn.init as init
device=torch.device('cuda'if torch.cuda.is_available() else 'cpu')

'''dynamic_patch_graph用的是v5版本的'''

class DGCN(nn.Module):
    def __init__(self, channels=128, num_nodes=170, diffusion_step=1, dropout=0.1, emb=None):
        super().__init__()
        self.emb = emb
        self.diffusion_step = diffusion_step

        self.dropout = nn.Dropout(dropout)
        self.conv1 = nn.Conv2d(channels, channels, (1, 1))
        self.conv2 = nn.Conv2d(channels*2, channels, (1, 1))

    def forward(self, x, supports):
        skip = x
        x = self.conv1(x)
        out = []
        for adj in supports:
            for i in range(0, self.diffusion_step):
                if adj.dim() == 3:
                    x = torch.einsum("bcnl,bnm->bcml", x, adj).contiguous()
                    out.append(x)
                elif adj.dim() == 2:
                    x = torch.einsum("bcnl,nm->bcml", x, adj).contiguous()
                    out.append(x)
        x = torch.cat(out, dim=1)
        x = self.conv2(x)
        x = self.dropout(x)

        x = x * self.emb + skip
        return x

class Splitting(nn.Module):  # 按奇偶时间步进行分离
    def __init__(self):
        super(Splitting, self).__init__()

    def even(self, x):
        return x[:, :, :, ::2]

    def odd(self, x):
        return x[:, :, :, 1::2]

    def forward(self, x):
        return (self.even(x), self.odd(x))


class IDGCN(nn.Module):
    def __init__(
            self,
            channels=64,
            diffusion_step=1,
            splitting=True,
            num_nodes=170,
            dropout=0.2, emb=None
    ):
        super(IDGCN, self).__init__()

        self.dropout = dropout
        self.num_nodes = num_nodes
        self.splitting = splitting
        self.split = Splitting()

        pad_l = 3
        pad_r = 3
        k1 = 5
        k2 = 3

        TConv = [
            nn.ReplicationPad2d((pad_l, pad_r, 0, 0)),
            nn.Conv2d(channels, channels, kernel_size=(1, k1)),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(self.dropout),
            nn.Conv2d(channels, channels, kernel_size=(1, k2)),
            nn.Tanh(),
        ]

        self.TSConv = nn.ModuleList([nn.Sequential(*TConv) for i in range(4)])
        self.dgcn = DGCN(channels, num_nodes, diffusion_step, dropout, emb)

    def forward(self, x, supports):
        if self.splitting:
            (x_even, x_odd) = self.split(x)
        else:
            (x_even, x_odd) = x

        x1 = self.TSConv[0](x_even)
        x1 = self.dgcn(x1, supports)
        d = x_odd.mul(torch.tanh(x1))

        x2 = self.TSConv[1](x_odd)
        x2 = self.dgcn(x2, supports)
        c = x_even.mul(torch.tanh(x2))

        x3 = self.TSConv[2](c)
        x3 = self.dgcn(x3, supports)
        x_odd_update = d + x3

        x4 = self.TSConv[3](d)
        x4 = self.dgcn(x4, supports)
        x_even_update = c + x4

        return [x_even_update, x_odd_update]

'''IDGCN树'''
class IDGCN_Tree(nn.Module):
    def __init__(
            self, seq_len, channels=64, diffusion_step=1, num_nodes=170, dropout=0.1
    ):
        super().__init__()

        self.memory = nn.ParameterList([nn.Parameter(torch.randn(channels, num_nodes, seq_len//2)),
                                        nn.Parameter(torch.randn(channels, num_nodes, seq_len//4)),
                                        nn.Parameter(torch.randn(channels, num_nodes, seq_len//4))])


        self.IDGCN = nn.ModuleList([IDGCN(
            splitting=True,
            channels=channels,
            diffusion_step=diffusion_step,
            num_nodes=num_nodes,
            dropout=dropout, emb=self.memory[i]
        ) for i in range(3)])

    def concat(self, even, odd):
        even = even.permute(3, 1, 2, 0)
        odd = odd.permute(3, 1, 2, 0)
        len = even.shape[0]
        _ = []
        for i in range(len):
            _.append(even[i].unsqueeze(0))
            _.append(odd[i].unsqueeze(0))
        return torch.cat(_, 0).permute(3, 1, 2, 0)

    def forward(self, x, supports):
        x_even_update1, x_odd_update1 = self.IDGCN[0](x, supports)
        x_even_update2, x_odd_update2 = self.IDGCN[1](x_even_update1, supports)
        x_even_update3, x_odd_update3 = self.IDGCN[2](x_odd_update1, supports)
        concat1 = self.concat(x_even_update2, x_odd_update2)
        concat2 = self.concat(x_even_update3, x_odd_update3)
        concat0 = self.concat(concat1, concat2)
        output = concat0 + x
        return output

class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)

class STIDGCN_dynamic_patch_graph(nn.Module):
    def __init__(
        self, args, supports, input_dim, num_nodes, channels, granularity,input_len,output_len,  points_per_hour, dropout=0.1
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.output_len = output_len
        self.input_len=input_len
        self.points_per_hour = points_per_hour
        diffusion_step = 1

        self.Temb = TemporalEmbedding(granularity, channels, points_per_hour, num_nodes)

        self.start_conv = nn.Conv2d(
            in_channels=input_dim, out_channels=channels, kernel_size=(1, 1)
        )

        self.tree = IDGCN_Tree(
            seq_len = self.input_len,
            channels=channels*2,
            diffusion_step=diffusion_step,
            num_nodes=self.num_nodes,
            dropout=dropout,
        )

        self.glu = GLU(channels*2, dropout)

        self.regression_layer = nn.Conv2d(
            channels*2, input_dim, kernel_size=(1, 1)
        )
        self.out=nn.Linear(self.input_len,self.output_len)

        self.args = args
        supports = transition_matrix(supports)
        self.fixed_supports = nn.Parameter(supports, requires_grad=False)
        self.xlstm=XLSTM_dynamic_graph(in_feature=input_dim, d_model=args.d_model, save_path=args.output_dir, num_nodes=num_nodes,pred_n=args.pred_len)
        self.flag = True

    def forward(self, input,adj,**kwargs):
        x = input

        mode = kwargs.get('mode')
        original_adj = torch.tensor(adj).unsqueeze(0).repeat(x.shape[0], 1, 1).to(x.device)

        adj_dynamic = self.xlstm(x, cell_past=self.fixed_supports, mode=kwargs.get('mode'))
        adj_dynamic_with_explicit = torch.where(original_adj > 0, torch.tanh(adj_dynamic), 0)
        adj_dynamic_with_implicit = torch.where(original_adj == 0, torch.tanh(adj_dynamic), 0)

        adjacency_matrices = [adj_dynamic_with_explicit, adj_dynamic_with_implicit]

        if mode == 'test' and self.flag == True:
            self.visual_Ori_Attention(self.fixed_supports, save_path=self.args.output_dir)
            self.visual_Attention(adjacency_matrices, save_path=self.args.output_dir)
            self.flag = False

        seqs_time=kwargs.get("seqs_time")
        targets_time=kwargs.get("targets_time")
        # Encoder
        # Data Embedding
        time_emb = self.Temb(seqs_time)  # output:(B,C,N,L)
        x = torch.cat([self.start_conv(x)] + [time_emb], dim=1)
        # IDGCN_Tree
        x = self.tree(x, adjacency_matrices)
        # Decoder
        gcn = self.glu(x) + x
        prediction = self.regression_layer(F.relu(gcn))
        prediction=self.out(prediction)
        return prediction


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
        sns.heatmap(tmp.detach().cpu().numpy(), annot=False, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Attention_explicit')
        # plt.legend()
        plt.savefig(os.path.join(save_path, 'Attention_explicit'))
        plt.close()

        tmp = A[1][0, :].clone()
        plt.figure()
        sns.heatmap(tmp.detach().cpu().numpy(), annot=False, cmap='coolwarm', vmin=-1, vmax=1)
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

'''利用XLSTM的思想生成动态图'''
class XLSTM_dynamic_graph(nn.Module):
    def __init__(self,in_feature,d_model,save_path,num_nodes,pred_n,**kwargs):
        super().__init__()
        self.patch_len = 3
        self.stride = 3
        self.num_nodes = num_nodes
        self.pred_n=pred_n

        self.weight_pool_k = nn.Linear(d_model,d_model)
        self.weight_pool_v = nn.Linear(d_model,d_model)

        self.init_fc = nn.Conv2d(in_feature*self.patch_len, d_model, (1, 1), padding=(0, 0), stride=(1, 1), bias=True)
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
            cell_list=[torch.zeros((B,N,N),device=x.device)]
        if normalize_past!=None:
            normalize_list=[normalize_past]
        else:
            normalize_list=[torch.zeros((B,N,1),device=x.device)]
        if m_past!=None:
            m_list=[m_past]
        else:
            m_list=[torch.zeros((B,N,1),device=x.device)]

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

        #TODO simple:线性映射
        # ----------------------------------------------
        key = self.weight_pool_k(xt)
        key = self.q_k_activation(self.k_d(self.k_ln(key)))
        key = key / math.sqrt(key.shape[-1])

        value = self.weight_pool_v(xt)
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