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
from layers.StandNorm import Normalize
from torch.nn.utils import weight_norm



def linear_space(seq_len, pred_len, is_forecast=True):
    horizon = pred_len if is_forecast else seq_len
    return np.arange(0, horizon) / horizon

class FITS(nn.Module):
    def __init__(self, dominance_freq, output_len):
        super(FITS, self).__init__()
        self.fft_len = 16
        self.dominance_freq = dominance_freq
        self.output_len = output_len

        self.length_ratio = 1.5

        # Linear layers for frequency upsampling of the real and imaginary parts
        self.freq_upsampler_real = nn.Linear(self.dominance_freq,
                                             int(self.dominance_freq * self.length_ratio))  # complex layer for frequency upcampling]
        self.freq_upsampler_imag = nn.Linear(self.dominance_freq,
                                             int(self.dominance_freq * self.length_ratio))  # complex layer for frequency upcampling]

    def forward(self, input_x):
        # [B,N,L,D]
        B,N,L,D = input_x.shape

        x = input_x.reshape(-1,L,D) #[B*N,L,D]
        low_specx = torch.fft.rfft(x, n = self.fft_len,dim=1)  
        # Compute real FFT along the time dimension, output length is n//2 + 1 due to symmetry of real FFT
        low_specx = torch.view_as_real(low_specx[:, 0:self.dominance_freq, :])  
        # Keep only the dominant frequencies and view complex values as real-imag pairs
        low_specx_real = low_specx[:, :, :, 0]
        low_specx_imag = low_specx[:, :, :, 1]

        real = self.freq_upsampler_real(low_specx_real.permute(0, 2, 1)).permute(0, 2, 1)
        imag = self.freq_upsampler_imag(low_specx_imag.permute(0, 2, 1)).permute(0, 2, 1)
        low_specxy_real = real - imag
        low_specxy_imag = real + imag

        # Pad real and imaginary components back to the full FFT length
        low_specxy_R = torch.zeros(
            [low_specxy_real.size(0), self.fft_len//2 + 1, low_specxy_real.size(2)],
            dtype=low_specxy_real.dtype).to(low_specxy_real.device)  
        low_specxy_R[:, 0:low_specxy_real.size(1), :] = low_specxy_real

        low_specxy_I = torch.zeros(
            [low_specxy_imag.size(0), self.fft_len//2 + 1, low_specxy_imag.size(2)],
            dtype=low_specxy_imag.dtype).to(low_specxy_imag.device)  
        low_specxy_I[:, 0:low_specxy_imag.size(1), :] = low_specxy_imag

        low_specxy = torch.complex(low_specxy_R, low_specxy_I)
        low_xy = torch.fft.irfft(low_specxy, n=self.output_len, dim=1)
        low_xy = low_xy.reshape(B,N,-1,D)

        return low_xy  #[B,N]

class FITS_simple(nn.Module):
    def __init__(self, seq_len, output_len, d_model):
        super(FITS_simple, self).__init__()
        self.seq_len = seq_len
        self.pred_len = output_len
        self.scale = 0.02
        self.fft_len = self.seq_len

        self.embed_size = self.seq_len
        self.hidden_size = d_model

        self.w = nn.Parameter(self.scale * torch.randn(1, self.embed_size))

        self.fc = nn.Sequential(
            nn.Linear(self.embed_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.pred_len)
        )

    def circular_convolution(self, x, w):

        x = torch.fft.rfft(x, n=self.fft_len,dim=2, norm='ortho')
        w = torch.fft.rfft(w, n=self.fft_len,dim=1, norm='ortho')
        y = x * w.unsqueeze(0).unsqueeze(-1)
        out = torch.fft.irfft(y, n=self.embed_size, dim=2, norm="ortho")
        return out

    def forward(self, x):


        x = self.circular_convolution(x, self.w.to(x.device))  # B, N, D
        x = self.fc(x.permute(0,1,3,2)).permute(0,1,3,2)

        return x



class Nbeats_simple(nn.Module):
    """
    Simplified N-BEATS block that enforces a seasonal component output
    """
    def __init__(self, seq_len, output_len, d_model):
        super(Nbeats_simple, self).__init__()

        thetas_dim = 8
        self.theta_b_fc = nn.Linear(seq_len, thetas_dim, bias=False)
        self.backcast_linspace = linear_space(seq_len, seq_len, is_forecast=False)

    def seasonality_model(self,thetas, t, device):
        p = thetas.size()[-1]
        assert p <= thetas.shape[2], 'thetas_dim is too big.'
        p1, p2 = (p // 2, p // 2) if p % 2 == 0 else (p // 2, p // 2 + 1)
        # Construct cosine and sine basis functions for seasonality
        s1 = torch.tensor(np.array([np.cos(2 * np.pi * i * t) for i in range(p1)])).float()  # H/2-1
        s2 = torch.tensor(np.array([np.sin(2 * np.pi * i * t) for i in range(p2)])).float()
        S = torch.cat([s1, s2])
        # Compute seasonality output via tensor contraction
        seasonality_output = torch.einsum("bdn, nl->bdl", thetas, S.to(device))


        return seasonality_output

    def forward(self, x):
        # Reshape from [B, N, L, D] to [B*N, D, L] for processing
        B,N,L,D = x.shape
        x = x.permute(0,1,3,2).reshape(B*N,D,L)
        theta = self.theta_b_fc(x)
        s = self.seasonality_model(theta,self.backcast_linspace,x.device)

        _,D,o = s.shape
        # s shape: [B*N, D, output_len]
        output = s.reshape(B,N,D,o)

        # Return tensor with shape [B, N, output_len, D]
        return output.permute(0,1,3,2)



class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        if type(self.kernel_size) == list:
            if len(self.kernel_size) == 1:
                self.kernel_size = self.kernel_size[0]
        front = x[:, 0:1, :].repeat(1, self.kernel_size - 1-math.floor((self.kernel_size - 1) // 2), 1)
        end = x[:, -1:, :].repeat(1, math.floor((self.kernel_size - 1) // 2), 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class series_decomp_FFT(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self,fft_len, top_k):
        super(series_decomp_FFT, self).__init__()
        self.fft_len = fft_len
        self.k = top_k

    def main_freq_part(self,x):
        # freq normalization
        # start = time.time()
        xf = torch.fft.rfft(x, n=self.fft_len,dim=1)
        k_values = torch.topk(xf.abs(), self.k, dim=1)
        indices = k_values.indices

        mask = torch.zeros_like(xf)
        mask.scatter_(1, indices, 1)
        xf_filtered = xf * mask

        # main frequency
        x_filtered = torch.fft.irfft(xf_filtered, n=x.shape[1],dim=1).real.float()
        # res
        norm_input = x - x_filtered
        # print(f"decompose take:{ time.time() - start} s")
        return x_filtered, norm_input

    def forward(self, x):
        S,R = self.main_freq_part(x)
        return S,R





class TemporalConvNet(nn.Module):
    """
    Temporal Convolutional Network (TCN) class
    """

    def __init__(self, input_size, output_size, num_layers, kernel_size=2, dropout=0.2):
        """
        Constructor

        Parameters:
        input_size (int): Dimensionality of input features
        output_size (int): Dimensionality of output features
        num_layers (int): Number of TCN layers
        kernel_size (int): Size of convolutional kernel
        dropout (float): Dropout rate
        """
        super(TemporalConvNet, self).__init__()
        layers = []

        for i in range(num_layers):
            dilation_size = 2 ** i
            in_channels = input_size
            out_channels = output_size

            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     dropout=dropout)]

        self.network = nn.Sequential(*layers)


    def forward(self, x):
        """
        Forward pass

        Parameters:
        x (torch.Tensor): Input sequence of shape [B, N, L, D]

        Returns:
        torch.Tensor: Output sequence of shape [B, output_size, L]
        """
        # Reshape to [B*N, D, L]
        B,N,L,D = x.shape
        x = x.permute(0, 1, 3, 2).contiguous().view(-1, D, L)

        # Pass through TCN layers
        x = self.network(x)

        # Reshape back to [B, N, C, L]
        _, C, L = x.size()
        x = x.view(B,N, C, L)

        return x[:,:,:,-1]


class TemporalBlock(nn.Module):
    """
    Temporal Block, contains a dilated convolution layer and a ResNet-style skip connection
    """

    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.padding = (kernel_size - 1) * dilation


        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, dilation=dilation))
        self.chomp1 = Chomp1d(self.padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, dilation=dilation))

        self.chomp2 = Chomp1d(self.padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net_1 = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 )
        self.net_2 = nn.Sequential(self.conv2, self.chomp2, self.relu2, self.dropout2)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        res = x
        padding_layer = nn.ConstantPad1d((self.padding*2, 0), 0)
        padded_x = padding_layer(x)
        out_1 = self.net_1(padded_x)
        padded_x = padding_layer(out_1)
        out = self.net_2(padded_x)

        return self.relu(out + res)


class Chomp1d(nn.Module):
    """
    Chomp layer to remove the padding added by the convolutional layer
    """

    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, self.chomp_size:].contiguous()




class Graph_Generator(nn.Module):
    def __init__(self, channels=128, num_nodes=170, diffusion_step=1, dropout=0.1):
        super().__init__()
        self.memory = nn.Parameter(torch.randn(channels, num_nodes))
        nn.init.xavier_uniform_(self.memory)
        self.fc = nn.Linear(2, 1)

    def forward(self, x):
        x = x.permute(0,3,1,2)
        adj_dyn_1 = torch.softmax(F.relu(torch.einsum("bcnl, cm->bnm", x, self.memory).contiguous()/ math.sqrt(x.shape[1])),-1,)
        adj_dyn_2 = torch.softmax(F.relu(torch.einsum("bcn, bcm->bnm", x.sum(-1), x.sum(-1)).contiguous()/ math.sqrt(x.shape[1])),-1,)

        adj_f = torch.cat([(adj_dyn_1).unsqueeze(-1)] + [(adj_dyn_2).unsqueeze(-1)], dim=-1)
        adj_f = self.fc(adj_f).squeeze()
        
        # Enforce sparsity by keeping top-80% strongest connections
        topk_values, topk_indices = torch.topk(adj_f, k=int(adj_f.shape[1] * 0.8), dim=-1)
        mask = torch.zeros_like(adj_f)
        mask.scatter_(-1, topk_indices, 1)
        adj_f = adj_f * mask  
        adj_f = torch.softmax(adj_f,dim=-1)

        return adj_f

class Adj_Generator(nn.Module):
    def __init__(self, channels=128, num_nodes=170, diffusion_step=1, dropout=0.1):
        super().__init__()
        self.memory = nn.Parameter(torch.randn(channels, num_nodes),requires_grad=True)
        #nn.init.normal_(self.memory, mean=0.0, std=1)

        self.fc = nn.Linear(2, 1)

    def forward(self, x):
        # Trend Item
        if len(x.shape)==3: 
            # x[B,N,D]
            x = x.permute(0,2,1)
            adj_dyn_1 = torch.tanh((torch.einsum("bcn, cm->bnm", x, self.memory).contiguous()/ math.sqrt(x.shape[1])))
            adj_dyn_2 = torch.tanh((torch.einsum("bcn, bcm->bnm", x, x).contiguous()/ math.sqrt(x.shape[1])))
        # Season Item
        else:     
            # x[B,N,L,D]
            x = x.permute(0, 3, 1, 2)
            adj_dyn_1 = torch.tanh((torch.einsum("bcnl, cm->bnm", x, self.memory).contiguous() / math.sqrt(x.shape[1])))
            adj_dyn_2 = torch.tanh((torch.einsum("bcn, bcm->bnm", x.sum(-1), x.sum(-1)).contiguous() / math.sqrt(x.shape[1])))

        adj_f = torch.cat([(adj_dyn_1).unsqueeze(-1)] + [(adj_dyn_2).unsqueeze(-1)], dim=-1)
        adj_f = torch.sigmoid(self.fc(adj_f).squeeze(-1))

        return adj_f


def nconv(x, A):
    """Multiply x by adjacency matrix along source node axis"""
    return torch.einsum('ncvl,vw->ncwl', (x, A.to(x.device))).contiguous()



class MultiScaleSeasonMixing(nn.Module):
    """
    Bottom-up mixing season pattern
    """

    def __init__(self, configs):
        super(MultiScaleSeasonMixing, self).__init__()

        self.down_sampling_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                    ),
                    nn.GELU(),
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                    ),

                )
                for i in range(configs.down_sampling_layers)
            ]
        )


    def forward(self, season_list):

        # mixing high->low
        out_high = season_list[0]
        out_low = season_list[1]
        out_season_list = [out_high.permute(0, 2, 1)]

        for i in range(len(season_list) - 1):
            out_low_res = self.down_sampling_layers[i](out_high) 
            out_low = out_low + out_low_res
            out_high = out_low
            if i + 2 <= len(season_list) - 1:
                out_low = season_list[i + 2]
            out_season_list.append(out_high.permute(0, 2, 1))

        return out_season_list


class MultiScaleTrendMixing(nn.Module):
    """
    Top-down mixing trend pattern
    """

    def __init__(self, configs):
        super(MultiScaleTrendMixing, self).__init__()

        self.up_sampling_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                        configs.seq_len // (configs.down_sampling_window ** i),
                    ),
                    nn.GELU(),
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.seq_len // (configs.down_sampling_window ** i),
                    ),
                )
                for i in reversed(range(configs.down_sampling_layers))
            ])

    def forward(self, trend_list):

        # mixing low->high
        trend_list_reverse = trend_list.copy()
        trend_list_reverse.reverse()
        out_low = trend_list_reverse[0]
        out_high = trend_list_reverse[1]
        out_trend_list = [out_low.permute(0, 2, 1)]

        for i in range(len(trend_list_reverse) - 1):
            out_high_res = self.up_sampling_layers[i](out_low)
            out_high = out_high + out_high_res
            out_low = out_high
            if i + 2 <= len(trend_list_reverse) - 1:
                out_high = trend_list_reverse[i + 2]
            out_trend_list.append(out_low.permute(0, 2, 1))

        out_trend_list.reverse()
        return out_trend_list

class PastDecomposableMixing(nn.Module):
    def __init__(self, configs, down_sampling_window,support_len):
        super(PastDecomposableMixing, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.down_sampling_window = down_sampling_window
        self.num_nodes = configs.num_nodes

        self.layer_norm = nn.LayerNorm(configs.d_model)
        self.dropout = nn.Dropout(0.1)

        channel_independence_timemixer = 0
        self.channel_independence = channel_independence_timemixer

        topk = 4
        self.decompsition = series_decomp_FFT(self.seq_len,topk)

        if channel_independence_timemixer == 0:
            self.cross_layer = nn.Sequential(
                nn.Linear(in_features=configs.d_model, out_features=configs.d_ff),
                nn.GELU(),
                nn.Linear(in_features=configs.d_ff, out_features=configs.d_model),
            )

        # Mixing season
        self.mixing_multi_scale_season = MultiScaleSeasonMixing(configs)

        # Mxing trend
        self.mixing_multi_scale_trend = MultiScaleTrendMixing(configs)

        self.out_cross_layer = nn.Sequential(
            nn.Linear(in_features=configs.d_model, out_features=configs.d_ff),
            nn.GELU(),
            nn.Linear(in_features=configs.d_ff, out_features=configs.d_model),
        )

        self.season_graph_generator = torch.nn.ModuleList(
            [Adj_Generator(configs.d_model,configs.num_nodes)
             for _ in range(configs.down_sampling_layers+1)])
        self.trend_graph_generator = torch.nn.ModuleList(
            [Adj_Generator(configs.d_model, configs.num_nodes)
             for _ in range(configs.down_sampling_layers + 1)])

        self.season_graph_conv = GraphConvNet(configs.d_model, configs.d_model,0.1, support_len=support_len ,order=2)
        self.trend_graph_conv = GraphConvNet(configs.d_model, configs.d_model, 0.1, support_len=support_len, order=2)

        self.trend_process = TemporalConvNet(configs.d_model,configs.d_model,1)
        #self.season_process = FITS_simple(seq_len=self.seq_len, output_len=self.pred_len,d_model=configs.d_model)
        self.season_process = torch.nn.ModuleList(
            [Nbeats_simple(seq_len=configs.seq_len // (configs.down_sampling_window ** (i)), output_len=self.pred_len, d_model=configs.d_model)
             for i in range(configs.down_sampling_layers+1)])


    def forward(self, x_list, adj):
        length_list = []
        for x in x_list:
            _, _, T = x.shape
            length_list.append(T)

        # Decompose to obtain the season and trend
        seasons = []

        season_list = []
        trend_list = []

        season_graph_list = []
        trend_graph_list = []

        for i in range(len(x_list)):
            x = x_list[i]
            _,D,L = x.shape
            season, trend = self.decompsition(x.permute(0,2,1))
            season = season.reshape(-1,self.num_nodes,L,D)
            trend = trend.reshape(-1,self.num_nodes,L,D)

            # trend process
            trend_state = self.trend_process(trend) 

            # season process
            season_state = self.season_process[i](season) 

            seasons.append(season_state)

            season_adj = self.season_graph_generator[i](season_state)
            trend_adj = self.trend_graph_generator[i](trend_state)

            season_graph_list.append(season_adj)
            trend_graph_list.append(trend_adj)

            if self.channel_independence == 0:
                season = self.season_graph_conv(season,adj+[season_adj]).reshape(-1,D,L)
                trend = self.trend_graph_conv(trend,adj+[trend_adj]).reshape(-1,D,L)
            season_list.append(season)
            trend_list.append(trend)

        # bottom-up season mixing
        out_season_list = self.mixing_multi_scale_season(season_list)
        # top-down trend mixing
        out_trend_list = self.mixing_multi_scale_trend(trend_list)

        out_list = []
        for ori, out_season, out_trend, length in zip(x_list, out_season_list, out_trend_list,
                                                      length_list):
            out = out_season + out_trend
            if self.channel_independence:
                out = ori + self.out_cross_layer(out)
            out_list.append(out[:, :length, :].permute(0,2,1))



        return out_list, season_graph_list, trend_graph_list, seasons



class DPG_Mixer_Predictor(nn.Module):
    def __init__(self, num_nodes, in_dim, out_dim,d_model,d_ff,pred_len,support_len,
                 dropout=0.3, kernel_size=2, blocks=4, layers=1,configs=None,**kwargs):
        super().__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.configs=configs
        self.d_model = d_model
        self.d_ff = d_ff

        self.down_sampling_window = configs.down_sampling_window



        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                        out_channels=d_model,
                                        kernel_size=(1, 1))

        self.supports_len = 2
        self.pred_len = pred_len
        self.out_dim = out_dim

        # self.end_conv = nn.Sequential(nn.Conv2d(skip_channels, end_channels, (1, 1), bias=True),
        #                               nn.ReLU(inplace=True),
        #                               nn.Conv2d(end_channels, out_dim * pred_len, (1, 1), bias=True))

        self.down_sampling_layers = configs.down_sampling_layers


        self.normalize_layers = torch.nn.ModuleList(
            [
                Normalize(self.d_model, affine=True, non_norm=False)
                for i in range(self.down_sampling_layers +1)
            ]
        )

        self.layers = 2
        self.pdm_blocks = nn.ModuleList([PastDecomposableMixing(self.configs,self.down_sampling_window,support_len)
                                         for _ in range(self.down_sampling_layers+1)])

        self.predict_layers = torch.nn.ModuleList(
                [
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        self.pred_len,
                    )
                    for i in range(configs.down_sampling_layers + 1)
                ]
            )


        self.projection_layer = nn.Linear(
                configs.d_model, out_dim, bias=True)

        self.out_res_layers = torch.nn.ModuleList([
                torch.nn.Linear(
                    configs.seq_len // (configs.down_sampling_window ** i),
                    configs.seq_len // (configs.down_sampling_window ** i),
                )
                for i in range(configs.down_sampling_layers + 1)
            ])

        self.regression_layers = torch.nn.ModuleList(
                [
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        self.pred_len,
                    )
                    for i in range(configs.down_sampling_layers + 1)
                ]
            )


    def _multi_scale_process_inputs(self, x_enc):
        B,D,N,L = x_enc.shape
        x_enc = x_enc.permute(0,2,1,3).reshape(B*N,D,L)
        down_pool = torch.nn.AvgPool1d(self.down_sampling_window)

        x_enc_ori = x_enc
        x_enc_sampling_list = []
        x_enc_sampling_list.append(x_enc)

        for i in range(self.down_sampling_layers):
            x_enc_sampling = down_pool(x_enc_ori)
            x_enc_sampling_list.append(x_enc_sampling)
            x_enc_ori = x_enc_sampling

        x_enc = x_enc_sampling_list

        return x_enc

    def out_projection(self, dec_out, i, out_res):

        out_res = self.out_res_layers[i](out_res)
        out_res = self.regression_layers[i](out_res).permute(0, 2, 1)
        dec_out = dec_out + out_res
        dec_out = self.projection_layer(dec_out)
        return dec_out


    def future_multi_mixing(self, B, enc_out_list, x_list):
        dec_out_list = []

        for i, enc_out, out_res in zip(range(len(x_list[0])), enc_out_list, x_list):
            dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(
                0, 2, 1)  # align temporal dimension
            dec_out = self.out_projection(dec_out, i, out_res)
            dec_out_list.append(dec_out)

        return dec_out_list


    def forward(self, x,adj,**kargs):
        # adj = [adj_dynamic_with_explicit,adj_dynamic_with_implicit]
        test_vision = kargs.get('test_vision')
        b,D,n,L = x.shape
        x = self.start_conv(x)  
        adjacency_matrices=adj

        x_list = self._multi_scale_process_inputs(x)
        B, N, T = x_list[0].size()
        enc_out_list = [value.clone() for value in x_list]

        # Past Decomposable Mixing as encoder for past
        for i in range(self.layers):
            enc_out_list,season_graphs,trend_graphs, seasons = self.pdm_blocks[i](enc_out_list,adjacency_matrices)

        enc_out_list = [value.permute(0,2,1) for value in enc_out_list]
        dec_out_list = self.future_multi_mixing(B, enc_out_list, x_list)
        dec_out = torch.stack(dec_out_list, dim=-1).sum(-1)
        pred = dec_out.reshape(b,n,self.pred_len,self.out_dim)

        return pred, season_graphs, trend_graphs, seasons


def nconv(x, A):
    """Multiply x by adjacency matrix along source node axis"""
    return torch.einsum('ncvl,nvw->ncwl', (x, A.to(x.device))).contiguous()


class GraphConvNet(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=2, order=2):
        super().__init__()
        c_in = (order * support_len + 1) * c_in
        self.final_conv = Conv2d(c_in, c_out, (1, 1), padding=(0, 0), stride=(1, 1), bias=True)
        self.dropout = dropout
        self.order = order
        self.nconv_norm = torch.nn.BatchNorm2d(c_out)

    def forward(self, x, support: list):
        B,N,_ = support[0].shape
        _,N,D,L = x.shape
        graph_in = x.permute(0,3,1,2)
        out = [graph_in]
        for a in support:
            x1 = self.nconv_norm(nconv(graph_in, a))
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv_norm(nconv(x1, a))
                out.append(x2)
                x1 = x2

        h = torch.cat(out, dim=1)
        h = self.final_conv(h)
        h = F.dropout(h, self.dropout, training=self.training)
        graph_out = h.permute(0,2,1,3)
        return graph_out