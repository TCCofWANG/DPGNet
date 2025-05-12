from layers.PMC_GCN_related import *
import torch.nn as nn



class PMC_GCN(nn.Module):
    def __init__(
            self,
            seq_len,
            adj,
            in_channels,
            embed_size,
            time_num,
            num_layers,
            T_dim,
            output_T_dim,
            heads,
            cheb_K,
            forward_expansion,
            dropout
    ):
        super(PMC_GCN, self).__init__()
        self.device = 'cuda'
        self.forward_expansion = forward_expansion
        self.dropout = dropout
        self.conv1 = nn.Conv2d(in_channels, embed_size, 1)
        self.Transformer = Transformer(
            seq_len,
            adj,
            embed_size,
            num_layers,
            heads,
            time_num,
            forward_expansion,
            cheb_K,
            device=self.device,
            dropout=self.dropout
        )

        # Reduce the time dimension. 
        # Example: T_dim=12 to output_T_dim=3, input 12 dimensions reduced to output 3 dimensions
        self.conv2 = nn.Conv2d(T_dim, output_T_dim, 1)
        # Reduce the number of channels to 1 dimension
        self.conv3 = nn.Conv2d(embed_size, 1, 1)
        self.relu = nn.ReLU()

    def forward(self, x,adj,**kwargs):
        # input x shape [C, N, T]
        # C: number of channels. N: number of sensors. T: number of time steps
        input_Transformer = self.conv1(x)
        input_Transformer = input_Transformer.permute(0, 2, 3, 1)
        # input_Transformer shape[N, T, C]   [B, N, T, C]
        output_Transformer = self.Transformer(input_Transformer, self.forward_expansion)  # [B, N, T, C]
        output_Transformer = output_Transformer.permute(0, 2, 1, 3)
        # output_Transformer shape[B, T, N, C]
        out = self.relu(self.conv2(output_Transformer))    # Left-hand side out shape: [1, output_T_dim, N, C]
        out = out.permute(0, 3, 2, 1)           # Left-hand side out shape: [B, C, N, output_T_dim]
        out = self.conv3(out)                    # Left-hand side out shape: [B, 1, N, output_T_dim]

        return out  # [B,C=1,N, output_dim]





