from torch_utils.graph_process import *
import torch.nn as nn
import torch

### Transformer
class Transformer(nn.Module):
    def __init__(
        self,
        seq_len,
        adj,
        embed_size,
        num_layers,
        heads,
        time_num,
        forward_expansion,
        cheb_K,
        device,
        dropout
    ):
        super(Transformer, self).__init__()
        self.device = device
        self.encoder = Encoder(
            seq_len,
            embed_size,
            num_layers,
            heads,
            adj,
            time_num,
            device,
            forward_expansion,
            cheb_K,
            dropout
        )


    def forward(self, src, t):
        ## scr: [N, T, C]   [B, N, T, C]
        enc_src = self.encoder(src, t)
        return enc_src # [B, N, T, C]


### Encoder
class Encoder(nn.Module):
    # Stacked Multi-layer ST-Transformer Block
    def __init__(
        self,
        seq_len,
        embed_size,
        num_layers,
        heads,
        adj,
        time_num,
        device,
        forward_expansion,
        cheb_K,
        dropout,
    ):

        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.dropout = dropout
        self.layers = nn.ModuleList(
            [
                STTransformerBlock(
                    seq_len,
                    embed_size,
                    heads,
                    adj,
                    time_num,
                    cheb_K,
                    dropout=self.dropout,
                    device=self.device,
                    forward_expansion=forward_expansion
                )
                for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, t):
    # x: [N, T, C]  [B, N, T, C]
        out = self.dropout(x)
        # In the Encoder the query, key, value are all the same.
        for layer in self.layers:
            out = layer(out, out, out, t)
        return out


class STTransformerBlock(nn.Module):
    def __init__(self, seq_len,embed_size, heads, adj, time_num, cheb_K, dropout, device, forward_expansion):
        super(STTransformerBlock, self).__init__()
        # self.STransformer = STransformer(embed_size, heads, adj, cheb_K, dropout, device, forward_expansion)
        # Replace STransformer model with ES-GCN model
        self.ES_GCN = ES_GCN(embed_size, adj, dropout, device)
        self.TTransformer = TTransformer(embed_size, heads, time_num, dropout, device, forward_expansion)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)
        self.complex_weight = nn.Parameter(torch.randn(1, 1, seq_len // 2 + 1, embed_size, 2, dtype=torch.float32) * 0.02)
        self.norm3 = nn.LayerNorm(embed_size)

    def forward(self, value, key, query, t):
        # value,  key, query: [N, T, C] [B, N, T, C]
        # Add skip connection,run through normalization and finally dropout

        # Output of ES-GCN model, replacing the value of x1, just need to ensure the format is correct
        x1 = self.norm1(self.ES_GCN(value, key, query) + query)

        ## Added code (Teacher)
        B, N, T, C = x1.shape
        query1 = x1
        x_rttf = torch.fft.rfft(query1, dim=2, norm='ortho') 
        #dim=2 is along the time dimension, 'ortho' means orthogonal normalization, i.e., dividing the Fourier transform result by the square root of the effective length
        weight = torch.view_as_complex(self.complex_weight)
        x_rttf = x_rttf * weight
        x_fft = torch.fft.irfft(x_rttf, n=T, dim=2, norm='ortho')
        hidden_states = x_fft
        x1 = self.norm3(hidden_states + query1)
        x2 = self.norm2(self.TTransformer(x1, x1, x1, t) + x1)
        return x2

class ES_GCN(nn.Module):
    def __init__(self, embed_size, adj, dropout, device):
        super(ES_GCN, self).__init__()
        self.adj = adj
        self.device = device
        self.D_S = adj.to(self.device)
        # Call GCN used in EST-GCN embed_size = input_dim; embed_size * 2 = hidden_dim; embed_size = output_dim
        self.gcn = GCN(adj, embed_size, embed_size, device)
        # Normalize adjacency matrix (ESTGCN way)
        self.register_buffer('laplacian', calculate_laplacian_with_self_loop(torch.FloatTensor(adj)))
        # self.norm_adj = nn.InstanceNorm2d(1)    # Normalize adjacency matrix
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query):
        # value, key, query: [N, T, C]  [B, N, T, C]
        # Spatial Embedding part
        B, N, T, C = query.shape
        D_S = get_sinusoid_encoding_table(N, C).to(self.device)
        D_S = D_S.expand(B, T, N, C)  # [B, T, N, C] equivalent to copying T times along the 2nd dimension, and B times along the 1st dimension
        D_S = D_S.permute(0, 2, 1, 3)  # [B, N, T, C]
        # GCN part
        X_G = torch.Tensor(B, N, 0, C).to(self.device)
        adj_laplacian = self.laplacian
        # Normalizing the adjacency matrix is necessary because the adjacency matrix is not composed of 0s and 1s, but rather the distances between two sensors, hence normalization is required
        for t in range(query.shape[2]):
            o = self.gcn(query[:, :, t, :], adj_laplacian)  # [B, N, C]
            o = o.unsqueeze(2)  # shape [N, 1, C] [B, N, 1, C]
            #             print(o.shape)
            X_G = torch.cat((X_G, o), dim=2)
        # Final X_G [B, N, T, C]
        # Apply dropout
        out = self.dropout(X_G)
        return out

class GCN_layer(nn.Module):
    def __init__(self, input_dim, output_dim, adj, device):
        super(GCN_layer, self).__init__()
        self.register_buffer('laplacian', calculate_laplacian_with_self_loop(torch.FloatTensor(adj)))
        self._num_nodes = adj.shape[0]
        self._input_dim = input_dim
        self._output_dim = output_dim
        self.device = device
        self.weights = nn.Parameter(torch.FloatTensor(self._input_dim, self._output_dim))
        # LH Add learnable parameter matrix α
        self.alpha_matrix = nn.Parameter(torch.eye(adj.shape[0]), requires_grad=True)
        # LH
        self.reset_parameters()

    def reset_parameters(self):
        # Note whether the parameters of alpha_matrix need to be initialized here

        nn.init.normal_(self.alpha_matrix, 0.5, 0.5)

        nn.init.xavier_uniform_(self.weights, gain=nn.init.calculate_gain('tanh'))

    def forward(self, inputs):
        # (batch_size, seq_len, num_nodes)
        batch_size = inputs.shape[0]
        # (num_nodes, batch_size, embedding)
        inputs = inputs.permute(1, 0, 2)
        # inputs = inputs.transpose(1, 0, 2)
        # (num_nodes, batch_size * seq_len)
        inputs = inputs.reshape((self._num_nodes, batch_size * self._input_dim))

        # LH
        # Since using CPU for training, no need to put alpha_matrix into GPU
        new_alpha_matrix = self.alpha_matrix * torch.eye(self._num_nodes).to(self.device)
        new_laplacian = self.laplacian + new_alpha_matrix
        # LH
        # AX (num_nodes, batch_size * seq_len)
        ax = new_laplacian @ inputs
        # (num_nodes, batch_size, seq_len)
        ax = ax.reshape((self._num_nodes, batch_size, self._input_dim))
        # (num_nodes * batch_size, seq_len)
        ax = ax.reshape((self._num_nodes * batch_size, self._input_dim))
        # act(AXW) (num_nodes * batch_size, output_dim)
        outputs = torch.tanh(ax @ self.weights)
        # (num_nodes, batch_size, output_dim)
        outputs = outputs.reshape((self._num_nodes, batch_size, self._output_dim))
        # (batch_size, num_nodes, output_dim)
        outputs = outputs.transpose(0, 1)
        return outputs

class GCN(nn.Module):
    def __init__(self, adj, input_dim, output_dim, device):
        super(GCN, self).__init__()
        # self.register_buffer('laplacian', calculate_laplacian_with_self_loop(torch.FloatTensor(adj)))
        self._num_nodes = adj.shape[0]
        self._input_dim = input_dim
        self._output_dim = output_dim
        self.device = device
        # Two-layer GCN
        self.gcn1 = GCN_layer(self._input_dim, self._input_dim * 2, adj, device)
        self.gcn2 = GCN_layer(self._input_dim * 2, self._output_dim, adj, device)

    def forward(self, inputs, adj_laplacian):
        output_first = self.gcn1(inputs)

        # Add activation function
        output_second = self.gcn2(output_first)

        return output_second

def get_sinusoid_encoding_table(n_position, d_model):
    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_model)
    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_model)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    return torch.FloatTensor(sinusoid_table)


class TTransformer(nn.Module):
    def __init__(self, embed_size, heads, time_num, dropout, device, forward_expansion):
        super(TTransformer, self).__init__()
        self.device = device
        # Temporal embedding One hot
        self.time_num = time_num
        #         self.one_hot = One_hot_encoder(embed_size, time_num)          # temporal embedding using one-hot method or
        #         self.temporal_embedding = nn.Embedding(time_num, embed_size)  # temporal embedding using nn.Embedding

        self.attention = TMultiHeadAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, t):
        B, N, T, C = query.shape

        # LH
        D_T = get_sinusoid_encoding_table(T, C).to(self.device)
        # LH
        D_T = D_T.expand(B, N, T, C)

        # Add temporal embedding to query. Original paper uses concatenated
        query = query + D_T

        attention = self.attention(query, query, query)

        # Add skip connection, run through normalization and finally dropout
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out


class TMultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(TMultiHeadAttention, self).__init__()

        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
                self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        # Use Linear as projection matrix

        self.W_V = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.W_K = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.W_Q = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, input_Q, input_K, input_V):
        '''
        input_Q: [batch_size, N, T, C]
        input_K: [batch_size, N, T, C]
        input_V: [batch_size, N, T, C]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        B, N, T, C = input_Q.shape
        # [B, N, T, C] --> [B, N, T, h * d_k] --> [B, N, T, h, d_k] --> [B, h, N, T, d_k]
        Q = self.W_Q(input_Q).view(B, N, T, self.heads, self.head_dim).permute(0, 3, 1, 2, 4)  # Q: [B, h, N, T, d_k]
        K = self.W_K(input_K).view(B, N, T, self.heads, self.head_dim).permute(0, 3, 1, 2, 4)  # K: [B, h, N, T, d_k]
        V = self.W_V(input_V).view(B, N, T, self.heads, self.head_dim).permute(0, 3, 1, 2, 4)  # V: [B, h, N, T, d_k]

        # attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1) # attn_mask : [batch_size, n_heads, seq_len, seq_len]

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context = ScaledDotProductAttention()(Q, K, V)  # [B, h, N, T, d_k]
        context = context.permute(0, 2, 3, 1, 4)  # [B, N, T, h, d_k]
        context = context.reshape(B, N, T, self.heads * self.head_dim)  # [B, N, T, C]
        # context = context.transpose(1, 2).reshape(batch_size, -1, n_heads * d_v) # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc_out(context)  # [batch_size, len_q, d_model]
        return output

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V):
        '''
        Q: [batch_size, n_heads, T(Spatial) or N(Temporal), N(Spatial) or T(Temporal), d_k]
        K: [batch_size, n_heads, T(Spatial) or N(Temporal), N(Spatial) or T(Temporal), d_k]
        V: [batch_size, n_heads, T(Spatial) or N(Temporal), N(Spatial) or T(Temporal), d_k]
        attn_mask: [batch_size, n_heads, seq_len, seq_len] may not exist
        '''
        B, n_heads, len1, len2, d_k = Q.shape
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        # scores : [batch_size, n_heads, T(Spatial) or N(Temporal), N(Spatial) or T(Temporal), N(Spatial) or T(Temporal)]
        # scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is True.

        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn,
                               V)  # [batch_size, n_heads, T(Spatial) or N(Temporal), N(Spatial) or T(Temporal), d_k]]
        return context