import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class AGCN(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k):
        super(AGCN, self).__init__()
        self.cheb_k = cheb_k
        self.weights = nn.Parameter(torch.FloatTensor(2 * cheb_k * dim_in, dim_out))  # 2 is the length of support
        self.bias = nn.Parameter(torch.FloatTensor(dim_out))
        nn.init.xavier_normal_(self.weights)
        nn.init.constant_(self.bias, val=0)

    def forward(self, x, supports):
        x_g = []
        support_set = []

        for support in supports[0:2]:
            support_ks = [torch.eye(support.shape[0]).to(support.device), support]
            for k in range(2, self.cheb_k):
                support_ks.append(torch.matmul(2 * support, support_ks[-1]) - support_ks[-2])
            support_set.extend(support_ks)
        for support in support_set:
            x_g.append(torch.einsum("nm,bmc->bnc", support, x))
        x_g = torch.cat(x_g, dim=-1)  # B, N, 2 * cheb_k * dim_in
        x_gconv = torch.einsum('bni,io->bno', x_g, self.weights) + self.bias  # b, N, dim_out

        return x_gconv


class AGCRNCell(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k):
        super(AGCRNCell, self).__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.gate = AGCN(dim_in + self.hidden_dim, 2 * dim_out, cheb_k)
        self.update = AGCN(dim_in + self.hidden_dim, dim_out, cheb_k)

    def forward(self, x, state, supports):
        # x: B, num_nodes, input_dim
        # state: B, num_nodes, hidden_dim
        state = state.to(x.device)
        input_and_state = torch.cat((x, state), dim=-1)
        z_r = torch.sigmoid(self.gate(input_and_state, supports))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z * state), dim=-1)
        hc = torch.tanh(self.update(candidate, supports))
        h = r * state + (1 - r) * hc
        return h

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)


class ADCRNN_Decoder(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, num_layers):
        super(ADCRNN_Decoder, self).__init__()
        assert num_layers >= 1, 'At least one RNN layer in the Decoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(AGCRNCell(node_num, dim_in, dim_out, cheb_k))
        for _ in range(1, num_layers):
            self.dcrnn_cells.append(AGCRNCell(node_num, dim_out, dim_out, cheb_k))

    def forward(self, xt, init_state, supports):
        # xt: (B, N, D)
        # init_state: (num_layers, B, N, hidden_dim)
        assert xt.shape[1] == self.node_num and xt.shape[2] == self.input_dim
        current_inputs = xt
        output_hidden = []
        for i in range(self.num_layers):
            state = self.dcrnn_cells[i](current_inputs, init_state[i], supports)
            output_hidden.append(state)
            current_inputs = state
        return current_inputs, output_hidden


class TimeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(TimeBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.res1 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.res2 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.drop = nn.Dropout(0.2)

    def forward(self, X):
        # X: (batch_size, num_nodes, num_timesteps, num_features=in_channels)
        X = X.permute(0, 3, 1, 2)
        # X: (batch_size, num_features=in_channels, num_nodes, num_timesteps)
        temp = self.conv1(X) + torch.sigmoid(self.res1(X))
        out = F.relu(temp + self.res2(X))
        out = out.permute(0, 2, 3, 1)
        # out: (batch_size, num_nodes, num_timesteps, num_features=out_channels)
        return out


class ChebConv(nn.Module):
    def __init__(self, dim_out, cheb_k, time_steps_in, feature_size):
        super(ChebConv, self).__init__()
        self.time_steps_in = time_steps_in
        self.dim_in = feature_size
        self.cheb_k = cheb_k
        self.weights = nn.Parameter(torch.FloatTensor(2 * cheb_k * self.dim_in, dim_out))
        # 2 is the length of support
        self.bias = nn.Parameter(torch.FloatTensor(dim_out))
        nn.init.xavier_normal_(self.weights)
        nn.init.constant_(self.bias, val=0)

    def forward(self, x, supports):
        # x: B, N, dim_in
        # supports: 2, N, N
        x_g = []
        support_set = []
        for support in supports:
            support_ks = [torch.eye(support.shape[0]).to(support.device), support]
            # support_ks: [I, L, L^2, ..., L^k-1]
            for k in range(2, self.cheb_k):
                support_ks.append(torch.matmul(2 * support, support_ks[-1]) - support_ks[-2])  # shape: N, N
            support_set.extend(support_ks)
            # support_set: [I, L, L^2, ..., L^k-1, I, L, L^2, ..., L^k-1]
        for support in support_set:
            x_g.append(torch.einsum("ij,jklm->kilm", support, x.permute(1,0,2,3)))  # b n t d
        x_g = torch.cat(x_g, dim=-1)  # B, N, t, 2 * cheb_k * dim_in
        x_gconv = torch.einsum('kilm,mo->kilo', x_g, self.weights) + self.bias  # B, N, dim_out
        # B, N, D = x_gconv.shape
        # x_gconv = x_gconv.reshape(B, N, self.time_steps_in, -1)  # B, N, T, dim_out
        return x_gconv


class ST_block(nn.Module):
    def __init__(self, in_channels, spatial_channels1, spatial_channels2, out_channels, num_nodes, time_steps_in,
                 kt=3):
        '''

        :param in_channels:
        :param spatial_channels1:
        :param spatial_channels2:
        :param out_channels:
        :param num_nodes:
        :param time_steps_in:
        :param kt:
        '''
        super(ST_block, self).__init__()
        self.temporal_conv1 = TimeBlock(in_channels, spatial_channels1)
        self.spatial_conv = ChebConv(spatial_channels2, cheb_k=3, time_steps_in=time_steps_in-kt+1, feature_size=spatial_channels1)
        self.temporal_conv2 = TimeBlock(spatial_channels2, out_channels)
        self.batch_norm = nn.BatchNorm2d(num_nodes)

    def forward(self, X, supports):
        # X: (batch_size, num_nodes, num_timesteps, num_features=in_channels)
        out1 = self.temporal_conv1(X)

        # out1: (batch_size, num_nodes, num_timesteps-2, num_features=out_channels)
        B, N, T, C = out1.size()
        # out1 = out1.reshape(B, N, T*C)
        # out1: (num_nodes, batch_size, (num_timesteps-2)*out_channels)

        out2 = self.spatial_conv(out1, supports)

        # out2: (batch_size, num_nodes, num_timesteps-2, num_features=spatial_channels)
        out2 = self.temporal_conv2(out2)

        # out2: (batch_size, num_nodes, num_timesteps-4, num_features=out_channels)
        out = F.relu(out2)

        return self.batch_norm(out)


class TCN_MEM(nn.Module):
    def __init__(self, batch_size,num_nodes, input_dim, output_dim, horizon, rnn_units, steps, num_layers=1, cheb_k=3,
                 mem_num=10, mem_dim=32, use_curriculum_learning=True):
        super(TCN_MEM, self).__init__()
        self.batch_size = batch_size
        self.steps = steps
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.rnn_units = rnn_units
        self.output_dim = output_dim
        self.horizon = horizon
        self.num_layers = num_layers
        self.cheb_k = cheb_k
        self.cl_decay_steps = 2000
        self.use_curriculum_learning = use_curriculum_learning
        # self.emd = 8
        # self.node_emb = nn.Parameter(torch.randn(self.num_nodes, self.emd), requires_grad=True)

        # memory
        self.mem_num = mem_num
        self.mem_dim = mem_dim
        self.memory = self.construct_memory()

        self.weight_key = nn.Parameter(torch.zeros(size=(self.mem_dim * (self.steps - 4), 1)))
        nn.init.xavier_uniform_(self.weight_key.data, gain=1.414)
        self.weight_query = nn.Parameter(torch.zeros(size=(self.mem_dim * (self.steps - 4), 1)))
        nn.init.xavier_uniform_(self.weight_query.data, gain=1.414)

        # block1
        self.block1 = ST_block(in_channels=input_dim, spatial_channels1=64, spatial_channels2=16, out_channels=32,
                                 num_nodes=num_nodes, time_steps_in=horizon, kt=3)
        # block2
        self.block2 = ST_block(in_channels=32*2, spatial_channels1=16, spatial_channels2=64, out_channels=8,
                                    num_nodes=num_nodes, time_steps_in=horizon-4, kt=3)
        self.last_temporal = TimeBlock(in_channels=64, out_channels=64)
        # output
        self.proj = nn.Linear(8*4, 1, bias=True)
        self.dropout = nn.Dropout(0.2)

        self.decoder = ADCRNN_Decoder(self.num_nodes, 1, 8*4, self.cheb_k,
                                      self.num_layers)

    def compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_steps / (self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))

    def self_graph_attention(self, input):

        bat, N, fea = input.size()
        key = torch.matmul(input, self.weight_key)
        query = torch.matmul(input, self.weight_query)
        data = key.repeat(1, 1, N).view(bat, N * N, 1) + query.repeat(1, N, 1)
        data = data.squeeze(2)
        data = data.view(bat, N, -1)
        data = F.relu(data)
        attention = F.softmax(data, dim=2)
        attention = self.dropout(attention)
        return attention

    def construct_memory(self):
        memory_dict = nn.ParameterDict()
        memory_dict['Memory'] = nn.Parameter(torch.randn(self.mem_num, self.mem_dim * (self.steps - 4)), requires_grad=True)  # (M, d)
        memory_dict['Wq'] = nn.Parameter(torch.randn(self.rnn_units, self.mem_dim * (self.steps - 4)),
                                         requires_grad=True)  # project to query
        memory_dict['We1'] = nn.Parameter(torch.randn(self.num_nodes, self.mem_num),
                                          requires_grad=True)  # project memory to embedding
        memory_dict['We2'] = nn.Parameter(torch.randn(self.num_nodes, self.mem_num),
                                          requires_grad=True)  # project memory to embedding

        # Memory: (fai, d)
        # Wq: (hidden, d)
        # We1: (N, fai)
        # We2: (N, fai)
        for param in memory_dict.values():
            nn.init.xavier_normal_(param)
        return memory_dict

    def query_memory(self, h_t: torch.Tensor):
        # h_t: (B, N, hidden)
        query = torch.matmul(h_t, self.memory['Wq'])  # （B,N,hidden）* (hidden, d) = (B, N, d)

        att_score = torch.softmax(torch.matmul(query, self.memory['Memory'].t()), dim=-1)


        # self.memory['Memory'].t(): (d, fai)
        # alpha: (B,N,d)*(d, fai) = (B, N, fai)

        value = torch.matmul(att_score, self.memory['Memory'])
        attention = self.self_graph_attention(value)
        # attention.shape:[32,140,140]
        attention = torch.mean(attention, dim=0)

        degree = torch.sum(attention, dim=1)
        attention = 0.5 * (attention + attention.T)
        degree_l = torch.diag(degree)
        diagonal_degree_hat = torch.diag(1 / (torch.sqrt(degree) + 1e-7))
        laplacian = torch.matmul(diagonal_degree_hat,
                                 torch.matmul(degree_l - attention, diagonal_degree_hat))
        # self.memory['Memory']: (fai, d)
        # (B, N, fai)*(fai, d) = (B, N, d)

        return attention, value, query

    def forward(self, x, labels=None, batches_seen=None):
        # x: (B, T, N, D)
        node_embeddings1 = torch.matmul(self.memory['We1'], self.memory['Memory'])  # (N, fai)*(fai, d) = (N, d)
        node_embeddings2 = torch.matmul(self.memory['We2'], self.memory['Memory'])  # (N, fai)*(fai, d) = (N, d)
        g1 = F.softmax(F.relu(torch.mm(node_embeddings1, node_embeddings2.T)), dim=-1)  # (N, N)
        g2 = F.softmax(F.relu(torch.mm(node_embeddings2, node_embeddings1.T)), dim=-1)  # (N, N)
        supports = [g1, g2]
        # supports shape: (2, N, N)
        # block1
        out1 = self.block1(x, supports)
        # out1: (batch_size, num_nodes, num_timesteps-4, num_features=out_channels)
        h_t = out1.reshape(out1.shape[0], out1.shape[1], -1)  # (B, N, T*C)
        # query memory
        support1, h_att, query = self.query_memory(h_t)
        g1_ = F.softmax(F.relu(torch.mm(node_embeddings1, node_embeddings2.T) + support1), dim=-1)  # (N, N)
        g2_ = F.softmax(F.relu(torch.mm(node_embeddings2, node_embeddings1.T) + support1), dim=-1)  # (N, N)
        supports1 = [g1_, g2_]
        # h_att: (B, N, d*(steps-4)))  query: (B, N, d)  pos: (B, N, d)  neg: (B, N, d)
        h_t = h_t.reshape(h_t.shape[0], h_t.shape[1], self.steps-4, -1)
        h_att = h_att.reshape(h_att.shape[0], h_att.shape[1], self.steps-4, -1)
        h_t = torch.cat([h_t, h_att], dim=-1)
        # h_t: (B, N, T, C*2)
        # block2
        out2 = self.block2(h_t, supports1)
        # out2: (batch_size, num_nodes, num_timesteps-8, num_features=out_channels)
        out2 = out2.reshape(out2.shape[0], out2.shape[1], -1)
        # out2: (B, N, (T-8)*C)
        ht_list = [out2] * self.num_layers
        # ht_list: (num_layers, B, N, (T-8)*C)
        go = torch.zeros((x.shape[0], self.num_nodes, 1), device=x.device)
        out = []
        for t in range(self.horizon):
            h_de, ht_list = self.decoder(go, ht_list, supports1)
            # torch.cat([go, y_cov[:, t, ...]], dim=-1): (B, N, output_dim)
            # ht_list: (num_layers, B, N, hidden + d)
            # h_de: (B, N, hidden + d)
            # ht_list: (num_layers, B, N, hidden + d)
            go = self.proj(h_de)
            # go: (B, N, output_dim)
            out.append(go)
            # curriculum learning
            if self.training:
                c = np.random.uniform(0, 1)
                if c < self.compute_sampling_threshold(batches_seen):
                    go = labels[:, t, ...]
        output = torch.stack(out, dim=1)
        # output: (B, T, N, output_dim)

        return output, query




