import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from model.graph import Graph


class IntentGCN(nn.Module):
    def __init__(self,
                 input_shape,
                 num_class,
                 graph_args=None,
                 edge_importance_weighting=True,
                 **kwargs):
        super().__init__()

        if type(input_shape) == list:
            input_shape = input_shape[0]
        max_T, num_joints, in_channels = input_shape
        self.in_channels = in_channels

        if graph_args is not None:
            self.graph = Graph(**graph_args)
        else:
            self.graph = Graph()
        A = torch.tensor(self.graph.A,
                         dtype=torch.float32,
                         requires_grad=False)
        self.register_buffer('A', A)

        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        self.kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.bn = nn.BatchNorm1d(in_channels * A.size(1))

        self._get_gcn_layers(**kwargs)

        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList(
                [nn.Parameter(torch.ones(self.A.size())) for i in self.blocks])
        else:
            self.edge_importance = [1] * len(self.blocks)

        self.fcn = nn.Linear(self.output_channels[-1], num_class)

    def _get_gcn_layers(self,
                        layer_num,
                        channels,
                        stride_2_layer_index,
                        dropout=0,
                        **kw):
        def str2intlist(inp_str):
            return [int(s) for s in inp_str.split(',')]

        def get_output_channels(layer_num, channels_list):
            step_num = len(channels_list)
            layer_steps = [
                int(layer_num / step_num) +
                1 if i < layer_num % step_num else int(layer_num / step_num)
                for i in range(step_num)
            ]
            return [
                channels_list[step] for step, layer in enumerate(layer_steps)
                for _ in range(layer)
            ]

        channels_list = str2intlist(channels)
        if layer_num < len(channels_list):
            raise ValueError(
                f'Too many channels given. Expected length larger than {len(channels)}, but got {layer_num}.'
            )
        stride_2_layers = str2intlist(stride_2_layer_index)
        output_channels = get_output_channels(layer_num, channels_list)
        self.output_channels = output_channels
        print('#channels:', output_channels)
        gcn_layer_list = []
        for i in range(layer_num):
            if i in stride_2_layers:
                stride = 2
            else:
                stride = 1
            if i == 0:
                gcn_layer_list.append(
                    GraphConvBlock(self.in_channels, output_channels[i],
                                   self.kernel_size, stride, **kw))
            else:
                gcn_layer_list.append(
                    GraphConvBlock(output_channels[i - 1], output_channels[i],
                                   self.kernel_size, stride, dropout, **kw))
        self.blocks = nn.ModuleList(gcn_layer_list)

    def get_model_name(self):
        return self.__class__.__name__

    def forward(self, x):

        if type(x) == list:
            x, lengths = x
        x = x.float()

        # N, T, V, C -> N, C, T, V
        x = x.permute(0, 3, 1, 2).contiguous()

        N, C, T, V = x.size()

        x = x.permute(0, 3, 1, 2).contiguous().view(N, V * C, T)
        x = self.bn(x)
        x = x.view(N, V, C, T).permute(0, 2, 3, 1).contiguous()

        for block, importance in zip(self.blocks, self.edge_importance):
            x = block(x, self.A * importance)

        hiddens = F.avg_pool2d(x, x.size()[2:]).view(x.size(0), -1)

        x = self.fcn(hiddens)

        return x

    def get_features(self, x):

        if type(x) == list:
            x, lengths = x
        x = x.float()

        # N, T, V, C -> N, C, T, V
        x = x.permute(0, 3, 1, 2).contiguous()

        N, C, T, V = x.size()

        x = x.permute(0, 3, 1, 2).contiguous().view(N, V * C, T)
        x = self.bn(x)
        x = x.view(N, V, C, T).permute(0, 2, 3, 1).contiguous()

        for block, importance in zip(self.blocks, self.edge_importance):
            x = block(x, self.A * importance)

        hiddens = F.avg_pool2d(x, x.size()[2:]).view(x.size(0), -1)

        return hiddens


class GraphConvBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0,
                 residual=True,
                 adap_graph=True,
                 weighted_sum=True,
                 n_head=4,
                 d_kc=0.25,
                 d_vc=0.25):
        super().__init__()

        self.scn = SpatialGraphConv(in_channels,
                                    out_channels,
                                    kernel_size[1],
                                    adap_graph=adap_graph,
                                    weighted_sum=weighted_sum,
                                    n_head=n_head,
                                    d_kc=d_kc,
                                    d_vc=d_vc)

        self.tcn = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, (kernel_size[0], 1),
                      (stride, 1), ((kernel_size[0] - 1) // 2, 0)),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels,
                          out_channels,
                          kernel_size=1,
                          stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A, length=None):

        res = self.residual(x)
        x = self.scn(x, A)
        x = self.tcn(x) + res

        return self.relu(x)


class SpatialGraphConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True,
                 adap_graph=True,
                 weighted_sum=True,
                 n_head=4,
                 d_kc=0.25,
                 d_vc=0.25):
        super().__init__()

        self.kernel_size = kernel_size
        self.adap_graph = adap_graph
        self.weighted_sum = weighted_sum

        self.bn = nn.BatchNorm2d(in_channels)

        if int(d_kc * in_channels) == 0:
            d_kc = 1
            d_vc = 1

        self.partconv = nn.Conv2d(in_channels,
                                  out_channels * kernel_size,
                                  kernel_size=(t_kernel_size, 1),
                                  padding=(t_padding, 0),
                                  stride=(t_stride, 1),
                                  dilation=(t_dilation, 1),
                                  bias=bias)

        if adap_graph is True:
            self.adapconv = AdapGraphConv(n_head,
                                          d_in=in_channels,
                                          d_out=out_channels,
                                          d_k=int(d_kc * out_channels),
                                          d_v=int(out_channels * d_vc),
                                          residual=True,
                                          res_fc=False)

        if weighted_sum is True:
            print('[Info] gate activated.')
            w = nn.Parameter(torch.tensor(1.0, dtype=torch.float32),
                             requires_grad=True)
            self.register_parameter('w', w)

        self.out = nn.Sequential(nn.BatchNorm2d(out_channels),
                                 nn.ReLU(inplace=True))

    def forward(self, x, A):

        inp = self.bn(x)

        f_c = self.partconv(inp)

        # spatial graph convolution
        N, KC, T, V = f_c.size()
        f_c = f_c.view(N, self.kernel_size, KC // self.kernel_size, T, V)
        f_c = torch.einsum('nkctv,kvw->nctw', (f_c, A))

        if self.adap_graph:
            N, C, T, V = inp.size()
            f_a = inp.permute(0, 2, 3, 1).contiguous().view(N * T, V, C)
            f_a, _ = self.adapconv(f_a, f_a, f_a)
            f_a = f_a.view(N, T, V, -1).permute(0, 3, 1, 2)  # N, C, T, V

            if self.weighted_sum:
                f = (f_a * self.w + f_c) / 2
            else:
                f = (f_a + f_c) / 2
        else:
            f = f_c

        f = self.out(f)

        return f


class AdapGraphConv(nn.Module):
    def __init__(self,
                 n_head,
                 d_in,
                 d_out,
                 d_k,
                 d_v,
                 residual=True,
                 res_fc=False,
                 dropout=0.1,
                 a_dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_in = d_in
        self.d_out = d_out
        self.d_k = d_k
        self.d_v = d_v

        self.residual = residual
        self.res_fc = res_fc

        self.w_q = nn.Linear(d_in, n_head * d_k, bias=False)
        self.w_k = nn.Linear(d_in, n_head * d_k, bias=False)
        self.w_v = nn.Linear(d_in, n_head * d_v, bias=False)

        self.fc = nn.Linear(n_head * d_v, d_out, bias=False)

        if residual:
            self.res = nn.Linear(
                d_in, d_out) if res_fc or (d_in != d_out) else lambda x: x

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.a_drop = nn.Dropout(a_dropout)

    def forward(self, q, k, v):

        assert self.d_in == v.size(2)

        NT, V, C = v.size()

        if self.residual:
            res = self.res(v)

        q = self.layer_norm(q)

        q = self.w_q(q).view(NT, V, self.n_head, self.d_k)
        k = self.w_k(k).view(NT, V, self.n_head, self.d_k)
        v = self.w_v(v).view(NT, V, self.n_head, self.d_v)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        A_adap = torch.matmul(q, k.transpose(2, 3)) / (self.d_k**0.5)
        A_adap = self.a_drop(F.softmax(A_adap, dim=3))

        x = torch.matmul(A_adap, v)  # NT, H, V, D_v

        x = x.transpose(1, 2).contiguous().view(NT, V, -1)
        x = self.dropout(self.fc(x))

        if self.residual:
            x += res

        return x, A_adap
