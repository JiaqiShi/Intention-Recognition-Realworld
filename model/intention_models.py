import torch
import torch.nn as nn
import torch.nn.functional as F
from model.intentgcn import *


class Multilayer_perceptron(nn.Module):
    def __init__(self, input_shape, out_channels) -> None:
        super().__init__()

        if type(input_shape) == list:
            input_shape = input_shape[0]

        input_channels = input_shape[0]

        self.layers = nn.Sequential(
            nn.Linear(input_channels, 512),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(512, 128),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(128, 64),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(64, out_channels)
        )

    def get_model_name(self):
        return self.__class__.__name__

    def get_features(self, x):
        return self.layers[0:-1](x)

    def forward(self, x):

        return self.layers(x)


class GCN_and_MLP(nn.Module):
    def __init__(self, input_shape, num_class, graph_args, edge_importance_weighting=True, load_dict_paths=None, is_detach=True, device='cuda:0', **kwargs):
        super().__init__()

        assert len(input_shape) == 2

        self.load_dict_paths = load_dict_paths
        self.is_detach = is_detach

        self.gcn = IntentGCN(
            input_shape[0], num_class, graph_args, edge_importance_weighting, **kwargs)

        self.mlp = Multilayer_perceptron(input_shape[1], num_class)

        self.output_layer = nn.Linear(
            self.gcn.output_channels[-1]+64, num_class)

        if load_dict_paths is not None:
            print(f'[Info] load models from {load_dict_paths}')
            assert len(load_dict_paths) == 2
            self.gcn.to(device).load_state_dict(torch.load(load_dict_paths[0]))
            self.mlp.to(device).load_state_dict(torch.load(load_dict_paths[1]))

    def get_model_name(self):
        return self.__class__.__name__

    def forward(self, x0, x1):

        if self.load_dict_paths is not None and self.is_detach:
            inp0 = self.gcn.get_features(x0).detach()
            inp1 = self.mlp.get_features(x1).detach()
        else:
            inp0 = self.gcn.get_features(x0)
            inp1 = self.mlp.get_features(x1)

        inp = torch.cat([inp0, inp1], dim=1)

        return self.output_layer(inp)


class GCN_MLP_MLP(nn.Module):
    def __init__(self, input_shape, num_class, graph_args, edge_importance_weighting=True, load_dict_paths=None, is_detach=True, device='cuda:0', **kwargs):
        super().__init__()

        assert len(input_shape) == 3

        self.load_dict_paths = load_dict_paths
        self.is_detach = is_detach

        self.gcn = IntentGCN(
            input_shape[0], num_class, graph_args, edge_importance_weighting, **kwargs)

        self.mlp1 = Multilayer_perceptron(input_shape[1], num_class)
        self.mlp2 = Multilayer_perceptron(input_shape[2], num_class)

        self.output_layer = nn.Linear(
            self.gcn.output_channels[-1]+64+64, num_class)

        if load_dict_paths is not None:
            print(f'[Info] load models from {load_dict_paths}')
            assert len(load_dict_paths) == 3
            self.gcn.to(device).load_state_dict(torch.load(load_dict_paths[0]))
            self.mlp1.to(device).load_state_dict(
                torch.load(load_dict_paths[1]))
            self.mlp2.to(device).load_state_dict(
                torch.load(load_dict_paths[2]))

    def get_model_name(self):
        return self.__class__.__name__

    def forward(self, x0, x1, x2):

        if self.load_dict_paths is not None and self.is_detach:
            inp0 = self.gcn.get_features(x0).detach()
            inp1 = self.mlp1.get_features(x1).detach()
            inp2 = self.mlp2.get_features(x2).detach()
        else:
            inp0 = self.gcn.get_features(x0)
            inp1 = self.mlp1.get_features(x1)
            inp2 = self.mlp2.get_features(x2)

        inp = torch.cat([inp0, inp1, inp2], dim=1)

        return self.output_layer(inp)
