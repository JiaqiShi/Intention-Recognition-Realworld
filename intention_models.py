#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
#%%
class CNN_and_MLP(nn.Module):
    def __init__(self, input_shape, out_channels):
        super().__init__()

        assert len(input_shape) == 2

        max_T, V, C = input_shape[0]
        point_embedding_dim = V*C

        res_input_channels = input_shape[1][0]

        self.layers_cnn = nn.Sequential(
            nn.Conv2d(1,256,kernel_size=(5,point_embedding_dim)),
            nn.ReLU(inplace=False),
            nn.Conv2d(256,256,kernel_size=(5,1)),
            nn.Dropout(p=0.1,inplace=False),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=(5,1)),
            nn.Conv2d(256,256,kernel_size=(5,1)),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.1,inplace=False),
            nn.MaxPool2d(kernel_size=(5,1)),
            nn.Conv2d(256,512,kernel_size=(5,1)),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.1,inplace=False),
            nn.MaxPool2d(kernel_size=(5,1)),
            nn.Conv2d(512,512,kernel_size=(3,1)),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.1,inplace=False)
        )

        self.cnn_dense = nn.Sequential(
            nn.Linear(512,128),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.2,inplace=False)
        )

        self.layers_mlp = nn.Sequential(
            nn.Linear(res_input_channels,512),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.2,inplace=False),
            nn.Linear(512,256),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.2,inplace=False),
            nn.Linear(256,128),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.2,inplace=False)
        )

        self.output_layer = nn.Linear(256, out_channels)

    def get_model_name(self):
        return self.__class__.__name__

    def forward(self, x0, x1):

        # x0, x1 = xs

        inp0, lengths = x0

        N, T, V, C = inp0.size()
        inp0 = inp0.view(N, 1, T, V*C)
        
        inp0 = self.layers_cnn(inp0)
        inp0 = F.max_pool2d(inp0, inp0.size()[2:]).view(N,-1)
        inp0 = self.cnn_dense(inp0)

        inp1 = self.layers_mlp(x1)

        inp = torch.cat([inp0, inp1], dim=1)

        return self.output_layer(inp)


class CNN_MLP_MLP(nn.Module):
    def __init__(self, input_shape, out_channels):
        super().__init__()

        assert len(input_shape) == 3

        max_T, V, C = input_shape[0]
        point_embedding_dim = V*C

        res_input_channels0 = input_shape[1][0]

        res_input_channels1 = input_shape[2][0]

        self.layers_cnn = nn.Sequential(
            nn.Conv2d(1,256,kernel_size=(5,point_embedding_dim)),
            nn.ReLU(inplace=False),
            nn.Conv2d(256,256,kernel_size=(5,1)),
            nn.Dropout(p=0.1,inplace=False),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=(5,1)),
            nn.Conv2d(256,256,kernel_size=(5,1)),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.1,inplace=False),
            nn.MaxPool2d(kernel_size=(5,1)),
            nn.Conv2d(256,512,kernel_size=(5,1)),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.1,inplace=False),
            nn.MaxPool2d(kernel_size=(5,1)),
            nn.Conv2d(512,512,kernel_size=(3,1)),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.1,inplace=False)
        )

        self.cnn_dense = nn.Sequential(
            nn.Linear(512,128),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.2,inplace=False)
        )

        self.layers_mlp0 = nn.Sequential(
            nn.Linear(res_input_channels0,256),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.2,inplace=False),
            nn.Linear(256,128),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.2,inplace=False)
        )

        self.layers_mlp1 = nn.Sequential(
            nn.Linear(res_input_channels1,256),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.2,inplace=False),
            nn.Linear(256,128),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.2,inplace=False)
        )

        self.output_layer = nn.Linear(128*3, out_channels)

    def get_model_name(self):
        return self.__class__.__name__

    def forward(self, x0, x1, x2):

        inp0, lengths = x0

        N, T, V, C = inp0.size()
        inp0 = inp0.view(N, 1, T, V*C)
        
        inp0 = self.layers_cnn(inp0)
        inp0 = F.max_pool2d(inp0, inp0.size()[2:]).view(N,-1)
        inp0 = self.cnn_dense(inp0)

        inp1 = self.layers_mlp0(x1)

        inp2 = self.layers_mlp1(x2)

        inp = torch.cat([inp0, inp1, inp2], dim=1)

        return self.output_layer(inp)

class CNN_MLP_MLP_double(nn.Module):
    def __init__(self, input_shape, out_channels):
        super().__init__()

        assert len(input_shape) == 3

        max_T, V, C = input_shape[0]
        point_embedding_dim = V*C

        res_input_channels0 = input_shape[1][0]

        res_input_channels1 = input_shape[2][0]

        self.layers_cnn = nn.Sequential(
            nn.Conv2d(1,256,kernel_size=(5,point_embedding_dim)),
            nn.ReLU(inplace=False),
            nn.Conv2d(256,256,kernel_size=(5,1)),
            nn.Dropout(p=0.1,inplace=False),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=(5,1)),
            nn.Conv2d(256,256,kernel_size=(5,1)),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.1,inplace=False),
            nn.MaxPool2d(kernel_size=(5,1)),
            nn.Conv2d(256,512,kernel_size=(5,1)),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.1,inplace=False),
            nn.MaxPool2d(kernel_size=(5,1)),
            nn.Conv2d(512,512,kernel_size=(3,1)),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.1,inplace=False)
            # nn.MaxPool2d(kernel_size=(3,1))
        )

        self.cnn_dense = nn.Sequential(
            nn.Linear(512,128),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.2,inplace=False)
        )

        self.layers_mlp0 = nn.Sequential(
            nn.Linear(res_input_channels0,512),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.2,inplace=False),
            nn.Linear(512,128),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.2,inplace=False)
        )

        self.layers_mlp1 = nn.Sequential(
            nn.Linear(res_input_channels1,512),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.2,inplace=False),
            nn.Linear(512,128),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.2,inplace=False)
        )

        self.output_layer = nn.Linear(128*3, out_channels)

    def get_model_name(self):
        return self.__class__.__name__

    def forward(self, x0, x1, x2):

        inp0, lengths = x0

        N, T, V, C = inp0.size()
        inp0 = inp0.view(N, 1, T, V*C)
        
        inp0 = self.layers_cnn(inp0)
        inp0 = F.max_pool2d(inp0, inp0.size()[2:]).view(N,-1)
        inp0 = self.cnn_dense(inp0)

        inp1 = self.layers_mlp0(x1)

        inp2 = self.layers_mlp1(x2)

        inp = torch.cat([inp0, inp1, inp2], dim=1)

        return self.output_layer(inp)