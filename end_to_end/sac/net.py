import numpy as np
import torch
import torch.nn as nn
from sac.utils import BatchRenorm


class Encoder(torch.nn.Module):
    """
    this is the encoding module
    """

    def __init__(
        self,
        input_dim,
        num_layers=2,
        hidden_size=512,
        history_length=1,
        concat_action=False,
        dropout=0.0,
    ):
        """
        state_dim: the state dimension
        stacked_frames: #timesteps considered in history
        hidden_size: hidden layer size
        num_layers: how many layers

        the input state should be of size [batch, stacked_frames, state_dim]
        the output should be of size [batch, hidden_size]
        """
        super().__init__()
        self.hidden_size = self.feature_dim = hidden_size

    def get_feature_dim(self):
        return self.feature_dim
    
    def forward(self, states, actions=None):
        return None


class MLPEncoder(Encoder):
    def __init__(
        self,
        input_dim,
        num_layers=2,
        hidden_size=512,
        history_length=1,
        concat_action=False,
        dropout=0.0,
    ):
        super().__init__(
            input_dim=input_dim,
            num_layers=num_layers,
            hidden_size=hidden_size,
            history_length=history_length,
            concat_action=concat_action,
            dropout=dropout,
        )

        layers = []
        for i in range(num_layers):
            input_dim = hidden_size if i > 0 else input_dim * history_length
            layers.append(torch.nn.Linear(input_dim, hidden_size))
            layers.append(torch.nn.ReLU())

        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)  # mlp will flatten time sequence
        return self.net(x)


class CNNEncoder(Encoder):
    def __init__(
        self,
        input_dim,
        num_layers=2,
        hidden_size=512,
        history_length=1,
        concat_action=False,
        dropout=0.0,):
        super().__init__(
            input_dim=input_dim,
            num_layers=num_layers,
            hidden_size=hidden_size,
            history_length=history_length,
            concat_action=concat_action,
            dropout=dropout,
        )
        print(self.feature_dim)
        #print(input_dim)
        layers = []
        if num_layers > 1:
            for i in range(num_layers - 1):
                input_channel = hidden_size if i > 0 else input_dim[0]
                layers.append(
                    nn.Conv1d(
                        in_channels=input_channel,
                        out_channels=hidden_size,
                        kernel_size=3,
                        padding=1,
                    )
                )
                layers.append(nn.ReLU())

            layers.extend(
                [
                    nn.Conv1d(
                        in_channels=hidden_size,
                        out_channels=hidden_size,
                        kernel_size=history_length,
                        padding=0,
                    ),
                    nn.ReLU(),
                ]
            )
        else:
            layers.extend(
                [
                    nn.Conv1d(
                        in_channels=input_dim,
                        out_channels=hidden_size,
                        kernel_size=history_length,
                        padding=0,
                    ),
                    nn.ReLU(),
                ]
            )

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        print('X shape')
        print(x.shape)
        #x = x.permute(0, 2, 1)  # [batch, state_dim, seq_len]
        #print('X shape after')
        #print(x.shape)
        x = self.net(x)
        return x.squeeze(-1)


class RNNEncoder(Encoder):
    def __init__(
        self,
        input_dim,
        num_layers=2,
        hidden_size=512,
        history_length=1,
        concat_action=False,
        dropout=0.0,
    ):
        super().__init__(
            input_dim=input_dim,
            num_layers=num_layers,
            hidden_size=hidden_size,
            history_length=history_length,
            concat_action=concat_action,
            dropout=dropout,
        )

        self.num_layers = num_layers
        self.net = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

    def forward(self, x):
        """
        always start with h0 = 0
        """
        h0 = torch.zeros(self.num_layers, x.shape[0], self.hidden_size).to(x.device)
        output, hn = self.net(x, h0)
        y = output[:, -1]
        return y


class DilatedCNNEncoder(Encoder):
    def __init__(self, input_dim, num_layers=2, hidden_size=512, history_length=4):
        super().__init__(
            input_dim=input_dim,
            num_layers=num_layers,
            hidden_size=hidden_size,
            history_length=history_length,
        )

        layers = []
        print("num_layers", num_layers)
        for i in range(num_layers):
            print(2**i)
            layers.append(
                nn.Conv1d(
                    in_channels=input_dim if i == 0 else hidden_size,
                    out_channels=hidden_size,
                    kernel_size=3,
                    dilation=2**i,
                    padding=2**i,
                )
            )
            layers.append(nn.ReLU6())

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # [batch, state_dim, seq_len]
        x = self.net(x)
        return x.squeeze(-1)


class TCNEncoder(Encoder):
    def __init__(
        self,
        input_dim,
        num_layers=2,
        hidden_size=512,
        history_length=10,
        kernel_size=3,
        dropout=0.2,
    ):
        super().__init__(
            input_dim,
            num_layers,
            hidden_size,
            history_length=history_length,
            dropout=dropout,
        )

        layers = []
        dilation = 1

        for i in range(num_layers):
            in_ch = input_dim if i == 0 else hidden_size
            padding = (dilation * (kernel_size - 1)) // 2
            layers.extend(
                [
                    nn.Conv1d(
                        in_ch,
                        hidden_size,
                        kernel_size,
                        dilation=dilation,
                        padding=padding,
                    ),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            dilation *= 2 # to expand the receptive field
        
        layers.append(nn.Conv1d(hidden_size, hidden_size, kernel_size=history_length))
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.net(x)
        return x.squeeze(-1)

def get_activation(activation_choice: str) -> nn.Module:
    if activation_choice.lower() == "relu6":
        return nn.ReLU6
    elif activation_choice.lower() == "tanh":
        return nn.Tanh
    elif activation_choice.lower() == "elu":
        return nn.ELU
    elif activation_choice.lower() == "relu":
        return nn.ReLU
    else:
        raise ValueError(f"Unsupported activation function: {activation_choice}")

class MLP(nn.Module):
    def __init__(self, input_dim, num_layers=2, hidden_layer_size=512,activation="relu6"):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_layer_size
        
        layers = []
        for i in range(num_layers):
            input_dim = hidden_layer_size if i > 0 else self.input_dim
            layers.append(nn.Linear(input_dim, hidden_layer_size))
            layers.append(get_activation(activation)())

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)
    
class MLP_CrossQ(nn.Module):
    def __init__(self, input_dim, num_layers=2, hidden_layer_size=512, activation="relu6"):
        """
        MLP CrossQ style:
        - num_layers: 
            - 1 means BatchRenorm + Linear + Activation + BatchRenorm
            - 2 means BatchRenorm + Linear + Activation + BatchRenorm + Linear + Activation + BatchRenorm
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_layer_size
        self.feature_dim = hidden_layer_size
        
        layers = []
        for i in range(num_layers):
            input_dim = hidden_layer_size if i > 0 else self.input_dim
            layers.append(BatchRenorm(input_dim))
            layers.append(nn.Linear(input_dim, hidden_layer_size))
            layers.append(get_activation(activation)())

        layers.append(BatchRenorm(hidden_layer_size, momentum=0.01))
        self.mlp = nn.Sequential(*layers)
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        print('Forward shpae', x.shape)
        return self.mlp(x)