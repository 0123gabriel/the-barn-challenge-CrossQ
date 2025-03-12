from net import CNNEncoder, MLPEncoder, RNNEncoder, TCNEncoder, DilatedCNNEncoder
import torch
import numpy as np
import torchsummary
#from torchinfo import summary

# Define dimensions
seq_length = 100
input_dim = 24
hidden_dim = 128
output_dim = 64
batch_size = 32
num_layers = 2
history_length = 1

# Create models
models = {
    'CNN': CNNEncoder(input_dim, num_layers, hidden_dim, history_length),
    #'MLP': MLPEncoder(input_dim, num_layers, hidden_dim, history_length),
    'RNN': RNNEncoder(input_dim, num_layers, hidden_dim, history_length),
    'TCN': TCNEncoder(input_dim, num_layers, hidden_dim, history_length),
    'DilatedCNN': DilatedCNNEncoder(input_dim, hidden_dim, history_length)
}

# Test each model
for name, model in models.items():
    print(f"Testing {name}")
    x = torch.randn(batch_size, seq_length, input_dim)
    y = model(x)
    print(f"Output shape: {y.shape}")

    # Print model summary
    torchsummary.summary(model, (seq_length, input_dim))
