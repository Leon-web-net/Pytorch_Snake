import torch
import torch.nn as nn
import torch.nn.functional as F

# option 1
class NeuralNet1(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNet1, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linea2 = nn.Linear(hidden_size,1)
        self.sigmoid = nn.Sigmoid()

        # other activation functions:
        # nn.Sigmoid nn.Softmax nn.TanH nn.LeakyReLU

    def forward(self,x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.sigmoid(out)
        return out

# option 2 (use activation functions directly in forward pass)
class NeuralNet2(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNet2, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linea2 = nn.Linear(hidden_size,1)

    def forward(self,x):
        out = torch.relu(self.linear1(x))
        out = torch.sigmoid(self.linear2(out))
        # OTHERS:
        # torch.softmax()
        # torch.tanh()
        # F.leaky_relu() # only available in F api

        return out


model = NeuralNet(input_size=28*28, hidden_size=5)
criterion = nn.BCELoss()
