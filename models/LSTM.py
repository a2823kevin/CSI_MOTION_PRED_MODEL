import torch as t
from torch import nn
from torch.autograd import Variable as V
import torch.optim as optim
import os
import torch.nn as nn
# LSTM
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bidirectional):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
        )

        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        out, (h_n, c_n) = self.lstm(inputs, None)
        outputs = self.fc(h_n.squeeze(0))

        return self.sigmoid(outputs)