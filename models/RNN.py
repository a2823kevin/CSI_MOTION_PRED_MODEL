import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, device, input_size, hidden_size, num_layers, num_classes, input_length):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, nonlinearity="relu", batch_first=True)
        self.fc = nn.Linear(hidden_size*input_length, num_classes)
        self.device = device
        self.to(device)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        out, _ = self.rnn(x, h0)
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)
        return out