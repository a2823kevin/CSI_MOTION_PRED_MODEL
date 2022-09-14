import numpy
import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, device):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, nonlinearity="relu", batch_first=True)
        self.fc = nn.Linear(hidden_size*128, num_classes)
        self.device = device
        self.to(device)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        out, _ = self.rnn(x, h0)
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)
        return out

def RMSE(prediction, target):
    mse = nn.MSELoss()
    return torch.sqrt(mse(prediction, target))

def check_accuracy(device, loader, model):
    losses = []

    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            scores = model(x)
            losses.append(RMSE(scores, y))

    # Toggle model back to train
    model.train()
    return numpy.mean(losses)