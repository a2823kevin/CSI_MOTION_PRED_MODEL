import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, device, input_size, hidden_size, num_layers, data_length, bidirectional=False):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=input_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True
        )

        self.fc = nn.Linear(input_size*data_length, hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.to(device)

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x, None)
        out = out.reshape(out.shape[0], -1)
        outputs = self.fc(out)
        return self.sigmoid(outputs)
