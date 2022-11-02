import torch
import torch.nn as nn

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()

        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class temporal_block(nn.Module):
    def __init__(self, n_input, n_output, kernel_size, dialation, padding, stride=1, dropout=0.2):
        super(temporal_block, self).__init__()

        self.conv1 = nn.Conv1d(n_input, n_output, kernel_size, stride, padding, dialation)
        self.conv1 = nn.utils.weight_norm(self.conv1)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_output, n_output, kernel_size, stride, padding, dialation)
        self.conv2 = nn.utils.weight_norm(self.conv2)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.downsampling = None
        if (n_input!=n_output):
            self.downsampling = nn.Conv1d(n_input, n_output, 1)
        self.relu = nn.ReLU()

        self.network = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1, self.conv2, self.chomp2, self.relu2, self.dropout2)

    def forward(self, x):
        out = self.network(x)
        res = x
        if (self.downsampling is not None):
            res = self.downsampling(res)

        return self.relu(res+out)

class temporal_convolution_network(nn.Module):
    def __init__(self, device, n_input, kernel_size, input_length, channels=[128, 128, 128, 128]):
        super(temporal_convolution_network, self).__init__()

        layers = []
    
        n_level = len(channels)
        for i in range(n_level):
            dialation = 2 ** i

            if (i==0):
                n_in = n_input
            else:
                n_in = channels[i-1]

            n_out = channels[i]

            layers.append(temporal_block(n_in, n_out, kernel_size, dialation, (kernel_size-1)*dialation))
        
        self.network = nn.Sequential(*layers)
        self.flat = nn.Flatten()
        self.dense = nn.Linear(input_length*channels[-1], channels[-1])
        self.softmax = nn.Softmax(1)
        self.to(device)

    def forward(self, x):
        out = self.network(x)
        out = self.dense(self.flat(out))
        return self.softmax(out)

if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = temporal_convolution_network(device, 912, 2, 50)
    data = torch.randn(10, 912, 50).to(device)
    print(network(data))