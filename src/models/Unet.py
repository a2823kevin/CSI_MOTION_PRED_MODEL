from pyexpat import model
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNBLOCK(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        super(CNNBLOCK, self).__init__()
        self.seq_block = nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        x = self.seq_block(x)
        return x

class CNN_SERIES(nn.Module):
    def __init__(self, n_conv, in_channels, out_channels, padding):
        super(CNN_SERIES, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(n_conv):
            self.layers.append(CNNBLOCK(in_channels, out_channels, padding=padding))
            in_channels = out_channels
            
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, padding, downhill=4):
        super(Encoder, self).__init__()
        self.enc_layers = nn.ModuleList()
        
        for i in range(downhill):
            self.enc_layers += [
                CNN_SERIES(n_conv=3, in_channels=in_channels, out_channels=out_channels, padding=padding),
                nn.MaxPool3d(2, 2, 2)
                               ]
            #in_channels - out_channels
            out_channels *= 2
        
        self.enc_layers.append(CNN_SERIES(n_conv=2, in_channels=in_channels, out_channels=out_channels, padding=padding))
        
    def forward(self, x):
        route_connection = []
        for layer in self.enc_layers:
            if isinstance(layer, CNN_SERIES):
                x = layer(x)
                route_connection.append(x)
            else:
                x = layer(x)
        return x, route_connection            
            
class Decoder(nn.Module):
    def __init__(self,in_channels,out_channels,exit_channels,padding,uphill = 4):
        super(Decoder, self).__init__()
        self.exit_channels = exit_channels
        self.layers = nn.ModuleList()
        
        for i in range(uphill):
            self.layers += [
                nn.ConvTranspose3d(in_channels, out_channels, kernel_size = 2, stride = 2),
                CNN_SERIES(n_conv = 2, in_channels = in_channels, out_channels = out_channels, padding = padding),
            ]
            in_channels //= 2
            out_channels //= 2
        self.layers.append(
        nn.Conv3d(in_channels, exit_channels, kernel_size = 1, padding = padding)
        )

class Unet(nn.Module):
    def __init__(self, in_channels, first_channels,exit_channels, downhill, padding = 0):
        super(Unet, self).__init__()
        self.encoder = Encoder(in_channels, first_channels,padding = padding , downhill = downhill)
        self.Encoder = Decoder(first_channels*(2**downhill),first_channels*(2**(downhill-1)),exit_channels,padding  = padding, uphill = downhill)
        
    def forward(self, x):
        enc_out, routes = self.encoder(x)
        out = self.decoder(enc_out, routes)
        
        return out

if __name__=="__main__":
    model = Unet(1, 1, 4, 4)
    print(model)