import numpy
import torch
import torch.nn as nn

def RMSE(prediction, target):
    mse = nn.MSELoss()
    return torch.sqrt(mse(prediction, target)).cpu()

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