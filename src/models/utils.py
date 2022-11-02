import numpy
import torch
import torch.nn as nn

def check_accuracy(device, loader, model):
    losses = []

    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = torch.argmax(y, 1).to(device=device)
            scores = model(x)
            scores = torch.argmax(scores, 1)
            losses.append(int(sum(scores==y))/y.shape[0])

    # Toggle model back to train
    model.train()
    return numpy.mean(losses)