import os
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torch import optim
from models.RNN import *
from models.LSTM import *
from models.TCN import *
from models.utils import *
from utils import *

def train_RNN(device, ds_path, data_length):
    input_size = get_feature_num(ds_path)
    hidden_size = 256
    num_layers = 8
    num_classes = 33
    learning_rate = 6e-5
    batch_size = 64
    num_epochs = 20

    train_dataset = generate_CSI_dataset(ds_path, "regression")
    test_dataset = train_dataset[len(train_dataset)*8//10:]
    train_dataset = train_dataset[0:len(train_dataset)*8//10]

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size)

    model = RNN(device, input_size, hidden_size, num_layers, num_classes, data_length)
    print(model)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    loss_min = 9999

    for epoch in range(num_epochs):
        for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
            data = data.to(device=device)
            targets = targets.to(device=device)
            # forward
            scores = model(data).reshape(data.shape[0], 1, num_classes)
            loss = criterion(scores, targets)

            # backward
            optimizer.zero_grad()
            loss.backward()
            
            # gradient descent update step/adam step
            optimizer.step()
        print(f"epoch {epoch}:")
        print(f"RMSE on training set: {check_accuracy(device, train_loader, model):2f}")
        loss = check_accuracy(device, test_loader, model)
        print(f"RMSE on test set: {loss:.2f}")
        if (loss<loss_min):
            torch.save(model.state_dict(), "./trained model/csi_rnn")
            loss_min = loss

def train_LSTM(device, ds_path, data_length):
    input_size = get_feature_num(ds_path)
    learning_rate = 5e-6
    batch_size = 50
    num_epochs = 20

    train_dataset = generate_CSI_dataset(ds_path, "regression", "lstm")
    test_dataset = train_dataset[len(train_dataset)*8//10:]
    train_dataset = train_dataset[0:len(train_dataset)*8//10]
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size)

    model = LSTM(device, input_size, 33, 8, data_length)
    print(model)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    loss_min = 9999

    for epoch in range(num_epochs):
        for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
            data = data.to(device=device)
            targets = targets.to(device=device)
            # forward
            scores = model(data)
            loss = criterion(scores, targets)

            # backward
            optimizer.zero_grad()
            loss.backward()
            
            # gradient descent update step/adam step
            optimizer.step()
        print(f"epoch {epoch}:")
        print(f"RMSE on training set: {check_accuracy(device, train_loader, model):2f}")
        loss = check_accuracy(device, test_loader, model)
        print(f"RMSE on test set: {loss:.2f}")
        if (loss<loss_min):
            torch.save(model.state_dict(), "./trained model/csi_lstm")
            loss_min = loss

def train_TCN(device, ds_path, data_length):
    input_size = get_feature_num(ds_path)
    learning_rate = 5e-6
    batch_size = 50
    num_epochs = 20
    channels = [input_size-(input_size-33)//5, input_size-(input_size-33)//5*2, input_size-(input_size-33)//5*3, input_size-(input_size-33)//5*4, 33]

    train_dataset = generate_CSI_dataset(ds_path, "regression")
    test_dataset = train_dataset[len(train_dataset)*8//10:]
    train_dataset = train_dataset[0:len(train_dataset)*8//10]
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size)

    model = temporal_convolution_network(device, input_size, 2, data_length, channels)
    print(model)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    loss_min = 9999

    for epoch in range(num_epochs):
        for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
            data = data.to(device=device)
            targets = targets.to(device=device)
            # forward
            scores = model(data)
            loss = criterion(scores, targets)

            # backward
            optimizer.zero_grad()
            loss.backward()
            
            # gradient descent update step/adam step
            optimizer.step()
        print(f"epoch {epoch}:")
        print(f"RMSE on training set: {check_accuracy(device, train_loader, model):2f}")
        loss = check_accuracy(device, test_loader, model)
        print(f"RMSE on test set: {loss:.2f}")
        if (loss<loss_min):
            torch.save(model.state_dict(), "./trained model/csi_tcn")
            loss_min = loss

if __name__=="__main__":
    os.environ['CUDA_LAUNCH_BLOCKING']='1'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds_path = "training data/20220915203948_8CCE4E9A045C_mp_skeleton.csv"
    data_length = 25

    train_RNN(device, ds_path, data_length)
    #train_LSTM(device, ds_path, data_length)
    #train_TCN(device, ds_path, data_length)
    