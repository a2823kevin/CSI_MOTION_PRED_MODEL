import os
import gc
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch import optim

from models.CNN_Classifier import *
from models.RNN import *
from utils import *

def train_classifier():
    #load data & generate dataset
    dataset = None

    #split dataset for train, validate & test
    train_set,valid_set = train_test_split(dataset, test_size=0.3,random_state=40)
    valid_set,test_set = train_test_split(valid_set, test_size=0.5,random_state=42)
    print(len(train_set),len(valid_set),len(test_set))

    #dataloader
    batch_size = 5
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,pin_memory=True) # num_workers=8, 
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True,pin_memory=True) # num_workers=8, 
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    #training
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(device)
    model = Classifier().to(device)
    model.device = device

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-5)

    n_epochs = 3


    for epoch in range(n_epochs):
        # ---------- Training ----------
        model.train()
        train_loss = []
        train_accs = []

        # Iterate the training set by batches.
        for batch in tqdm(train_loader):
            imgs, labels = batch
            # Forward the data. (Make sure data and model are on the same device.)
            logits = model(imgs.to(device=device, dtype=torch.float))   #改成浮點數
            loss = criterion(logits, labels.to(device))

            optimizer.zero_grad()

            loss.backward()

            # Clip the gradient norms for stable training.
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)

            # Update the parameters with computed gradients.
            optimizer.step()
            acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

            train_loss.append(loss.item())
            train_accs.append(acc)

        train_loss = sum(train_loss) / len(train_loss)
        train_acc = sum(train_accs) / len(train_accs)
        print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

        # ---------- Validation ----------
        model.eval()

        valid_loss = []
        valid_accs = []

        # Iterate the validation set by batches.
        for batch in tqdm(valid_loader):

            imgs, labels = batch

            with torch.no_grad():
                logits = model(imgs.to(device=device, dtype=torch.float))

            loss = criterion(logits, labels.to(device))

            # Compute the accuracy for current batch.
            acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

            # Record the loss and accuracy.
            valid_loss.append(loss.item())
            valid_accs.append(acc)

        # The average loss and accuracy for entire validation set is the average of the recorded values.
        valid_loss = sum(valid_loss) / len(valid_loss)
        valid_acc = sum(valid_accs) / len(valid_accs)

        print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")

        #save model
        torch.save(model.state_dict(), "./trained model/csi_classifier")

def train_RNN(device):
    input_size = 128
    sequence_length = 1
    hidden_size = 256
    num_layers = 8
    num_classes = 33
    learning_rate = 6e-5
    batch_size = 64
    num_epochs = 20

    train_dataset = generate_CSI_dataset("training data/20220818111807_8CCE4E9A045C_mp_skeleton.csv", "regression")
    test_dataset = train_dataset[len(train_dataset)*8//10:]
    train_dataset = train_dataset[0:len(train_dataset)*8//10]

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size)

    model = RNN(input_size, hidden_size, num_layers, num_classes, device)
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

def train_TCN(device):
    input_size = 128
    learning_rate = 5e-6
    batch_size = 64
    num_epochs = 20

    train_dataset = generate_CSI_dataset("training data/20220818111807_8CCE4E9A045C_mp_skeleton.csv", "regression")
    test_dataset = train_dataset[len(train_dataset)*8//10:]
    train_dataset = train_dataset[0:len(train_dataset)*8//10]
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size)

    model = temporal_convolution_network(input_size, 2, [109, 90, 71, 52, 33])
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
    train_TCN(device)