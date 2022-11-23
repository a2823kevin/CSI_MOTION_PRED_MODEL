from random import shuffle
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torch import optim
from utils import *
from models.LSTM import *
from models.TCN import *
from models.utils import *

def train_TCN(device, dataset, data_length, settings):
    
    input_size = dataset[0][0].shape[0]
    learning_rate = 1e-4
    batch_size = 50
    num_epochs = 10000

    test_dataset = dataset[len(dataset)*8//10:]
    train_dataset = dataset[0:len(dataset)*8//10]
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size)

    model = temporal_convolution_network(device, input_size, 2, data_length, [int(input_size-(input_size-6)*(0.2*i)) for i in range(1, 6)])
    print(model)

    class_weight = torch.tensor(settings["class_weight"], dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=1/class_weight)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    accuracy_max = 0
    early_stop_patience = 5

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
        print(f"accuracy on training set: {check_accuracy(device, train_loader, model):2f}")
        accuracy = check_accuracy(device, test_loader, model)
        print(f"accuracy on validating set: {accuracy}")
        if (accuracy>accuracy_max):
            torch.save(model.state_dict(), "assets/trained model/csi_tcn")
            if (accuracy-accuracy_max<0.01):
                early_stop_patience -= 1
            else:
                early_stop_patience = 5
            accuracy_max = accuracy
        print(f"max accuracy: {accuracy_max}")
        if (early_stop_patience==0):
            return model

if __name__=="__main__":
    with open("src/settings.json", "r") as fin:
        settings = json.load(fin)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    ds = generate_CSI_dataset("20221109202112", settings, 40, "tcn", n_PCA_components=30)
    shuffle(ds)
    ds4testing = divide_dataset_by_class(ds[len(ds)*8//10:])

    model = train_TCN(device, ds[0:len(ds)*8//10], 40, settings)
    model.eval()

    #testing
    for key in ds4testing.keys():
        test_loader = DataLoader(ds4testing[key], batch_size=1)
        accuracy = check_accuracy(device, test_loader, model)
        print(f"accuracy of action {key} on testing set: {accuracy}")