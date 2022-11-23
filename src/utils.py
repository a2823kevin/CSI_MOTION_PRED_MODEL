import os
import math
import socket
import struct
import multiprocessing
from multiprocessing import Manager
import pandas
import torch
from sklearn.decomposition import PCA

from preprocessing import *
from models.TCN import *
from proc_utils import *

def parse2csi(packet, amp_only=True):
    csi = {}
    csi["client_MAC"] = packet[0:6].hex(":").upper()
    csi["recieved_time"] = struct.unpack("f", packet[6:10])[0]
    csi["CSI"] = []
    for i in range(10, 138, 2):
        if (amp_only==True):
            csi["CSI"].append(math.sqrt(struct.unpack("b", packet[i:i+1])[0]**2+struct.unpack("b", packet[i+1:i+2])[0]**2))
        else:
            subcarrier = []
            subcarrier.append(math.sqrt(struct.unpack("b", packet[i:i+1])[0]**2+struct.unpack("b", packet[i+1:i+2])[0]**2))
            subcarrier.append(math.atan2(struct.unpack("b", packet[i:i+1])[0], struct.unpack("b", packet[i+1:i+2])[0]))
            csi["CSI"].append(subcarrier)
    return csi

def generate_CSI_dataset(time, settings, data_length, model=None, threshold=0.75, n_Txs=3, n_PCA_components=None):
    dfs = get_dfs_by_time(time)
    for i in range(len(dfs)):
        dfs[i] = eliminate_excluded_subcarriers(dfs[i], settings)
        dfs[i] = fix_timestamp(dfs[i])
    dfs = synchronize(dfs)
    df = merge_df(dfs)

    #preprocess
    df = interpolate(df)
    df = labeling(df, time, settings, True)
    labels = df["label"].iloc[1:].reset_index(drop=True)
    df = convert_to_diff_serie(df.iloc[:, 1:], update_settings=False)
    df = normalize(df, settings["max_amp_diff"], settings["min_amp_diff"])

    datas = numpy.zeros(df.shape, numpy.float64)
    for i in range(df.shape[1]):
        datas[:, i] = reduce_noise(df.iloc[:, i])

    if (n_PCA_components is not None):
        pca = PCA(n_PCA_components)
        new_datas = numpy.zeros((len(datas), n_PCA_components*n_Txs), numpy.float64)
        for i in range(0, n_Txs):
            new_datas[:, n_PCA_components*i:n_PCA_components*(i+1)] = pca.fit_transform(datas[:, (datas.shape[1]//n_Txs)*i:(datas.shape[1]//n_Txs)*(i+1)])
        datas = new_datas

    dataset = []
    for i in range(len(datas)-data_length):
        data = datas[i:i+data_length, :]
        action_count = labels.iloc[i:i+data_length].value_counts()

        main_action = action_count.idxmax()
        if (action_count[main_action]/data_length>=threshold):
            data = torch.tensor(data, dtype=torch.float)
            if (model=="tcn"):
                data = torch.transpose(data, 0, 1)
            label = [0 for i in range(0, 6)]
            label[int(main_action)] = 1
            label = torch.tensor(label, dtype=torch.float)
            dataset.append((data, label))

    return dataset

def merge_datasets(dataset_lst):
    for i in range(1, len(dataset_lst)):
        for j in range(len(dataset_lst[i])):
            dataset_lst[0].append(dataset_lst[i][j])
    return dataset_lst[0]

def divide_dataset_by_class(ds):
    action_dict = {
        0: "no_person", 
        1: "standing",
        2: "walking",
        3: "getting_up_down",
        4: "jumping",
        5: "waving_hands"
    }

    ds_dict = {}
    for key in action_dict.values():
        ds_dict[key] = []
    for (data, label) in ds:
        ds_dict[action_dict[int(torch.argmax(label, 0))]].append((data, label))
    
    return ds_dict

def load_weights(model, fpath):
    with open(fpath, "rb") as fin:
        state_dict = torch.load(fin)
        model.load_state_dict(state_dict)
    return model

def test_model(device, model, data_length, settings, mode="rt", ds=None, n_Txs=3, n_PCA_components=None):
    model.eval()
    action_dict = {
        0: "no_person", 
        1: "standing",
        2: "walking",
        3: "getting_up_down",
        4: "jumping",
        5: "waving_hands"
    }

    if (mode=="rt"):
        #init
        proc_dict = Manager().dict()
        proc_dict["data_length"] = data_length
        proc_dict["dfs"] = []
        for i in range(len(settings["STA_MAC_ARRDS"])):
            proc_dict["dfs"][i] = None
        proc_dict["data"] = None

        #process
        rx_proc = multiprocessing.Process(target=UDP_server_proc, args=(proc_dict, settings))
        rx_proc.start()
        preprocessing_proc = multiprocessing.Process(target=data_preprocessing_proc, args=(proc_dict, settings))
        preprocessing_proc.start()

        #predict
        while True:
            if (proc_dict["data"] is not None):
                scores = model(proc_dict["data"])
                print(f"predicted action: {action_dict[int(torch.argmax(scores, 1))]}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    with open("src/settings.json", "r") as fin:
        settings = json.load(fin)

    ds = generate_CSI_dataset("20221109202112", settings, 40, "tcn", n_PCA_components=30)

    input_size = ds[0][0].shape[0]
    data_length = 40
    model = temporal_convolution_network(device, input_size, 2, data_length, [int(input_size-(input_size-6)*(0.2*i)) for i in range(1, 6)])
    model = load_weights(model, "assets/trained model/csi_tcn")
    print(model)
    test_model(device, model, data_length, settings, n_PCA_components=30)