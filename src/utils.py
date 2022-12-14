import math
import socket
import struct
import multiprocessing
from multiprocessing import Manager
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

def generate_CSI_dataset(time, settings, data_length, model=None, threshold=0.75, n_Txs=3, n_PCA_components=None, return_pca=False):
    dfs = get_dfs_by_time(time)
    for i in range(len(dfs)):
        dfs[i] = eliminate_excluded_subcarriers(dfs[i], settings)
        dfs[i] = fix_timestamp(dfs[i])
    dfs = synchronize(dfs)
    df = merge_df(dfs)

    #preprocess
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

    if (return_pca==True):
        return (dataset, pca)
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

def test_model(data_length, settings, lock, data_arr, testing_data, mode="rt", ds=None, n_PCA_components=None, pca=None):
    if (mode=="rt"):
        #predict
        n_Txs = len(settings["STA_MAC_ARRDS"])

        while True:
            lock.value = True
            arr = list(data_arr)
            lock.value = False
            start = True
            for i in range(n_Txs):
                print(len(arr[i]))
                if (len(arr[i])<data_length+1):
                    start = False
            if (start==False):
                continue

            for i in range(n_Txs):
                arr[i] = eliminate_excluded_subcarriers(numpy.array(arr[i]), settings, "numpy")
            df = merge_df(arr, "numpy")
            df = pandas.DataFrame(df)
            df = convert_to_diff_serie(df)
            df = normalize(df, settings["max_amp_diff"], settings["min_amp_diff"])

            data = numpy.zeros(df.shape, numpy.float64)
            for i in range(df.shape[1]):
                data[:, i] = reduce_noise(df.iloc[:, i])

            if (n_PCA_components is not None):
                new_data = numpy.zeros((len(data), n_PCA_components*n_Txs), numpy.float64)
                for i in range(0, n_Txs):
                    new_data[:, n_PCA_components*i:n_PCA_components*(i+1)] = pca.transform(data[:, (data.shape[1]//n_Txs)*i:(data.shape[1]//n_Txs)*(i+1)])
                data = new_data

            testing_data.put(data.tolist())




if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    with open("src/settings.json", "r") as fin:
        settings = json.load(fin)

    ds, pca = generate_CSI_dataset("20221214205155", settings, 40, "tcn", n_PCA_components=30, return_pca=True)
    print(len(ds))

    input_size = ds[0][0].shape[0]
    data_length = 40
    model = temporal_convolution_network(device, input_size, 2, data_length, [int(input_size-(input_size-6)*(0.2*i)) for i in range(1, 6)])
    model = load_weights(model, "assets/trained model/csi_tcn")
    print(model)

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("", 3333))
    print("started UDP server.")

    for i in range(10):
        s.sendto("rx".encode(), (settings["AP_IP_ADDR"], 3333))

    #init
    model.eval()
    action_dict = {
        0: "no_person", 
        1: "standing",
        2: "walking",
        3: "getting_up_down",
        4: "jumping",
        5: "waving_hands"
    }
    n_PCA_components = 30
    indata = {}
    sta_map = {}
    data_arr = Manager().list()
    testing_data = multiprocessing.Queue()
    lock = multiprocessing.Value("i", False)

    n_Txs = len(settings["STA_MAC_ARRDS"])
    for i in range(n_Txs):
        sta_map[settings["STA_MAC_ARRDS"][i]] = i
        data_arr.append([])

    testing_proc = multiprocessing.Process(target=test_model, args=(data_length, settings, lock, data_arr, testing_data, "rt", None, n_PCA_components, pca))
    testing_proc.start()

    while True:
        (packet, _) = s.recvfrom(200)
        data = parse2csi(packet)
        if (data["client_MAC"] not in settings["STA_MAC_ARRDS"]):
            continue

        if (lock.value==False):
            idx = sta_map[data["client_MAC"]]
            lst = data_arr[idx]
            while (len(lst)>data_length):
                lst.pop(0)
            lst.append(data["CSI"])
            data_arr[idx] = lst
        
        if (testing_data.qsize()>0):
            d = testing_data.get(True)
            d = torch.tensor(d, dtype=torch.float).reshape(1, n_PCA_components*n_Txs, data_length).to(device)
            scores = model(d)
            print(f"predicted action: {action_dict[int(torch.argmax(scores, 1))]}")