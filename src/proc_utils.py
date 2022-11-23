import socket
import multiprocessing
import numpy
import torch
from sklearn.decomposition import PCA

from utils import parse2csi
from preprocessing import *

def UDP_server_proc(proc_dict, settings):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("", 3333))
    print("started UDP server.")

    for i in range(10):
        s.sendto("rx".encode(), (settings["AP_IP_ADDR"], 3333))
    
    for mac_addr in settings["STA_MAC_ARRDS"]:
        proc_dict[f"{mac_addr}_data"] = []

    data_gen_proc = multiprocessing.Process(target=data_generating_proc, args=(proc_dict, settings))
    data_gen_proc.start()

    while (True):
        (indata, _) = socket.recvfrom(200)
        csi = parse2csi(indata)
        if (csi["client_MAC"] in settings["STA_MAC_ARRDS"]):
            proc_dict[f"{csi['client_MAC']}_data"].append(csi["CSI"])

def data_generating_proc(proc_dict, settings):
    while (True):
        for i in range(len(settings["STA_MAC_ARRDS"])):
            if (len(proc_dict[f"{settings['STA_MAC_ARRDS'][i]}_data"])>proc_dict["data_length"]+1):
                proc_dict["dfs"][i] = numpy.array(proc_dict[f"{settings['STA_MAC_ARRDS'][i]}_data"])[-50:-1]
                while (len(proc_dict[f"{settings['STA_MAC_ARRDS'][i]}_data"])>proc_dict["data_length"]+1):
                    proc_dict[f"{settings['STA_MAC_ARRDS'][i]}_data"].pop(0)


def data_preprocessing_proc(proc_dict, settings, n_PCA_components):
    start = False
    while (start==False):
        start = True
        for df in proc_dict["dfs"]:
            if (df is None):
                start = False

    n_Txs = len(proc_dict["dfs"])
    while (True):
        for i in range(n_Txs):
            proc_dict["dfs"][i] = eliminate_excluded_subcarriers(proc_dict["dfs"][i], settings, "numpy")
        df = merge_df(proc_dict["dfs"], "numpy")
        df = pandas.DataFrame(df)
        df = convert_to_diff_serie(df)
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

        proc_dict["data"] = torch.tensor(datas, dtype=torch.float).reshape(1, n_PCA_components*n_Txs, proc_dict["data_length"]).to(proc_dict["device"])