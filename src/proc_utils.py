import socket
import multiprocessing
import numpy
import torch
from sklearn.decomposition import PCA
from preprocessing import *

def data_updating_proc(indata, data_arr, sta_map):
    while (True):
        idx = sta_map[indata["client_MAC"]]
        data_arr[idx].append(indata["CSI"])

def data_preprocessing_proc(lock, data_arr, testing_data, settings, data_length, n_PCA_components):
    #start while all data has enough length
    while True:
        for data in data_arr:
            if (len(data)<data_length+1):
                continue
        break

    n_Txs = len(data_arr)
    while True:
        with lock.acquire():
            for i in range(n_Txs):
                while (len(data_arr[i])>data_length+1):
                    data_arr[i].pop(0)
            dfs = numpy.array(data_arr)
        lock.release()

        for i in range(n_Txs):
            dfs[i] = eliminate_excluded_subcarriers(dfs[i], settings, "numpy")
        df = merge_df(dfs, "numpy")
        df = pandas.DataFrame(df)
        df = convert_to_diff_serie(df)
        df = normalize(df, settings["max_amp_diff"], settings["min_amp_diff"])

        data = numpy.zeros(df.shape, numpy.float64)
        for i in range(df.shape[1]):
            data[:, i] = reduce_noise(df.iloc[:, i])

        if (n_PCA_components is not None):
            pca = PCA(n_PCA_components)
            new_data = numpy.zeros((len(data), n_PCA_components*n_Txs), numpy.float64)
            for i in range(0, n_Txs):
                new_data[:, n_PCA_components*i:n_PCA_components*(i+1)] = pca.fit_transform(data[:, (data.shape[1]//n_Txs)*i:(data.shape[1]//n_Txs)*(i+1)])
            data = new_data
        
        data = data.tolist()
        lock.acquire()
        for i in range(len(data)):
            testing_data[i] = data
        lock.release()