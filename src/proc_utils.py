import socket
import multiprocessing
import numpy
import torch
from sklearn.decomposition import PCA
from preprocessing import *

'''
def data_updating_proc(lock, indata, data_arr, sta_map):
    while (True):
        if (lock.value==True):
            continue
        try:
            idx = sta_map[indata["client_MAC"]]
            lst = list(data_arr[idx])
        except:
            continue
        lst.append(indata["CSI"])
        data_arr[idx] = lst
        print(data_arr)
'''

'''
def data_preprocessing_proc(lock, data_arr, testing_data, settings, data_length, n_PCA_components):
    while True:
        #lock.value = True
        continue
        dfs = list(data_arr)
        n_Txs = len(dfs)

        print(dfs)
        for i in range(n_Txs):
            while (len(dfs[i])>data_length+1):
                dfs[i].pop(0)
        lock.value = False

        for i in range(n_Txs):
            dfs[i] = eliminate_excluded_subcarriers(numpy.array(dfs[i]), settings, "list")
        df = merge_df(dfs, "numpy")
        df = pandas.DataFrame(df)
        df = convert_to_diff_serie(df)
        df = normalize(df, settings["max_amp_diff"], settings["min_amp_diff"])
        print(df)

        data = numpy.zeros(df.shape, numpy.float64)
        for i in range(df.shape[1]):
            data[:, i] = reduce_noise(df.iloc[:, i])
        print(data)

        pca = None
        if (n_PCA_components is not None):
            if (pca is None):
                pca = PCA(n_PCA_components)
            new_data = numpy.zeros((len(data), n_PCA_components*n_Txs), numpy.float64)
            for i in range(0, n_Txs):
                new_data[:, n_PCA_components*i:n_PCA_components*(i+1)] = pca.fit_transform(data[:, (data.shape[1]//n_Txs)*i:(data.shape[1]//n_Txs)*(i+1)])
            data = new_data
            print(data)
        
        data = data.tolist()
        print(data)
        #lock.acquire()
        for i in range(len(data)):
            testing_data[i] = data
        print(testing_data)
        #lock.release()

'''