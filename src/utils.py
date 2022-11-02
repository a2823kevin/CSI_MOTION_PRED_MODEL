import os
import pandas
import torch
from sklearn.decomposition import PCA

from preprocessing import *

def generate_CSI_dataset(time, settings, data_length, model=None, threshold=0.75, n_PCA_components=None):
    dfs = get_dfs_by_time(time)
    for i in range(len(dfs)):
        dfs[i] = eliminate_excluded_subcarriers(dfs[i])
        dfs[i] = fix_timestamp(dfs[i])
    dfs = synchronize(dfs)
    df = merge_df(dfs)

    #preprocess
    df = interpolate(df)
    df = convert_to_diff_serie(df, update_settings=False)
    labels = df["label"]

    df = normalize(df, settings["max_amp_diff"], settings["min_amp_diff"])

    datas = numpy.zeros(df.shape, numpy.float64)
    for i in range(df.shape[1]):
        datas[:, i] = reduce_noise(df.iloc[:, i])
    
    if (n_PCA_components is not None):
        pca = PCA(n_PCA_components)
        new_datas = numpy.zeros((len(datas), n_PCA_components*3), numpy.float64)
        for i in range(0, 3):
            new_datas[:, n_PCA_components*i:n_PCA_components*(i+1)] = pca.fit_transform(datas[:, 52*i:52*(i+1)])
        datas = new_datas

    dataset = []
    action_dict = {
        "standing": 0,
        "walking": 1,
        "get_down": 2,
        "sitting": 3,
        "get_up": 4,
        "lying": 5,
        "no_person": 6
    }

    for i in range(len(datas)-data_length):
        data = datas[i:i+data_length, :]
        action_count = labels.iloc[i:i+data_length].value_counts()
        main_action = action_count.idxmax()
        if (action_count[main_action]/data_length>=threshold):
            data = torch.tensor(data, dtype=torch.float)
            if (model=="tcn"):
                data = torch.transpose(data, 0, 1)
            label = [0 for i in range(0, 7)]
            label[action_dict[main_action]] = 1
            label = torch.tensor(label, dtype=torch.float)
            dataset.append((data, label))

    return dataset

def merge_datasets(dataset_lst):
    for i in range(1, len(dataset_lst)):
        for j in range(len(dataset_lst[i])):
            dataset_lst[0].append(dataset_lst[i][j])
    return dataset_lst[0]

def divide_dataset_by_class(dataset):
    action_dict = {
        0: "standing", 
        1: "walking",
        2: "get_down",
        3: "sitting",
        4: "get_up",
        5: "lying",
        6: "no_person"
    }

    ds_dict = {}
    for key in action_dict.values():
        ds_dict[key] = []
    for (data, label) in dataset:
        ds_dict[action_dict[int(torch.argmax(label, 0))]].append((data, label))
    
    return ds_dict

if __name__=="__main__":
    #preprocess datas
    '''
    ds_folder = "assets/wifi_csi_har_dataset"
    for room in os.listdir(ds_folder):
        for session in os.listdir(f"{ds_folder}/{room}"):
            if (session!=".DS_Store"):
                folder_path = f"{ds_folder}/{room}/{session}"
                merge_data_and_label(folder_path)
    '''

    
    with open("src/settings.json", "r") as fin:
        settings = json.load(fin)

    fpath = "assets/preprocessed_datasets/room_1_session1.csv"
    ds = generate_CSI_dataset(fpath, settings, 25, n_PCA_components=114)
    
    print(divide_dataset_by_class(ds))