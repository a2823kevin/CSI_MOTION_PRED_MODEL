import json
import numpy
import pandas
import cv2
import torch

def generate_CSI_dataset(fpath, ds_for, label=None, size=128):
    #load data
    if (ds_for!="segmentation"):
        fin = pandas.read_csv(fpath)
    else:
        fin= pandas.read_csv(fpath[0])
        with open(fpath[1], "r") as f:
            MP_fin = json.load(f)

    dataset = []
    #for classifier model
    if (ds_for=="classification"):
        for key in fin.keys():
            if (key[0:10]!="subcarrier"):
                fin = fin.drop(key, axis=1)
        #generate
        for i in range(len(fin)-size):
            data = fin.iloc[i:i+size, :]
            data = (torch.tensor(data.transpose().to_numpy()).type(torch.float)).reshape(1, 128, size)
            dataset.append((data, label))
    
    #for MP skeleton model
    elif (ds_for=="regression"):
        node_name = []
        for n in ["head", "chest", "left_elbow", "left_hand", "right_elbow", "right_hand", "hip", "left_knee", "left_foot", "right_knee", "right_foot"]:
            node_name.append(n+"_x")
            node_name.append(n+"_y")
            node_name.append(n+"_z")
        MP_fin = fin.copy(True)

        for key in fin.keys():
            if (key[0:10]!="subcarrier"):
                fin = fin.drop(key, axis=1)
        for key in MP_fin.keys():
            if (key not in node_name):
                MP_fin = MP_fin.drop(key, axis=1)
        #generate
        for i in range(len(fin)-size):
            data = fin.iloc[i:i+size, :]
            data = (torch.tensor(data.transpose().to_numpy()).type(torch.float)).reshape(1, 128, size)
            label = MP_fin.iloc[i+size-1, :]
            label = (torch.tensor(label.transpose().to_numpy()).type(torch.float)).reshape(1, 33)
            dataset.append((data, label))

    #for MP mask model
    elif (ds_for=="segmentation"):
        with open("./settings.json", "r") as s:
            settings = json.load(s)
        timestamp = fin["timestamp"]
        for key in fin.keys():
            if (key[0:10]!="subcarrier"):
                fin = fin.drop(key, axis=1)
        #generate
        for i in range(len(fin)-size):
            data = fin.iloc[i:i+size, :]
            data = (torch.tensor(data.transpose().to_numpy()).type(torch.float)).reshape(1, 128, size)
            label = numpy.zeros((settings["canvas_height"], settings["canvas_width"]), numpy.uint8)
            (cts, hole_idx) = MP_fin[str(timestamp[i+size-1])]
            for i in range(len(cts)):
                cts[i] = numpy.array(cts[i])
            for i in range(len(cts)):
                if (i<hole_idx):
                    cv2.drawContours(label, cts, i, 1, cv2.FILLED)
                else:
                    cv2.drawContours(label, cts, i, 0, cv2.FILLED)
            label = (torch.tensor(label)).type(torch.float).reshape(1, settings["canvas_height"], settings["canvas_width"])
            dataset.append((data, label))
    
    return dataset

def merge_datasets(dataset_lst):
    for i in range(1, len(dataset_lst)):
        for j in range(len(dataset_lst[i])):
            dataset_lst[0].append(dataset_lst[i][j])
    return dataset_lst[0]

if __name__=="__main__":
    #for classifier
    ds4classifier = generate_CSI_dataset("./training data/20220803143411_8CCE4E9A045C_test.csv", 
    "classification", 
    0)
    print(ds4classifier[0])

    #for MP skeleton
    ds4MP_skeleton = generate_CSI_dataset("./training data/20220803143454_8CCE4E9A045C_mp_skeleton.csv", 
    "regression")
    print(ds4MP_skeleton[0])

    #for MP mask
    ds4MP_mask = generate_CSI_dataset(("./training data/20220803143548_8CCE4E9A045C_mp_mask.csv", "./training data/20220803143548_mask.json"), 
    "segmentation")
    print(ds4MP_mask[0])