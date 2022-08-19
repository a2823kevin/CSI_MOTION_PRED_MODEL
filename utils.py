import json
import numpy
import pandas
import cv2
import torch
from models.RNN import *
from mp_utils import *

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
            data = (torch.tensor(data.transpose().to_numpy()).type(torch.float))
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

def test_model(device, model, drawing_utils, mode="from_ds", ds=None):
        model.eval()
        if (mode=="from_ds"):
            for i in range(len(ds)):
                pcanvas, tcanvas = numpy.copy(drawing_utils["canvas"]), numpy.copy(drawing_utils["canvas"])
                prediction = model(ds[i][0].to(device=device).reshape(1, 128, 128))
                prediction_lm = generate_landmark(drawing_utils["landmark_template"], prediction)
                solutions.drawing_utils.draw_landmarks(pcanvas, prediction_lm, drawing_utils["connection"], drawing_utils["pls"])
                pcanvas = cv2.flip(pcanvas, 1)
                cv2.putText(pcanvas, "prediction", (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))

                target_lm = generate_landmark(drawing_utils["landmark_template"], ds[i][1])
                solutions.drawing_utils.draw_landmarks(tcanvas, target_lm, drawing_utils["connection"], drawing_utils["pls"])
                tcanvas = cv2.flip(tcanvas, 1)
                cv2.putText(tcanvas, "target", (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
                cv2.imshow("Skeleton mask", numpy.concatenate((pcanvas, tcanvas), 1))
                cv2.waitKey(1)

if __name__=="__main__":
    with open("settings.json", "r") as fin:
        settings = json.load(fin)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = generate_CSI_dataset("training data/20220818111807_8CCE4E9A045C_mp_skeleton.csv", "regression")

    input_size = 128
    hidden_size = 256
    num_layers = 8
    num_classes = 33

    model = RNN(input_size, hidden_size, num_layers, num_classes, device)
    model.load_state_dict(torch.load("trained model\csi_rnn"))

    drawing_utils = {}
    drawing_utils["landmark_template"] = get_landmark_template()
    drawing_utils["connection"] = get_simplified_pose_connections()
    drawing_utils["pls"] = get_simplified_pose_landmarks_style()
    drawing_utils["canvas"] = numpy.zeros((settings["canvas_height"], settings["canvas_width"], 3), dtype=numpy.uint8)

    test_model(device, model, drawing_utils, ds=ds)