import json
import socket
import multiprocessing
from multiprocessing import Manager
import numpy
import pandas
import cv2
import torch

from models.RNN import *
from models.LSTM import *
from models.TCN import *

from mp_utils import *

def get_feature_num(fpath):
    fin= pandas.read_csv(fpath)
    num = 0
    for key in fin.keys():
        if ("subcarrier" in key):
            num += 1
    return num

def generate_CSI_dataset(fpath, ds_for, model=None, label=None, size=25):
    #load data
    if (ds_for!="segmentation"):
        fin = pandas.read_csv(fpath)
    else:
        fin= pandas.read_csv(fpath[0])
        with open(fpath[1], "r") as f:
            MP_fin = json.load(f)

    n_feat = get_feature_num(fpath)
    dataset = []
    #for classifier model
    if (ds_for=="classification"):
        for key in fin.keys():
            if (key[0:10]!="subcarrier"):
                fin = fin.drop(key, axis=1)
        #generate
        for i in range(len(fin)-size):
            data = fin.iloc[i:i+size, :]
            data = (torch.tensor(data.transpose().to_numpy()).type(torch.float)).reshape(1, n_feat, size)
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
            if (model=="lstm"):
                data = torch.transpose(data, 0, 1)
            label = MP_fin.iloc[i+size-1, :]
            label = (torch.tensor(label.transpose().to_numpy()).type(torch.float))
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
            data = (torch.tensor(data.transpose().to_numpy()).type(torch.float)).reshape(1, n_feat, size)
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

def cam_proc(device, drawing_utils):
    mppp = get_mp_pose_proccessor()
    video_stream = cv2.VideoCapture(0)
    while (True):
        (ret, frame) = video_stream.read()
        frame.flags.writeable = False
        sp = get_simplified_pose(frame, mppp, skip_incomplete=False)
        frame.flags.writeable = True
        nodes = []

        if (sp is not None):
            for i in range(len(sp)):
                nodes.append(sp[i][0])
                nodes.append(sp[i][1])
                nodes.append(sp[i][2])
            
            tcanvas = numpy.copy(drawing_utils["canvas"])
            target = torch.tensor(nodes, dtype=torch.float, device=device).reshape(1, 33)
            target_lm = generate_landmark(drawing_utils["landmark_template"], target)
            solutions.drawing_utils.draw_landmarks(tcanvas, target_lm, drawing_utils["connection"], drawing_utils["pls"])
            tcanvas = cv2.flip(tcanvas, 1)
            cv2.putText(tcanvas, "target", (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
            drawing_utils["tcanvas"] = tcanvas
        else:
            continue

def test_model(device, model, drawing_utils, mode="from_ds", ds=None):
        model.eval()
        if (mode=="from_ds"):
            for i in range(len(ds)):
                pcanvas, tcanvas = numpy.copy(drawing_utils["canvas"]), numpy.copy(drawing_utils["canvas"])
                prediction = model(ds[i][0].to(device=device))
                prediction_lm = generate_landmark(drawing_utils["landmark_template"], prediction)
                solutions.drawing_utils.draw_landmarks(pcanvas, prediction_lm, drawing_utils["connection"], drawing_utils["pls"])
                pcanvas = cv2.flip(pcanvas, 1)
                cv2.putText(pcanvas, "prediction", (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))

                target_lm = generate_landmark(drawing_utils["landmark_template"], ds[i][1])
                solutions.drawing_utils.draw_landmarks(tcanvas, target_lm, drawing_utils["connection"], drawing_utils["pls"])
                tcanvas = cv2.flip(tcanvas, 1)
                cv2.putText(tcanvas, "target", (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
                cv2.imshow("Skeleton mask", numpy.concatenate((pcanvas, tcanvas), 1))
                cv2.imwrite(f"test{i}.png", numpy.concatenate((pcanvas, tcanvas), 1))
                cv2.waitKey(10)

        elif (mode=="realtime"):
            #load settings
            with open("settings.json", "r") as fin:
                settings = json.load(fin)

            #create UDP socket
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.bind(("", 3333))
            print("started UDP server.")

            #send "rx" to AP first
            for i in range(10):
                s.sendto("rx".encode(), (settings["AP_IP_ADDR"], 3333))

            camproc = multiprocessing.Process(target=cam_proc, args=(device, drawing_utils))
            camproc.start()

            data = []
            while (True):
                (indata, _) = s.recvfrom(4096)
                try:
                    indata = json.loads(indata.decode())
                    if (indata["client_MAC"] in settings["STA_MAC_ARRDS"]):
                        csi = []
                        for i in range(len(indata["CSI_info"])):
                            csi.append(indata["CSI_info"][i][0])
                            csi.append(indata["CSI_info"][i][1])
                        data.append(csi)
                except:
                    continue

                while (len(data)>128):
                    data.pop(0)
                if (len(data)==128):
                    prediction = model(torch.transpose(torch.tensor(data, dtype=torch.float, device=device), 0, 1))
                    prediction_lm = generate_landmark(drawing_utils["landmark_template"], prediction)
                    pcanvas = numpy.copy(drawing_utils["canvas"])
                    solutions.drawing_utils.draw_landmarks(pcanvas, prediction_lm, drawing_utils["connection"], drawing_utils["pls"])
                    pcanvas = cv2.flip(pcanvas, 1)
                    cv2.putText(pcanvas, "prediction", (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
                    if (drawing_utils["tcanvas"] is not None):
                        cv2.imshow("Skeleton mask", numpy.concatenate((pcanvas, drawing_utils["tcanvas"]), 1))
                        cv2.waitKey(1)

if __name__=="__main__":
    with open("settings.json", "r") as fin:
        settings = json.load(fin)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    ds_path = ""
    ds = generate_CSI_dataset(ds_path, "regression", size=75)
    input_size = get_feature_num(ds_path)

    drawing_utils = Manager().dict()
    drawing_utils["landmark_template"] = get_landmark_template()
    drawing_utils["connection"] = get_simplified_pose_connections()
    drawing_utils["pls"] = get_simplified_pose_landmarks_style()
    drawing_utils["canvas"] = numpy.zeros((settings["canvas_height"], settings["canvas_width"], 3), dtype=numpy.uint8)
    drawing_utils["tcanvas"] = None

    rnn = RNN(device, input_size, 256, 8, 33, 25)
    lstm = LSTM(device, input_size, 33, 8, 25)
    tcn = temporal_convolution_network(device, input_size, 2, 25, channels=[109, 90, 71, 52, 33])

    model = None
    with open("", "rb") as fin:
        model.load_state_dict(torch.load(fin))
    test_model(device, model, drawing_utils, ds=ds)