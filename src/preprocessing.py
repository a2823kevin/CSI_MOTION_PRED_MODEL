import os
import sys
import json
import warnings
import numpy
import matplotlib.pyplot as plt
import pandas
import pywt

warnings.simplefilter(action="ignore", category=Warning)

def update_sampling_period():
    avg = 9999
    for fname in os.listdir("assets/training data"):
        if (fname.endswith("csv")):
            fin = pandas.read_csv(f"assets/training data/{fname}")
            fin = fin["timestamp"].diff()
            fin = fin.dropna()
            eq_zero = (fin==0)
            for idx in range(1, len(eq_zero==0)+1):
                if (eq_zero[idx]==True):
                    fin = fin.drop(idx)
        
            if (fin.mean()<avg):
                avg = fin.mean()
    
    with open("src/settings.json", "r") as fin:
        settings = json.load(fin)
    
    settings["CSI_sampling_period"] = avg
    with open("src/settings.json", "w") as fout:
        json.dump(settings, fout)

def get_dfs_by_time(time):
    dfs = []
    for fname in os.listdir("assets/training data"):
        if (fname.split("_")[0]==time and fname.endswith("csv")):
            dfs.append(pandas.read_csv(f"assets/training data/{fname}"))
    
    return dfs

def generate_label(time, start_time=0, end_time=None, action=None):
    action_record = {}

    try:
        with open(f"assets/training data/{time}_label.json", "r") as fin:
            action_record = json.load(fin)
    except:
        pass

    lst = []
    for key in action_record.keys():
        if (float(key)>start_time):
            lst.append(key)
    for key in lst:
        action_record.pop(key)

    action_record[str(start_time)] = action
    if (end_time is not None):
        action_record[str(end_time)] = None

    with open(f"assets/training data/{time}_label.json", "w") as fout:
        json.dump(action_record, fout)

def eliminate_excluded_subcarriers(df, settings, object_type="df"):
    if (object_type=="df"):
        #search & eliminate excluded subcarriers
        for num in settings["excluded_subcarriers"]:
            try:
                df = df.drop("subcarrier"+str(num).rjust(2, "0")+"_amp", axis=1)
                df = df.drop("subcarrier"+str(num).rjust(2, "0")+"_ang", axis=1)
            except KeyError:
                continue
    
    elif (object_type=="numpy"):
        for i in range(len(settings["excluded_subcarriers"])):
            df = numpy.delete(df, settings["excluded_subcarriers"][i]-i-1, 1)

    return df

def fix_timestamp(df, fix_dup=False):
    timestamps = df["timestamp"]
    discontinuities = []

    #search timing that reset
    for i in range(len(timestamps)-1):
        if (timestamps[i]>timestamps[i+1]):
            discontinuities.append(i+1)
    discontinuities.append(len(timestamps))

    for i in range(len(discontinuities)-1):
        for idx in range(discontinuities[i], discontinuities[i+1]):
            df["timestamp"].loc[idx] += df["timestamp"][discontinuities[i]-1]

    if (fix_dup==True):
        #search timing that is identical to the previous one
        timestamps = df["timestamp"]
        for i in range(1, len(timestamps)):
            if (timestamps[i]==timestamps[i-1]):
                df = df.drop(i, axis=0)
    
    return df.reset_index(drop=True)

def synchronize(dfs, strict=False):
    if (strict==True):
        #find timestamp intersection of all files
        timestamps = set(dfs[0]["timestamp"].values.tolist())
        for i in range(1, len(dfs)):
            timestamps = timestamps.intersection(set(dfs[i]["timestamp"].values.tolist()))

        #drop rows that aren't in the timestamp
        for i in range(0, len(dfs)):
            for j in range(len(dfs[i]["timestamp"])):
                if (dfs[i]["timestamp"][j] not in timestamps):
                    dfs[i] = dfs[i].drop(j, axis=0)
            dfs[i] = dfs[i].reset_index(drop=True)
    else:
        min_length = 99999999
        for i in range(1, len(dfs)):
            if (len(dfs[i])<min_length):
                min_length = len(dfs[i])
        for i in range(0, len(dfs)):
            dfs[i] = dfs[i].iloc[:min_length, :]
    return dfs

def merge_df(dfs, object_type="df"):
    if (object_type=="df"):
        for i in range(0, len(dfs)):
            if (i>0):
                dfs[i] = dfs[i].drop("timestamp", axis=1)

            key_dict = {}
            for key in dfs[i].keys():
                if (key[0:10]=="subcarrier"):
                    key_dict[key] = f"T{i+1}R1_{key}"
            dfs[i] = dfs[i].rename(key_dict, axis=1)
            
            if (i>0):
                dfs[0] = dfs[0].join(dfs[i])
    
    elif (object_type=="numpy"):
        for i in range(1, len(dfs)):
            dfs[0] = numpy.concatenate((dfs[0], dfs[i]), axis=1)

    return dfs[0]

def interpolate(df):
    with open("src/settings.json", "r") as fin:
        settings = json.load(fin)

    interval = df["timestamp"].diff()

    row = {}
    for key in df.keys():
        row[key] = numpy.NaN
    row["timestamp"] = []

    #search timing to interpolate
    for i in interval.keys():
        if (interval[i]==0):
            continue
        if (settings["CSI_sampling_period"]/interval[i]>=0.4 and settings["CSI_sampling_period"]/interval[i]<=0.7):
            row["timestamp"].append((df["timestamp"][i]+df["timestamp"][i-1])/2)
    
    #add to data & interpolate
    df = pandas.concat([df, pandas.DataFrame(row)])
    df = df.sort_values("timestamp")
    df = df.interpolate("linear")

    return df.reset_index(drop=True)

def update_class_weight(df):
    with open("src/settings.json", "r") as fin:
        settings = json.load(fin)
    
    settings["class_weight"] = [0 for i in range(6)]
    label = df["label"]
    for i in range(len(label)):
        settings["class_weight"][int(label[i])] += 1
    for i in range(6):
        settings["class_weight"][i] /= len(label)

    with open("src/settings.json", "w") as fout:
        json.dump(settings, fout)

def labeling(df, time, settings, update_weight=False):
    with open(f"assets/training data/{time}_label.json", "r") as fin:
        action_record = json.load(fin)
    
    for key in action_record:
        if (action_record[key] is None):
            action_record[key] = numpy.NaN
    action_record[str(df["timestamp"][0])] = numpy.NaN
    action_record[str(df["timestamp"][len(df)-1])] = numpy.NaN

    timing = list(action_record.keys())
    for i in range(len(timing)):
        timing[i] = float(timing[i])
    timing = sorted(timing)

    df.insert(0, "label", [numpy.NaN for i in range(len(df))])
    ptr = 0
    for i in range(len(df)):
        if (df["timestamp"][i]>=timing[ptr] and df["timestamp"][i]<timing[ptr+1]):
            try:
                df.iloc[i, 0] = settings["action_dict"][action_record[str(timing[ptr])]]
            except KeyError:
                continue
        else:
            ptr += 1
    df = df.dropna().reset_index(drop=True)

    if (update_weight==True):
        update_class_weight(df)

    return df

def update_amp_diff_settings(df):
    with open("src/settings.json", "r") as fin:
        settings = json.load(fin)
    
    data = df.iloc[:, 1:].to_numpy()
    settings["max_amp_diff"] = float(data.max())
    settings["min_amp_diff"] = float(data.min())
    with open("src/settings.json", "w") as fout:
        json.dump(settings, fout)

def convert_to_diff_serie(df, drop_timestamp=True, update_settings=False, object_type="df"):
    if (object_type=="df"):
        if (drop_timestamp==True):
            try:
                df = df.drop("timestamp", axis=1)
            except:
                pass
            df = df.diff()
        else:
            timestamp = df["timestamp"]
            df = df.diff()
            df.iloc[1:len(df), 0] = timestamp[1:]

        df = df.dropna()

        if (update_settings==True):
            update_amp_diff_settings(df)
    
    elif (object_type=="numpy"):
        df = numpy.diff(df, axis=1)
        df = numpy.delete(df, 0, 0)
    return df

def normalize(df, max_val, min_val):
    return (df.select_dtypes(include=[numpy.float64])-min_val) / (max_val-min_val)

def hampel(df, k=7, t=3):
    L = 1.4826
    rolling_median = df.rolling(k).median()
    difference = numpy.abs(rolling_median-df)
    median_abs_deviation = difference.rolling(k).median()
    threshold = t * L * median_abs_deviation
    outlier_idx = difference > threshold
    df[outlier_idx] = rolling_median

    return df

def reduce_noise(df):
    #outlier removal
    data = hampel(df).to_numpy()
    data_length = len(data)

    #wavelet analysis
    wavelet = pywt.Wavelet("sym5")
    max_level = pywt.dwt_max_level(len(data), wavelet.dec_len)

    #decompose
    coefficients = pywt.wavedec(data, wavelet, level=max_level)

    #filtering
    for i in range(len(coefficients)):
        #see coefficients 5% smaller than max coef as noise
        coefficients[i] = pywt.threshold(coefficients[i], 0.05*max(coefficients[i]))

    #reconstruct
    data = pywt.waverec(coefficients, wavelet)

    return data[:data_length]

def plot_CSI_signal(fpath, batch_size=15000, batch_num=1, part="mag", subcarriers=[4, 8, 16, 32, 64]):
    fin = pandas.read_csv(fpath)
    start_idx = (batch_num-1) * batch_size
    end_idx = batch_num * batch_size
    signals = pandas.DataFrame()

    #search & collect data
    for n in subcarriers:
        if ("subcarrier"+str(n).rjust(2, "0")+f"_{part}" in fin.keys()):
            signals["subcarrier"+str(n).rjust(2, "0")] = fin["subcarrier"+str(n).rjust(2, "0")+f"_{part}"].iloc[start_idx:end_idx]

    #plot
    plt.figure()
    plt.xlabel("packets")
    plt.ylabel(part)
    plt.ylim([0, 80])
    for k in signals.keys():
        signals[k].plot(linewidth=0.2)
    plt.legend()
    plt.show()

if __name__=="__main__":
    if (len(sys.argv)>1):
        if (sys.argv[1]=="-l"):
            time = input("time of dataset: ")
            if (len(get_dfs_by_time(time))==0):
                print("No file.")
            else:
                while (True):
                    try:
                        start_time = input("start time (send \"exit\" to exit): ")
                        if (start_time=="exit"):
                            break
                        start_time = float(start_time)
                        end_time = input("end time: ")

                        if (end_time!=""):
                            end_time = float(end_time)
                            if (end_time<=start_time):
                                print("wrong input")
                                continue
                        else:
                            end_time = None
                        action = input("action: ")
                    except:
                        continue
                    generate_label(time, start_time, end_time, action)
                    print(f"labeled \"{action}\" for {start_time}~{end_time}")

    else:
        with open("src/settings.json", "r") as fin:
            settings = json.load(fin)

        dfs = get_dfs_by_time("20221102190533")
        for i in range(len(dfs)):
            dfs[i] = eliminate_excluded_subcarriers(dfs[i])
            dfs[i] = fix_timestamp(dfs[i])
        dfs = synchronize(dfs)
        df = merge_df(dfs)
        df = interpolate(df)
        df = convert_to_diff_serie(df, update_settings=True)
        df = normalize(df, settings["max_amp_diff"], settings["min_amp_diff"])
        print(df)