import os
import json
import warnings
import numpy
import matplotlib.pyplot as plt
import pandas
from pandas.core.frame import DataFrame
import pywt

warnings.simplefilter(action='ignore', category= Warning)

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

def trim(fpath, start=None, end=None, motion=None):
    fin = pandas.read_csv(fpath)
    if (start==None):
        start = fin["timestamp"][0]
    if (end==None):
        end = fin["timestamp"][len(fin)-1]
    
    #search start point & end point
    time = fin["timestamp"]
    (start_idx, end_idx) = (None, None)
    for i in range(len(time)):
        if (time[i]<=start and time[i+1]>=start):
            start_idx = i + 1
            break
    for i in range(start_idx+1, len(time)):
        if (time[i]<=end and time[i+1]>=end):
            end_idx = i
            break
    
    #trim
    fin = fin.iloc[start_idx:end_idx+1, :]

    #save file
    if (motion!=None):
        fpath = fpath[:-4] + f"_{motion}.csv"
    fin.to_csv(fpath, index=False)
    print("clipped record from "+str(fin["timestamp"][start_idx])+" to "+str(fin["timestamp"][end_idx])+f" and saved it to {fpath}")

def get_dfs_by_time(time):
    dfs = []
    for fname in os.listdir("assets/training data"):
        if (fname.split("_")[0]==time and fname.endswith("csv")):
            dfs.append(pandas.read_csv(f"assets/training data/{fname}"))
    
    return dfs

def eliminate_excluded_subcarriers(df):
    with open("src/settings.json", "r") as fin:
        settings = json.load(fin)

    #search & eliminate excluded subcarriers
    for num in settings["excluded_subcarriers"]:
        try:
            df = df.drop("subcarrier"+str(num).rjust(2, "0")+"_amp", axis=1)
            df = df.drop("subcarrier"+str(num).rjust(2, "0")+"_ang", axis=1)
        except KeyError:
            continue
    
    return df

def fix_timestamp(df):
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

    #search timing that is identical to the previous one
    timestamps = df["timestamp"]
    for i in range(1, len(timestamps)):
        if (timestamps[i]==timestamps[i-1]):
            df = df.drop(i, axis=0)
    
    return df.reset_index(drop=True)

def synchronize(dfs):
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
    return dfs

def merge_df(dfs):
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

def update_amp_diff_settings(df):
    with open("src/settings.json", "r") as fin:
        settings = json.load(fin)
    
    data = df.iloc[:, 1:].to_numpy()
    settings["max_amp_diff"] = float(data.max())
    settings["min_amp_diff"] = float(data.min())
    with open("src/settings.json", "w") as fout:
        json.dump(settings, fout)

def convert_to_diff_serie(df, drop_timestamp=True, update_settings=False):
    if (drop_timestamp==True):
        df = df.drop("timestamp", axis=1)
        df = df.diff()
    else:
        timestamp = df["timestamp"]
        df = df.diff()
        df.iloc[1:len(df), 0] = timestamp[1:]

    df = df.dropna()

    if (update_settings==True):
        update_amp_diff_settings(df)

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
    with open("src/settings.json", "r") as fin:
        settings = json.load(fin)

    dfs = get_dfs_by_time("20221102190533")
    for i in range(len(dfs)):
        dfs[i] = eliminate_excluded_subcarriers(dfs[i])
        dfs[i] = fix_timestamp(dfs[i])
        print(dfs[i])
    dfs = synchronize(dfs)
    df = merge_df(dfs)
    df = interpolate(df)
    df = convert_to_diff_serie(df, update_settings=True)
    df = normalize(df, settings["max_amp_diff"], settings["min_amp_diff"])