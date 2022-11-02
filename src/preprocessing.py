import os
import json
import warnings
import numpy
import matplotlib.pyplot as plt
import pandas
import pywt

warnings.simplefilter(action='ignore', category= Warning)

def get_fpaths_by_time(time):
    files = os.listdir("PC/CSI Collector/record")
    fpaths = []
    for file in files:
        if (file.split("_")[0]==time and file.endswith("csv")):
            fpaths.append(f"PC/CSI Collector/record/{file}")
    
    return fpaths

def eliminate_excluded_subcarriers(fpath, subcarriers):
    fin = pandas.read_csv(fpath)

    #search & eliminate excluded subcarriers
    for num in subcarriers:
        try:
            fin = fin.drop("subcarrier"+str(num).rjust(2, "0")+"_amp", axis=1)
            fin = fin.drop("subcarrier"+str(num).rjust(2, "0")+"_ang", axis=1)
        except KeyError:
            continue
    
    #save file
    fin.to_csv(fpath, index=False)
    print(f"eliminated unused subcarriers: {subcarriers} for {fpath}")

def fix_timestamp(fpath):
    fin = pandas.read_csv(fpath)
    timestamps = fin["timestamp"]
    discontinuities = []

    #search timing that reset
    for i in range(len(timestamps)-1):
        if (timestamps[i]>timestamps[i+1]):
            discontinuities.append(i+1)
    discontinuities.append(len(timestamps))

    for i in range(len(discontinuities)-1):
        for idx in range(discontinuities[i], discontinuities[i+1]):
            fin["timestamp"].loc[idx] += fin["timestamp"][discontinuities[i]-1]

    #search timing that is identical to the previous one
    timestamps = fin["timestamp"]
    for i in range(1, len(timestamps)):
        if (timestamps[i]==timestamps[i-1]):
            fin = fin.drop(i, axis=0)
    
    #save file
    fin.to_csv(fpath, index=False)

def get_sampling_period(fpath):
    fin = pandas.read_csv(fpath)
    fin = fin["timestamp"].diff()
    fin = fin.dropna()
    eq_zero = (fin==0)
    for idx in range(1, len(eq_zero==0)+1):
        if (eq_zero[idx]==True):
            fin = fin.drop(idx)

    #return min & avg
    return (fin.min(), fin.mean())

def interpolate(fpath, sampling_period):
    fin = pandas.read_csv(fpath)
    interval = fin["timestamp"].diff()

    row = {}
    for key in fin.keys():
        row[key] = numpy.NaN
    row["timestamp"] = []

    #search timing to interpolate
    for i in interval.keys():
        if (interval[i]==0):
            continue
        if (sampling_period/interval[i]>=0.4 and sampling_period/interval[i]<=0.7):
            row["timestamp"].append((fin["timestamp"][i]+fin["timestamp"][i-1])/2)
    
    #add to data & interpolate
    fin = pandas.concat([fin, pandas.DataFrame(row)])
    fin = fin.sort_values("timestamp")
    fin = fin.interpolate("linear")

    #save file
    fin.to_csv(fpath, index=False)
    print(f"avg sampling period after interpolation: {get_sampling_period(fpath)[1]}")

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

def synchronize(CSI_fpaths, MP_fpath=None):
    if (MP_fpath is not None):
        CSI_fin = pandas.read_csv(CSI_fpaths)
        with open(MP_fpath, "r") as fin:
            MP_fin = json.load(fin)

        #search datas that are not recorded in json
        timestamps = list(MP_fin.keys())
        time = CSI_fin["timestamp"]
        idx_lst = []
        for key in time.keys():
            if (str(time[key]) not in timestamps):
                idx_lst.append(key)

        #drop
        for i in range(len(idx_lst)):
            CSI_fin = CSI_fin.drop(idx_lst[i], axis=0)

        #save file
        CSI_fin.to_csv(CSI_fpaths, index=False)
        with open(MP_fpath, "w") as fout:
            json.dump(MP_fin, fout)
    else:
        fins = []

        #load files
        for fpath in CSI_fpaths:
            fins.append(pandas.read_csv(fpath))
        
        #find timestamp intersection of all files
        timestamps = set(fins[0]["timestamp"].values.tolist())
        for i in range(1, len(fins)):
            timestamps = timestamps.intersection(set(fins[i]["timestamp"].values.tolist()))

        #drop rows that aren't in the timestamp
        for i in range(0, len(fins)):
            for j in range(len(fins[i]["timestamp"])):
                if (fins[i]["timestamp"][j] not in timestamps):
                    fins[i] = fins[i].drop(j, axis=0)

        #save file
        for i in range(len(fins)):
            fins[i].to_csv(CSI_fpaths[i], index=False)

def rename(fpath):
    fname = "preprocessed_" + fpath.split("/")[-1]
    new_path = ""
    for i in range(0, len(fpath.split("/"))-1):
        new_path += fpath.split("/")[i]
        new_path += "/"
    new_path += fname
    print(new_path)

    with open(new_path, "w") as fout:
        with open(fpath, "r") as fin:
            fout.write(fin.read())

def normalize(data, max_amp, min_amp):
    return (data.select_dtypes(include=[numpy.float64])-min_amp) / (max_amp-min_amp)

def hampel(data, k=7, t=3):
    L = 1.4826
    rolling_median = data.rolling(k).median()
    difference = numpy.abs(rolling_median-data)
    median_abs_deviation = difference.rolling(k).median()
    threshold = t * L * median_abs_deviation
    outlier_idx = difference > threshold
    data[outlier_idx] = rolling_median

    return data

def reduce_noise(data):
    #outlier removal
    data = hampel(data).to_numpy()
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

if __name__=="__main__":
    print(get_sampling_period("PC/CSI Collector/record/20221026200112_30C6F751B4A8.csv"))