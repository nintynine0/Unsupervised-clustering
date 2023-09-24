import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
import obspy
from obspy import read
from obspy.signal.trigger import classic_sta_lta, trigger_onset
from scipy.signal import stft
from plotspec import plot_spec
from scipy.interpolate import interp1d
from tqdm import tqdm
from glob import glob
import config

def all_process(event):
    """
    event[0]: Obspy trace object
    event[1]: [starttime, endtime] of a event
    """
    trace = event[0]
    t = trace.stats.starttime
    df = trace.stats.sampling_rate

    start = event[1][0]
    end = event[1][1]
    
    tr1 = trace.copy()
    tr1.trim(t+start/df, t+end/df)
    energy = np.sqrt(np.mean(tr1.data**2))

    # Center the maximum in spectrogram and cut by left 2s + right 2s
    if len(tr1.data)<128:
        time = tr1.stats.starttime
    else:
        f1, t1, zxx1 = stft(tr1.data, fs=df, nperseg=128, noverlap=128*0.9, nfft=512)
        j,k = np.where(np.abs(zxx1)==np.max(np.abs(zxx1))) # j-f,k-t
        time = tr1.stats.starttime + t1[k][0]
    tr2 = trace.copy()
    tr2.trim(time-2, time+2) # window length = 4s
    if len(tr2.data)!=(4*round(df)+1):
        print('false')
        #with open(p+'/../events/false to extract.txt','a') as f:
        #    f.writelines(tr2)
    f2, t2, zxx2 = stft(tr2.data, fs=df, nperseg=512, noverlap=512*0.9, nfft=512)
    f2 = f2[1:]; amp_zxx = np.abs(zxx2)[1:] # remove the first frequency 0Hz
    print(t2, f2) # Copy this to the config file

    """
    # Log the frequency
    new_t = t2
    fmin = 1; fmax = 500/2
    f_log = np.logspace(np.log10(fmin), np.log10(fmax), 50)
    new_f = f_log
    amp_zxx = []
    for i in range(len(t2)):
        y = np.abs(zxx2[:,i])
        #print(f2, f_log)
        f = interp1d(f2, y, kind='linear', bounds_error=False, fill_value='extrapolate')
        amp_zxx.append(f(f_log))
    amp_zxx = np.array(amp_zxx).T # (50, 40)
    """
    
    # Normalization
    amp_zxx = amp_zxx - np.min(amp_zxx)
    amp_zxx = amp_zxx/np.max(amp_zxx)

    # Save figure and spectrogram and waveform
    title = tr2.stats.station+'_'+str(tr2.stats.starttime+2)+'_'+str((end-start)/df)+'s'
    np.savetxt("../events/spectrograms/data/%s.txt"%title, amp_zxx)
    tr2.write("../events/waveforms/%s.sac"%title, format = "SAC") 
    with open("../events/catalog.txt", "a") as f:
        # station    channel    starttime_of_this_event    duration    center_time    RMS
        f.writelines(tr2.stats.station+"   "+tr2.stats.channel+"   "+str(tr1.stats.starttime)+"   "+str((end-start)/df)+"   "+str(tr2.stats.starttime+2)+"   "+str(energy)+"\n")
    plot_spec(t2, f2, tr2.data, amp_zxx, title)


if __name__ == "__main__":
    # Read in
    path = "/home/husir/Desktop/unsupervised_clustering_code/data/"
    days = glob(path+"Line1_RAW_ENZ_Day*")
    for day in days:
        times = glob(day+"/*.Z.sac") # Only use Z-component
        for k in tqdm(range(len(times))):
            fpath = times[k]
            if int(fpath.split('/')[-1][6:9]) in config.STATIONS:
                print(fpath+"  start...")
                st = read(fpath)
                raw_tr = st[0]
    
                # detrend and taper
                tr = raw_tr.copy()
                tr.detrend()
                tr.taper(max_percentage=0.05)
            
                # bandpass filtering
                tr = tr.filter("highpass", freq=1)
            
                # detect events
                t = tr.stats.starttime
                data = tr.data
                df = tr.stats.sampling_rate
                cft = classic_sta_lta(data, int(0.5*df), int(30*df))
                picks = trigger_onset(cft, thres1 = 10, thres2 = 3)
            
                pool = Pool(100)
                events = [[[tr, picks[i]]] for i in range(len(picks))]
                result = pool.starmap_async(all_process, events).get()
                
                pool.close()
                pool.join()
                
                print('-------------------------complete------------------------')
