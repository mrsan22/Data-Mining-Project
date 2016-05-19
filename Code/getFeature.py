import os, sys, datetime, csv, glob
import numpy as np
import pandas as pd
from scipy import signal
from sklearn.preprocessing import scale

# band pass Buterworth filter
fs = 90 # sampling rate
N  = 5    # Filter order
lowcut_1 = 0.3
highcut_1 = 15
lowcut_2 = 0.6
highcut_2 = 2.5
nyquist = 0.5 * fs

def butterFilter(data,lowcut,highcut):
    low = lowcut / nyquist
    high = highcut / nyquist
    B, A = signal.butter(N, [low, high], btype='band', output='ba')
    data_f = signal.lfilter(B,A,data)
    Y=np.fft.fft(data_f)
    n=len(Y)
    power = abs(Y[0:(n/2)])**2
    freq=np.arange(0,n/2,1)/(n/2.0)*nyquist
    return (freq,power)


def windowCharacter(x):
    tmp = np.zeros((x.shape[0]))
    n=0
    for row in x.iterrows():
        tmp[n] = signalMag(row[1]['X'],row[1]['Y'],row[1]['Z'])
        n=n+1

    # if np.std(tmp) > 5:
    #     return None
    # else:

    p_25 = np.percentile(tmp,25)
    p_75 = np.percentile(tmp,75)
    tmp_25 = [each for each in tmp if each < p_25]
    tmp_75 = [each for each in tmp if each < p_75]

    data_dm = scale(tmp,with_mean=True, with_std=False) # demean data

    (freq_1,power_1) = butterFilter(data_dm,lowcut_1,highcut_1)
    idx_1 = np.argmax(power_1)
    freq_1_sec = np.delete(freq_1,idx_1)
    power_1_sec = np.delete(power_1,idx_1)
    idx_1_sec = np.argmax(power_1_sec)

    (freq_2,power_2) = butterFilter(data_dm,lowcut_2,highcut_2)
    idx_2 = np.argmax(power_2)

    return np.mean(tmp), np.std(tmp), np.median(tmp), np.linalg.norm(tmp_25), np.linalg.norm(tmp_75),np.sum(power_1), freq_1[idx_1],power_1[idx_1], freq_1_sec[idx_1_sec], power_1_sec[idx_1_sec], freq_2[idx_2],power_2[idx_2],freq_1[idx_1]/np.sum(power_1)

def signalMag(x,y,z):
    return np.sqrt(x*x + y*y + z*z)

placements = ['Dominant-Wrist','Dominant-Hip','Dominant-Thigh','Dominant-Ankle','Dominant-Upper-Arm']
activities = ['sitting:-legs-straight','stairs:-inside-and-up','walking:-natural','lying:-on-back']

count = 0

First = True

for file in glob.glob('output/*-activity.csv'):
    f = os.path.basename(file)
    sub = f.split('-')[0]
    if First:
        First = False
    else:
        myfile.close()
        print("Finished processing subject :" + sub)

    fileObj = pd.read_csv(file)
    fName = 'features/' + sub + '-features.csv'
    myfile = open(fName, 'wb')
    wr = csv.writer(myfile, delimiter=',')
    wr.writerow(('SensorLocation','Activity','MeanSM','StDevSM','MdnSM', 'belowPer25SM','belowPer75SM', 'TotPower_0.3_15','FirsDomFre_0.3_15','PowFirsDomFre_0.3_15','SecDomFre_0.3_15','PowSecDomFre_0.3_15','FirsDomFre_0.6_2.5','PowFirsDomFre_0.6_2.5','FirsDomFre_per_TotPower_0.3_15'))
    # myfile.close()
    for site in placements:
        site_fileObj = (fileObj.loc[fileObj['SensorLocation'] == site])
        for act in activities:
            act_site_fileObj = (site_fileObj.loc[site_fileObj['Activity'] == act])
            try:
                first = next(act_site_fileObj.iterrows())[1]
                start_time_stamp = datetime.datetime.utcfromtimestamp(int(first['Timestamp'])/1000.0)
                start_ind = 0
                rowList = []
                ifFirst = True
                for row in act_site_fileObj.iterrows():
                    time_stamp = datetime.datetime.utcfromtimestamp(int(row[1]['Timestamp'])/1000.0)
                    t  = time_stamp - start_time_stamp
                    t_new = (datetime.datetime.min + t).time()
                    t_s = t_new.second
                    t_m =  t_new.microsecond
                    t_m /= 1000000.0
                    t_s = t_s + t_m

                    if t_s > 10:
                        if t_s < 11:
                            end_ind = row[0]
                            new_fileObj = act_site_fileObj.ix[start_ind:end_ind,0:3]
                            A = windowCharacter(new_fileObj)
                            # if A:
                            (mu,sd,md,per25,per75,tot_po,df1_1,po1_1,df1_2,po1_2,df2_1,po2_1, df1_1_per_tot_po) = A
                            print(site,act,mu,sd,md,per25,per75,tot_po,df1_1,po1_1,df1_2,po1_2,df2_1,po2_1, df1_1_per_tot_po)
                            wr.writerow((site,act,mu,sd,md,per25,per75,tot_po,df1_1,po1_1,df1_2,po1_2,df2_1,po2_1, df1_1_per_tot_po))
                        start_ind = row[0]
                        start_time_stamp = time_stamp
            except:
                pass

myfile.close()
print("Finished processing subject :" + sub)


