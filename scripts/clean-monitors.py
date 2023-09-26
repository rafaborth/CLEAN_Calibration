#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 12:51:31 2023

@author: leohoinaski
"""

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

def openMonitor(folder_path,pollutant):

    file_list = os.listdir(folder_path) #lista os arquivos do folder_path
    print(file_list) 
    monitors = []
    for file_name in file_list:
        if file_name.startswith('CLEAN_'+pollutant):
            mon = pd.read_csv(folder_path+'/'+file_name)
            idx = pd.DatetimeIndex(mon['DateTime'])
            mon.set_index(idx, inplace=True)
            monitors.append(mon)
    
    monitors = pd.concat(monitors, axis=1)
    monitors = monitors[['measuring']]
    col=[]
    for ii in range(0,monitors.shape[1]):
        col.append('Sensor_'+str(ii+1))
    
    monitors.columns = col
        
    return monitors


    
def averages (monitors):
    monitors['year'] = monitors.index.year
    monitors['month'] = monitors.index.month
    monitors['day'] = monitors.index.day
    monitors['hour'] = monitors.index.hour
    monitors['minute'] = monitors.index.minute
    monitors['datetime'] = monitors.index
    #minuteCount = monitors.groupby(['year','month','day','hour','minute']).count()
    ave15min = monitors.resample(rule='15Min', on='datetime').mean()
    ave30min = monitors.resample(rule='30Min', on='datetime').mean()
    ave60min = monitors.resample(rule='60Min', on='datetime').mean()
    ave1min = monitors.resample(rule='1Min', on='datetime').mean()
    ave5min = monitors.resample(rule='5Min', on='datetime').mean()
    
    gaps =[np.isnan(ave1min.iloc[:,0]).sum(),
           np.isnan(ave5min.iloc[:,0]).sum(),
           np.isnan(ave15min.iloc[:,0]).sum(),
           np.isnan(ave30min.iloc[:,0]).sum(),
           np.isnan(ave60min.iloc[:,0]).sum()]

    return ave15min, gaps


def selectWindow(ave15min):
    windows = []
    count=[]
    cc = 0
    for ii in range(0,ave15min.shape[0]):
        if np.isnan(ave15min['Sensor_1'][ii])==False:
            count.append(ave15min['Sensor_1'][ii])
            cc = 0

        else:
            cc = cc+1
            if cc < 4:
                count.append(ave15min['Sensor_1'][ii])
            else:
                windows.append(count)
                count=[]

    windows = [x for x in windows if x]
    return windows
    
    
    
# def dateTimeCorrection(minuteAve):
#     clean = pd.DataFrame()
#     minuteAve = minuteAve.reset_index()
#     minuteAve['datetime'] = pd.to_datetime(minuteAve[['year','month','day','hour','minute']])
#     minuteAve.set_index(minuteAve['datetime'], inplace=True)
#     datePfct = pd.date_range(minuteAve['datetime'].min(),minuteAve['datetime'].max(),freq='1min')  
#     clean['datetime'] = datePfct
#     clean.set_index(datePfct, inplace=True)
#     clean2 = pd.concat([clean,minuteAve], axis=1)
#     minuteCount = clean2.groupby(clean2.index).count()
    
