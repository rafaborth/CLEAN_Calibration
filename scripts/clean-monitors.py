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
import statsmodels.graphics.tsaplots as tsa

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

    return ave5min,ave15min, gaps


def selectWindow(ave15min,nSensor):
    windows = []
    timeWindows = []
    count=[]
    countTime = []
    cc = 0
    for ii in range(0,ave15min.shape[0]):
        if np.isnan(ave15min['Sensor_'+str(nSensor)][ii])==False:
            count.append(ave15min['Sensor_'+str(nSensor)][ii])
            countTime.append(ave15min.index[ii])

        else:
            cc = cc+1
            if cc < 2:
                count.append(ave15min['Sensor_'+str(nSensor)][ii])
                countTime.append(ave15min.index[ii])
            else:
                windows.append(count)
                timeWindows.append(countTime)
                count=[]
                countTime=[]
                cc = 0
    windows = [x for x in windows if x]
    timeWindows = [x for x in timeWindows if x]
    adel = []
    dataWin=[]
    dateTimeWin=[]
    for ii,win in enumerate(windows):
        if len(win)>5*24:
            print('OK timeWindow')
            dataWin.append(win)
            dateTimeWin.append(timeWindows[ii])
        else:
            adel.append(ii)
            

    return dataWin,dateTimeWin


def plotWindows(windows,timeWindows):
    
    winLen = len(windows)
    fig, ax = plt.subplots(winLen)
    stat = pd.DataFrame()
    stat['autocorr'] = np.zeros(winLen)
    stat['min']=  np.zeros(winLen)
    stat['max'] =  np.zeros(winLen)
    stat['mean']=  np.zeros(winLen)
    stat['std'] = np.zeros(winLen)

    for ii in range(0,winLen):
        ax[ii].plot(timeWindows[ii],windows[ii])
        print(np.isnan(windows[ii]).sum())
        #print(np.(windows[ii], windows[ii], mode='full'))
        #stat['autocorr'][ii] = np.correlate(windows[ii], windows[ii], mode='full')
        stat['min'][ii] = np.nanmin(windows[ii])
        stat['max'][ii] = np.nanmax(windows[ii])
        stat['mean'][ii] = np.nanmax(windows[ii])
        stat['std'][ii] = np.nanstd(windows[ii])
        

    fig, ax = plt.subplots()
    
    for ii in range(0,winLen):
        ax.plot(timeWindows[ii],windows[ii])
        print(np.isnan(windows[ii]).sum())
        
    fig, ax = plt.subplots(winLen)
    
    for ii in range(0,winLen):
        data = pd.DataFrame()
        data['timeseries'] = windows[ii]
        data_filled = data.fillna(np.nanmean(windows[ii]))
        tsa.plot_acf(data_filled, lags=100, alpha=0.05, missing ='raise',
                     title='Auto-correlation coefficients for lags 1 through 40',ax = ax[ii])
    
    return stat


 


folder_path = r'C:\Users\rafab\OneDrive\Documentos\CLEAN_Calibration\data\data_clean\dados_brutos'
#folder_path = '/mnt/sdb1/CLEAN_Calibration/data/2.input_equipo/dados_brutos'
monitors = openMonitor(folder_path,'O3')
ave5min,ave15min, gaps = averages (monitors)
dataWin,dateTimeWin = selectWindow(ave15min,1)
stat = plotWindows(dataWin,dateTimeWin)
# https://timeseriesreasoning.com/contents/correlation/

    
