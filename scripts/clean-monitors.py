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
import statsmodels.graphics.tsaplots as gtsa
import statsmodels.tsa as tsa
import statsmodels.api as sm
from pmdarima.arima import auto_arima
import ruptures as rpt

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
        gtsa.plot_acf(data_filled, lags=len(windows[ii])-1, alpha=0.05, missing ='raise',
                     title='',ax = ax[ii])
    
    fig, ax = plt.subplots(winLen)
    for ii in range(0,winLen):
        data = pd.DataFrame()
        data['timeseries'] = windows[ii]
        data_filled = data.fillna(np.nanmean(windows[ii]))
        gtsa.plot_pacf(data_filled, lags=len(windows[ii])/50,  method="ywm",
                     title='',ax = ax[ii])
    
    return stat


def bestWindow(windows,dateTimeWin):
    winLen = len(windows)
    for ii in range(0,winLen):
        ts = pd.Series(windows[ii], index=dateTimeWin[ii]).dropna()
        tscumsum = ts.cumsum()
        tsdif = ts.diff()
        rollSTD = ts.rolling(30).std()
        
        ts.diff().plot()
        ts.rolling(2).std().plot()
        
        rollSTD = abs(ts.diff().dropna())
        plt.boxplot(abs(ts.diff().dropna()))
        
        plt.boxplot(rollSTD.dropna())

        rollSTD[rollSTD>np.nanpercentile(rollSTD,99)]
        rollSTD.plot()
        
        
        
        
        n = len(windows[ii])  # number of samples
        sigma = np.nanstd(windows[ii])
        model = "l2"  # "l1", "rbf", "linear", "normal", "ar"
        algo = rpt.Binseg(model=model).fit(np.array(windows[ii]))
        my_bkps = algo.predict(epsilon=3*n*sigma**2)
        my_bkps = algo.predict(n_bkps=20)
        
        # show results
        rpt.show.display(np.array(windows[ii]), [], my_bkps, figsize=(10, 6))
        plt.show()
        
        # show results
        rpt.show.display(np.array(windows[ii]), [], my_bkps, figsize=(10, 6))
        plt.show()
        
        algo = rpt.Pelt(model="l2")
        algo.fit(np.array(windows[ii]))
        result = algo.predict(pen=10)
        rpt.display(np.array(windows[ii]), [],result)
    
    return result #calibWind
    

 
def modelFit(windows,dateTimeWin):  
    winLen = len(windows)
    fig, ax = plt.subplots(winLen)
    checkModel=[]
    model_fit=[]
    winvar =[]
    for ii in range(0,winLen):
        winvar.append(np.nanvar(windows[ii]))
        data = pd.DataFrame()
        data['timeseries'] = windows[ii]
        data.index = pd.to_datetime(dateTimeWin[ii])
        data_filled = data.fillna(np.nanmean(windows[ii]))
        checkModel.append(tsa.stattools.adfuller(data_filled))

        auto_arima_model = auto_arima(y=data_filled,
                                      seasonal=True,
                                      m=4*24, #seasonality
                                      information_criterion="aic",
                                      trace=True)
        arima_model = sm.tsa.SARIMAX(data_filled[0:round(data_filled.shape[0]/2)], 
                                     order=auto_arima_model.order,
                                     seasonal_order = auto_arima_model.seasonal_order)
        model = arima_model.fit()
        model_fit.append(model.summary())
        model_forecast = model.forecast(data_filled.shape[0]-round(data_filled.shape[0]/2))
        fcast = model.get_forecast(data_filled.shape[0]-round(data_filled.shape[0]/2)).summary_frame()
        ax[ii].plot(data_filled.index,data_filled['timeseries'])
        ax[ii].plot(data_filled.index[round(data_filled.shape[0]/2):],
                    model.forecast(data_filled.shape[0]-round(data_filled.shape[0]/2)))
        ax[ii].fill_between(data_filled.index[round(data_filled.shape[0]/2):],fcast['mean_ci_lower'], fcast['mean_ci_upper'], color='k', alpha=0.1);    

    return checkModel,model_fit,model_forecast



#folder_path = '/media/leohoinaski/HDD/CLEAN_Calibration/data/2.input_equipo/dados_brutos'
#folder_path = '/mnt/sdb1/CLEAN_Calibration/data/2.input_equipo/dados_brutos'
folder_path="C:/Users/Leonardo.Hoinaski/Documents/CLEAN_Calibration/scripts/data/2.input_equipo/dados_brutos"
monitors = openMonitor(folder_path,'O3')
ave5min,ave15min, gaps = averages (monitors)
dataWin,dateTimeWin = selectWindow(ave15min,1)
stat = plotWindows(dataWin,dateTimeWin)
#checkModel,model_fit,yhat_conf_int = modelFit(dataWin,dateTimeWin)

# https://timeseriesreasoning.com/contents/correlation/
# https://www.iese.fraunhofer.de/blog/change-point-detection/
# https://facebook.github.io/prophet/docs/trend_changepoints.html#automatic-changepoint-detection-in-prophet
# https://zillow.github.io/luminaire/tutorial/dataprofiling.html

    
