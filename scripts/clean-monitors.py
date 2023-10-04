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
from sklearn.neighbors import KernelDensity
from statsmodels.nonparametric.bandwidths import bw_silverman
from sklearn.mixture import GaussianMixture
from scipy.stats import norm

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


def kde_sklearn(x, x_grid, bandwidth=1, **kwargs):
    """Kernel Density Estimation with Scikit-learn"""
    kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
    kde_skl.fit(x[:, np.newaxis])
    # score_samples() returns the log-likelihood of the samples
    log_pdf = kde_skl.score_samples(x_grid[:, np.newaxis])
    return np.exp(log_pdf)


def getExtremePoints(data, typeOfExtreme = None, maxPoints = None):
    """
    This method returns the indeces where there is a change in the trend of the input series.
    typeOfExtreme = None returns all extreme points, max only maximum values and min
    only min,
    """
    a = np.diff(data)
    asign = np.sign(a)
    signchange = ((np.roll(asign, 1) - asign) != 0).astype(int)
    idx = np.where(signchange ==1)[0]
    if typeOfExtreme == 'max' and data[idx[0]] < data[idx[1]]:
        idx = idx[1:][::2]
        
    elif typeOfExtreme == 'min' and data[idx[0]] > data[idx[1]]:
        idx = idx[1:][::2]
    elif typeOfExtreme is not None:
        idx = idx[::2]
    
    # sort ids by min value
    if 0 in idx:
        idx = np.delete(idx, 0)
    if (len(data)-1) in idx:
        idx = np.delete(idx, len(data)-1)
    idx = idx[np.argsort(data[idx])]
    # If we have maxpoints we want to make sure the timeseries has a cutpoint
    # in each segment, not all on a small interval
    if maxPoints is not None:
        idx= idx[:maxPoints]
        if len(idx) < maxPoints:
            return (np.arange(maxPoints) + 1) * (len(data)//(maxPoints + 1))
    
    return idx


def getPeaks(ts):
    # Figures - detecting peaks
    fig, ax = plt.subplots()
    hh=ax.hist(ts['timeseries'].values)
    x_grid = np.linspace(hh[1].min(), hh[1].max(), 1000)
    silverman_bandwidth = bw_silverman(ts['timeseries'].dropna().values)
    pdf = kde_sklearn(np.array(ts['timeseries'].dropna()),x_grid, bandwidth=silverman_bandwidth*15)
    ax2 = ax.twinx()
    # Geting peaks
    idx = getExtremePoints(pdf, typeOfExtreme = 'max', maxPoints = None)
    ax2.plot(x_grid, pdf, color='blue', alpha=0.5, lw=3)
    ax2.scatter(x_grid[idx], pdf[idx], color='cyan', alpha=0.3)
    # Decomposing pdfs
    gmm = GaussianMixture(n_components=idx.shape[0])
    gmm.fit(np.array(ts['timeseries'].dropna()).reshape(-1, 1))
    means = gmm.means_
    # Conver covariance into Standard Deviation
    standard_deviations = gmm.covariances_.reshape(-1,1)**0.5  
    # Useful when plotting the distributions later
    weights = gmm.weights_  
    peaks = pd.DataFrame()
    peaks['means'] = pd.DataFrame(means)
    peaks['stds'] = standard_deviations
    peaks['weights'] = pd.DataFrame(weights)
    
    # PDF PLOT
    fig, axes = plt.subplots()
    axes.hist(ts['timeseries'].dropna(), bins=50, alpha=0.5)
    x = np.linspace(min(np.array(ts['timeseries'].dropna())), 
                    max(np.array(ts['timeseries'].dropna())), 100)
    ii=0
    axes2 = axes.twinx()
    pdfs=[]
    for mean, std, weight in zip(means, standard_deviations, weights):
         pdf = weight*norm.pdf(x, mean, std)
         axes2.plot(x.reshape(-1, 1), pdf.reshape(-1, 1), alpha=0.5)
         pdfs.append(pdf)           
         ii=ii+1
       
    return peaks

def multi2unimodal(windows,dateTimeWin):
    # Geting local information
    bestSignal,allPeaks,bestPeak = bestWindow(windows,dateTimeWin)
    # Standardizing not reference data
    correctTs=[]
    correctDt=[]
    for ii,winD in enumerate(windows):
        winD = np.array(winD)
        for jj in range(0,allPeaks[ii].shape[0]):
            lowLocal = allPeaks[ii]['means'][jj]-5*allPeaks[ii]['stds'][jj]
            upLocal = allPeaks[ii]['means'][jj]+5*allPeaks[ii]['stds'][jj]
            print(str(lowLocal)+' - '+ str(upLocal))
            # winD[(winD>lowLocal) & (winD<upLocal)] = \
            #     (winD[(winD>lowLocal) & (winD<upLocal)]- allPeaks[ii]['means'][jj])/allPeaks[ii]['stds'][jj]
            correctTs.append((winD[(winD>lowLocal) & (winD<upLocal)]- allPeaks[ii]['means'][jj])/allPeaks[ii]['stds'][jj])   
            correctDt.append(pd.DataFrame(np.array(dateTimeWin[ii])[(winD>lowLocal) & (winD<upLocal)]))
        
    # Correcting using best signal as reference
    stdData=[]
    for ii,cts in enumerate(correctTs):
        cts = np.array(cts)
        cts = (cts + bestPeak['means'][0])*bestPeak['stds'][0]
        stdData.append(pd.DataFrame(cts))
        
    stdData= pd.concat(stdData)
    stdData.columns=['timeseries']
    correctDt= pd.concat(correctDt)
    
    stdData['datetime'] = np.array(correctDt)
    stdData = stdData.set_index('datetime')
    stdData = stdData.drop_duplicates()
    stdData = stdData.sort_index()

    return stdData


def bestWindow(windows,dateTimeWin):
    winLen = len(windows)
    allPeaks=[]
    for ii in range(0,winLen):
        ts = pd.DataFrame()
        ts['timeseries'] = windows[ii]
        ts['datetime'] = dateTimeWin[ii]
        ts = ts.set_index(pd.DatetimeIndex(ts['datetime']))
        ts = ts.dropna()
        peaks = getPeaks(ts)
        allPeaks.append(peaks)
        # tscumsum = ts['timeseries'].cumsum()
        # tsdif = ts['timeseries'].diff()
        # rollSTD = ts['timeseries'].rolling(30).std()
        # rollMean = ts['timeseries'].rolling(20).mean()
        # fig, ax = plt.subplots(4)
        # ax[0].plot(tsdif.index,tsdif,color='orange')
        # ax[0].scatter(tsdif[tsdif>np.nanpercentile(tsdif,99)].index,
        #         tsdif[tsdif>np.nanpercentile(tsdif,99)],color='green')
        # ax[0].scatter(tsdif[tsdif<np.nanpercentile(tsdif,1)].index,
        #         tsdif[tsdif<np.nanpercentile(tsdif,1)],color='green')
        # ax[1].plot(rollSTD.index,rollSTD,color='red')
        # ax[1].scatter(rollSTD[rollSTD>np.nanpercentile(rollSTD,99)].index,
        #         rollSTD[rollSTD>np.nanpercentile(rollSTD,99)],color='green')
        # ax[2].plot(rollMean.index,
        #         rollMean,color='black')
        # # ax[2].scatter(rollMean[rollMean<np.nanpercentile(rollMean,1)].index,
        # #         rollMean[rollMean<np.nanpercentile(rollMean,1)],color='green')
        # # ax[2].scatter(rollMean[rollMean>np.nanpercentile(rollMean,99)].index,
        # #         rollMean[rollMean>np.nanpercentile(rollMean,99)],color='green')
        # ax[3].plot(ts.index,ts['timeseries'],color='blue')
    
    minPeaks=[]
    d2 = np.nan

    for allp in allPeaks:
        minPeaks.append(allp.shape[0])
    minPeaks = np.min(minPeaks)
    for idx,allp in enumerate(allPeaks):
        if allp.shape[0] == minPeaks:
            d1 = np.max(allp.stds)
            if d1!=d2:
                d2=d1
                windowsId = idx
  
        
    bestSignal = pd.DataFrame()
    bestSignal['timeseries'] = windows[windowsId]
    bestSignal['datetime'] = dateTimeWin[windowsId]
    bestSignal = bestSignal.set_index(pd.DatetimeIndex(bestSignal['datetime']))
    bestPeak = getPeaks(bestSignal)
    
    return  bestSignal,allPeaks,bestPeak
    

 
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



folder_path = '/media/leohoinaski/HDD/CLEAN_Calibration/data/2.input_equipo/dados_brutos'
#folder_path = '/mnt/sdb1/CLEAN_Calibration/data/2.input_equipo/dados_brutos'
#folder_path="C:/Users/Leonardo.Hoinaski/Documents/CLEAN_Calibration/scripts/data/2.input_equipo/dados_brutos"
monitors = openMonitor(folder_path,'O3')
ave5min,ave15min, gaps = averages (monitors)
dataWin,dateTimeWin = selectWindow(ave15min,1)
stat = plotWindows(dataWin,dateTimeWin)
stdData = multi2unimodal(dataWin,dateTimeWin)
#stat = plotWindows(stdData.timeseries,stdData.index)
stdData.plot()
#checkModel,model_fit,yhat_conf_int = modelFit(dataWin,dateTimeWin)

# https://timeseriesreasoning.com/contents/correlation/
# https://www.iese.fraunhofer.de/blog/change-point-detection/
# https://facebook.github.io/prophet/docs/trend_changepoints.html#automatic-changepoint-detection-in-prophet
# https://zillow.github.io/luminaire/tutorial/dataprofiling.html

    
