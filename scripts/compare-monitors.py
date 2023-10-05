# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 13:33:22 2023

@author: rafab
"""
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#%%
def openDiam(ref_path):

    file_list = os.listdir(ref_path) #lista os arquivos do folder_path
    print(file_list) 
    refMonitors = []
    for file_name in file_list:
        if file_name.endswith('.csv'):
            mon = pd.read_csv(ref_path+'/'+file_name)
            idx = pd.DatetimeIndex(mon['Data'])
            mon.set_index(idx, inplace=True)
            refMonitors.append(mon)
    
    refMonitors = pd.concat(refMonitors, axis=1)
     
    return refMonitors

def fixDatetime(refMonitors):
    df = pd.DataFrame({'year':refMonitors.index.year.values,
                   'month':refMonitors.index.month.values,
                    'day':refMonitors.index.day.values,
                    'hour':refMonitors.index.hour.values})
    refMonitors = refMonitors.set_index(pd.to_datetime(df[['year', 'month', 'day', 'hour']]))
    return refMonitors


ref_path = 'C:/Users/rafab/OneDrive/Documentos/CLEAN_Calibration/data/data_diamante'
refMonitors = openDiam(ref_path)
refMonitors = fixDatetime(refMonitors)

refMonMerge = refMonitors.merge(ave60min, left_index=True,right_index=True) #alterar codigo principal para unir em uma coluna dois sensores quando for o caso

#%% 

plt.plot(refMonMerge['timeseries'])
pol = 'NO2'
plt.plot(refMonMerge[pol])

plt.scatter(refMonMerge['timeseries'].values,np.nanmean(refMonMerge[pol],axis=1))
np.corrcoef(refMonMerge['timeseries'].values,np.nanmean(refMonMerge[pol],axis=1))

scipy.stats.spearmanr(refMonMerge['timeseries'].values,refMonMerge[pol].iloc[:,1], axis=0, nan_policy='omit')
    