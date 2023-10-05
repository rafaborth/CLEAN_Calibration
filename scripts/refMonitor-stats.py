# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 13:49:55 2023

@author: rafab
"""

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import autocorrelation_plot


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
    refMonitors.index.name = 'Datetime'
    refMonitors.drop(['Data', 'Data'], axis=1, inplace=True)
    return refMonitors

#%%

def merge_columns_with_nan(columns):
    merged_column = pd.Series(columns.iloc[:,0])
    merged_column.drop(columns=['Month'], inplace=True)
    for i in range(1, columns.shape[1]):
        merged_column = merged_column.combine_first(columns.iloc[:, i])
    return merged_column


def stats_diam(referenceMonitor, year):
    for column_name in referenceMonitor.columns:
        column_data = referenceMonitor[column_name][referenceMonitor.index.year == year]
        
        sns.set(font_scale=0.8, style='whitegrid')
        fig, ax = plt.subplots(3, figsize=(12, 8), gridspec_kw={'wspace': 0.2, 'hspace': 0.5})
        fig.suptitle(f'Análise de - {column_name}', fontsize=10)

        ax[0].plot(column_data.index, column_data.values, lw=1.0)
        ax[0].set_xlabel('Data', fontsize=8)
        ax[0].set_ylabel('Concentração', fontsize=8)
                
        ax[1].hist(column_data.dropna(), bins=30)
        ax[1].set_xlabel('Concentração', fontsize=8)
        ax[1].set_ylabel('Frequencia', fontsize=8)
        
        monthly_boxplot_data = []
        for month in range(1, 13):
            ref_monthly = referenceMonitor[(referenceMonitor.index.year == 2023) & (referenceMonitor.index.month == month)]
            data_to_plot = ref_monthly[column_name]
            monthly_boxplot_data.append(data_to_plot)

            ax[2].boxplot(monthly_boxplot_data, vert=True, widths=0.6, showfliers=True, patch_artist=True, 
                          boxprops={'linewidth': 0.5, 'color': 'blue'}, capprops={'linewidth': 0.5, 'color': 'blue'}, 
                          medianprops={'linewidth': 0.5, 'color': 'black'})
            ax[2].set_xlabel('Meses', fontsize=8)
            ax[2].set_ylabel('Concentração', fontsize=8)
        plt.show()
    
    return referenceMonitor




ref_path = 'C:/Users/rafab/OneDrive/Documentos/CLEAN_Calibration/data/data_diamante'
refMonitors = openDiam(ref_path)
refMonitors = fixDatetime(refMonitors)
merged_ref = refMonitors.groupby(refMonitors.columns, axis=1).agg(merge_columns_with_nan)
specific_year = 2023
stats_diam(merged_ref, specific_year) 
