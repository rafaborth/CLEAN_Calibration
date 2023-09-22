# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 19:15:23 2023

@author: rafab
"""

from openEquipoMonitor import pos_values
from openRefMonitor import merge_df

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#%%
pos_values = pd.read_csv(r'C:\Users\rafab\OneDrive\LCQAr\CLEAN_calibration\3.output/merge_lcqar_positivos.csv')
pos_values['DateTime'] = pd.to_datetime(pos_values['DateTime'], format='mixed')
pos_values.index = pos_values['DateTime']
pos_values.drop(columns=['DateTime'], inplace=True)

merge_df = pd.read_csv(r'C:\Users\rafab\OneDrive\LCQAr\CLEAN_calibration\3.output/merge_diamante.csv')
merge_df.rename(columns = {'Data':'DateTime'}, inplace=True)
merge_df['DateTime'] = pd.to_datetime(merge_df['DateTime'], format='mixed')
merge_df.index = merge_df['DateTime']
merge_df.drop(columns=['DateTime'], inplace=True)

#%% SÉRIE TEMPORAL PARA CADA POLUENTE - COMPARAÇÃO ESTAÇÕES

def plot_columns(df1, df2, year_to_plot):
    # Get a list of common column names between the two DataFrames
    common_columns = list(set(df1.columns) & set(df2.columns))

    # Set the style, font scale, and background color
    sns.set(style='whitegrid', font_scale=1.2, rc={'axes.facecolor': 'white'})

    # Filter the DataFrames for the specified year
    df1 = df1[df1.index.year == year_to_plot]
    df2 = df2[df2.index.year == year_to_plot]

    for column in common_columns:
        plt.figure(figsize=(12, 6))  # Adjust figure size as needed
        sns.lineplot(data=df1, x=df1.index, y=column, label='LCQAr')
        sns.lineplot(data=df2, x=df2.index, y=column, label='Diamante')

        plt.xlabel('Tempo')
        plt.ylabel('Concentração')
        plt.title(f'Série temporal - {column}')
        plt.legend()
        plt.tight_layout()
        plt.show()

plot_columns(pos_values, merge_df, year_to_plot=2023)

#%% pra O3 e SO2 que tem dois sensores no equipamento, faz gráficos separados
plt.figure(figsize=(12, 6))  # Adjust figure size as needed

sns.lineplot(data=pos_values, x=pos_values.index, y='O31', label='LCQAr sensor 1')
sns.lineplot(data=pos_values, x=pos_values.index, y='O32', label='LCQAr sensor 2')
sns.lineplot(data=merge_df, x=merge_df.index, y='O3', label='Diamante')

plt.xlabel('Tempo')
plt.ylabel('Concentração')
plt.title(f'Série temporal - O3')
plt.legend()
plt.tight_layout()
plt.show()

#%% pra O3 e SO2 que tem dois sensores no equipamento, faz gráficos separados
plt.figure(figsize=(12, 6))  # Adjust figure size as needed

sns.lineplot(data=pos_values, x=pos_values.index, y='SO21', label='LCQAr sensor 1')
sns.lineplot(data=pos_values, x=pos_values.index, y='SO22', label='LCQAr sensor 2')
sns.lineplot(data=merge_df, x=merge_df.index, y='SOX', label='Diamante')

plt.xlabel('Tempo')
plt.ylabel('Concentração')
plt.title(f'Série temporal - SO2')
plt.legend()
plt.tight_layout()
plt.show()

#%% BOX PLOTS POR MES COMPARAÇÃO DIAMANTE VS EQUIPO LCQAR

def compare_boxplots(df1, df2):
    sns.set(font_scale=0.5, style='whitegrid')

    # Extract column names
    columns = df1.columns

    # Iterate through columns
    for column in columns:
        plt.figure(figsize=(10, 6))

        # Iterate through months
        for month in range(1, 13):
            # Filter data for the current month
            df1_monthly = df1[df1.index.month == month]
            df2_monthly = df2[df2.index.month == month]

            # Combine the data for the current month and column
            data_to_plot = pd.concat([df1_monthly[column], df2_monthly[column]], axis=1)
            data_to_plot.columns = ['LCQAr', 'Diamante']

            # Create a box plot for the current month and column
            plt.suptitle(f'Box Plots mensais - {column}', fontsize=14)
            plt.subplot(3, 4, month)  # Create a subplot for each month
            sns.boxplot(data=data_to_plot, palette='crest', linewidth=0.5)
            plt.xlabel(f'Mês {month}')
            plt.ylabel('Concentração')
            plt.tight_layout()

        plt.subplots_adjust(top=0.85)  # Adjust the title position
        plt.show()

# Call the function to compare box plots for each column and each month
compare_boxplots(pos_values, merge_df)

#%%