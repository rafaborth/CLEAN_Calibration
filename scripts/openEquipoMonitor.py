# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 17:29:08 2023

@author: rafab
"""

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#%% JUNTANDO CSVS DE POLUENTES

folder_path = r'C:\Users\rafab\OneDrive\LCQAr\CLEAN_calibration\2.input_equipo\dados_quarter'

def read_multiple_csv_files(folder_path):
    dataframes = []  # cria lista pra colocar os dfs
    file_list = os.listdir(folder_path) #lista os arquivos do folder_path
    print(file_list) 

    for file_name in file_list:
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name) #cria o caminho de cada file quando esses forem csv
            print(f"Processing file: {file_path}")
            df = pd.read_csv(file_path) #lê só os arquivos que são csv dentro da lsita 
            df = transform_dataframe(df, file_name) #transforma os arquivos em dfs
            dataframes.append(df)
            #This line of code appends the DataFrame df to the dataframes list. In other words, it adds the DataFrame to the end of the list.
            #After the first iteration, dataframes will contain one DataFrame.
            #After the second iteration, dataframes will contain two DataFrames, and so on, for each CSV file processed.
    return pd.concat(dataframes, axis=1) #axis=1 especifica que a concatenação vai ser ao longo das colunas

def transform_dataframe(df, filename):
    new_columns = {col: f'{filename}_{col}' for col in df.columns}
    df.rename(columns=new_columns, inplace=True)
    return df

df = read_multiple_csv_files(folder_path)
df.rename(columns = {'CO.csv_DateTime':'DateTime'}, inplace=True)
df['DateTime'] = pd.to_datetime(df['DateTime'], format='mixed')
df.set_index('DateTime', inplace=True)

#%%
# renomeando colunas de concentração com base no nome dos poluentes
merge_df = df.filter(like='_measuring')
new_column_names = ['CO', 'PM25','PM10','NO2','O31','O32','SO21','SO22']
merge_df.columns = new_column_names

#merge_df.to_csv(r'C:\Users\rafab\OneDrive\LCQAr\CLEAN_calibration\3.output/merge_lcqar.csv') #salvou como arq excel

#%% SÉRIE TEMPORAL TODOS JUNTOS 

pos_values = merge_df.clip(lower=0) #tira os valores negativos de erro

#pos_values.to_csv(r'C:\Users\rafab\OneDrive\LCQAr\CLEAN_calibration\3.output/merge_lcqar_positivos.csv') #salvou como arq excel

columns_to_plot = pos_values.columns
plt.figure(figsize=(10, 6))

for column in columns_to_plot:
    plt.plot(pos_values[column], label=column, linewidth=0.8)

plt.xlabel('Tempo')
plt.ylabel('Concentração (ug/m3)')
plt.legend()
plt.show()

#%% SERIE TEMPORAL SEPARADO

sns.set(font_scale=0.8, style='whitegrid')

# Create individual line plots for each column as time series
for column in pos_values.columns:
    plt.figure(figsize=(8, 4))  # Adjust the figsize as needed
    sns.lineplot(x=pos_values.index, y=pos_values[column], linewidth=1.0)
    plt.title(column)
    plt.xlabel('Tempo (15 minutos)')
    plt.ylabel('Concentração (ug/m3)')
    plt.tight_layout()
    plt.show()

def plot_time_series(data, column_name):
    plt.figure(figsize=(8, 4))
    y = data[column_name]

    plt.plot(y, linewidth=1.0)
    plt.title(column_name)
    plt.xlabel('Tempo (15 minutos)')
    plt.ylabel('Concentração (ug/m3)')
    plt.tight_layout()
    plt.show()

sns.set(font_scale=0.8, style='whitegrid')

# Loop through each column and plot separately using the custom function
for column in pos_values.columns:
    plot_time_series(pos_values, column) #plota sem os valores negativos de erro

#%% BOX PLOT MENSAIS EQUIPAMENTO LCQAR

def plot_boxplots(file):
    sns.set(font_scale=0.8, style='whitegrid')

    # Extract year and month from the datetime index
    file['Year'] = file.index.year
    file['Month'] = file.index.month

    columns_to_plot = file.columns.tolist()  # Get a list of all column names

    #cria box plot para cada coluna do dataframe
    for column in columns_to_plot:
        # Create a facet grid of box plots, organized by year
        g = sns.catplot(
            data=file,
            x='Month',
            y=column,
            col='Year',
            kind='box',
            aspect=1.5,
            height=4,
            col_order=sorted(file['Year'].unique()),
            palette='crest', #paleta azul
            linewidth=0.5,
        )
        #Adicao de titulo 
        g.set_axis_labels('Month', 'Concentração (ug/m3)')
        g.set_titles('{col_name}')
        g.fig.suptitle(f'{column}', fontsize=14)

        plt.tight_layout()
        plt.show()

    # Remove the 'Year' and 'Month' columns added earlier
    file.drop(['Year', 'Month'], axis=1, inplace=True)

# Call the function to plot box plots for all columns in 'merge_df'
plot_boxplots(pos_values)

#%%






















































