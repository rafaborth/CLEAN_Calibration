# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 11:38:21 2023

@author: rafab
"""


import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#%% processa os dados de vila moema

folder_path = r'C:\Users\rafab\OneDrive\LCQAr\CLEAN_calibration\1.input_diamante'

def read_multiple_csv_files(folder_path):
    dataframes = []  # cria lista pra colocar os dfs
    file_list = os.listdir(folder_path) #lista os arquivos do folder_path
    print(file_list) 

    for file_name in file_list:
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name) #cria o caminho de cada file quando esses forem csv
            print(f"Processing file: {file_path}")
            df = pd.read_csv(file_path) #lê só os arquivos que são csv dentro da lsita 
            df = transform_dataframe(df) #transforma os arquivos em dfs
            dataframes.append(df)
            #This line of code appends the DataFrame df to the dataframes list. In other words, it adds the DataFrame to the end of the list.
            #After the first iteration, dataframes will contain one DataFrame.
            #After the second iteration, dataframes will contain two DataFrames, and so on, for each CSV file processed.
    return pd.concat(dataframes, ignore_index=True)

def transform_dataframe(df):
    return df

merge_df = read_multiple_csv_files(folder_path)
merge_df['Data'] = merge_df['Data'].str.replace(':00', '', regex=True)
merge_df['Data'] = pd.to_datetime(merge_df['Data'], format='%d/%m/%Y %H:%M')
merge_df.index = merge_df['Data']
merge_df.drop(columns='Data', inplace=True)
print(merge_df.head()) #Retorna a lista de DataFrames após o loop ter processado todos os arquivos no diretório.

#%%  SERIE TEMPORAL DIAMANTE - TODOS

merge_df['CO'] = merge_df['CO'] * 1150 #transforma CO em ug/m3 

columns_to_plot = merge_df.columns
plt.figure(figsize=(10, 6))

for column in columns_to_plot:
    plt.plot(merge_df[column], label=column)

plt.xlabel('Tempo')
plt.ylabel('Concentração (ug/m3)')
plt.legend()
plt.show()

#%% SERIE TEMPORAL DIAMANTE - SEPARADO

sns.set(font_scale=0.8, style='whitegrid')

# Create individual line plots for each column as time series
for column in merge_df.columns:
    plt.figure(figsize=(8, 4))  # Adjust the figsize as needed
    sns.lineplot(x=merge_df.index, y=merge_df[column], linewidth=1.0)
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
for column in merge_df.columns:
    plot_time_series(merge_df, column) #plota sem os valores negativos de erro


#%% BOX-PLOT DIAMANTE ANUAIS POR POLUENTE

sns.set(font_scale=0.8, style='whitegrid')
# Create a figure and axis for the subplots
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(25, 10), 
                         sharex=False, sharey=False,  # Ensure no shared axes
                         gridspec_kw={'wspace': 0.2, 'hspace': 0.5})  # Use a 3x3 grid for 9 columns

# Create box plots for each column
for i, column in enumerate(merge_df.columns):
    row = i // 3  # Determine the row for the subplot
    col = i % 3   # Determine the column for the subplot
    
    # Extract year and month from the datetime index
    merge_df['Year'] = merge_df.index.year
    merge_df['Month'] = merge_df.index.month
    
    # Create the box plot using the extracted year and the current column
    sns.boxplot(x='Year', y=column, data=merge_df, ax=axes[row, col], palette='crest', linewidth=0.5)
    
    # Set the title and labels
    axes[row, col].set_title(column)
    axes[row, col].set_xlabel('Year')
    axes[row, col].set_ylabel('Concentração (ug/m3)')

# Remove the 'Year' and 'Month' columns added during the loop
merge_df.drop(['Year', 'Month'], axis=1, inplace=True)

plt.tight_layout()
plt.show()

#%% BOX PLOT MENSAIS POR POLUENTE
def plot_boxplots(file):
    sns.set(font_scale=0.8, style='whitegrid')

    # Extract year and month from the datetime index
    file['Year'] = file.index.year
    file['Month'] = file.index.month

    columns_to_plot = file.columns.tolist()  # Get a list of all column names

    # Create box plots for each column
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
            palette='crest',  # Use a single color palette
            linewidth=0.5,
        )

        # Set titles for each facet
        g.set_axis_labels('Month', 'Concentração (ug/m3)')
        g.set_titles('{col_name}')
        g.fig.suptitle(f'{column}', fontsize=14)

        plt.tight_layout()
        plt.show()

    # Remove the 'Year' and 'Month' columns added earlier
    file.drop(['Year', 'Month'], axis=1, inplace=True)

# Call the function to plot box plots for all columns in 'merge_df'
plot_boxplots(merge_df)

#%%
#merge_df.to_csv(r'C:\Users\rafab\OneDrive\LCQAr\CLEAN_calibration\3.output/merge_diamante.csv') #salvou como arq excel



























