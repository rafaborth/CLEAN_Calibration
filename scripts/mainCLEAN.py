#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 12:26:55 2023

@author: leohoinaski
"""
import clean-monitors as cl

#folder_path = '/mnt/sdb1/CLEAN_Calibration/data/2.input_equipo/dados_brutos'
folder_path = '/media/leohoinaski/HDD/CLEAN_Calibration/data/2.input_equipo/dados_brutos'

pollutant = 'O3'

merge_df = opCLEAN.read_multiple_csv_files(folder_path)