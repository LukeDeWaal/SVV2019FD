# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 17:10:31 2019
@author: chris
"""

import sys, string, os, subprocess
import pandas as pd
import numpy as np
from src.data_extraction.mat_import import *

datafile_path = get_data_file_path(r'FlightData.mat')
MatFile = MatFileImport(datafile_path)
keys = MatFile.get_keys()
descr = MatFile.get_descriptions()
data = MatFile.get_data()
units = MatFile.get_units()

curr_dir = os.path.dirname(os.path.realpath(__file__))
file_path_def = os.path.join(curr_dir, 'thrust.dat')

# create data columns for manipulation
matlab_data = pd.DataFrame()
matlab_data['h'] = [i[0] for i in list(data['Dadc1_alt'])]
matlab_data['M'] = [i[0] for i in list(data['Dadc1_mach'])]
matlab_data['SAT'] = [i[0] for i in list(data['Dadc1_sat'])]
matlab_data['dT'] = matlab_data['SAT']-(288.15-(0.0098*matlab_data['h']))
matlab_data['FFl'] = [i[0] for i in list(data['lh_engine_FMF'])]
matlab_data['FFr'] = [i[0] for i in list(data['rh_engine_FMF'])]
matlab_data.drop(['SAT'],axis=1,inplace=True)
matlab_data = matlab_data.round(4)

# the pressure altitude,  the Mach number,  the temperature difference, fuel flow left jet engine, fuel flow right jet engine

matlab_data.to_csv("..\external_sources\matlab.dat".replace("\\", "/"), sep=' ', header=False, index=False)

#thrust_data = pd.read_csv(r"..\external_sources\thrust.dat".replace("\\", "/"), sep=' ')