# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 17:10:31 2019

@author: chris
"""

import sys,string,os
import pandas as pd

curr_dir = os.path.dirname(os.path.realpath(__file__))
file_path_def = os.path.join(curr_dir, 'thrust.dat')


matlab_file = pd.to_csv(r"..\external_sources\matlab.dat".replace("\\","/"),sep=' ')


os.system(r"..\external_sources\thrust.exe".replace("\\","/"),sep=' ')


df = pd.read_csv(r"..\external_sources\thrust.dat".replace("\\","/"),sep=' ')

# =============================================================================
# "C:\Users\chris\Dropbox\Uni Delft\Year 3\SVV\SVV2019FD\src\external_sources"
# =============================================================================
