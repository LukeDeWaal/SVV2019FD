# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 17:10:31 2019

@author: chris
"""

import sys,string,os,subprocess
import pandas as pd
import numpy as np

curr_dir = os.path.dirname(os.path.realpath(__file__))
file_path_def = os.path.join(curr_dir, 'thrust.dat')

#the pressure altitude,  the Mach number,  the temperature difference, fuel flow left jet engine, fuel flow right jet engine
matlab_file = np.array([[9848, 0.7, 3, 0.18, 0.18],
                        [9880, 0.7, 2, 0.18, 0.18],
                        ])
matlab_file = pd.DataFrame(matlab_file)
matlab_file.to_csv("..\external_sources\matlab.dat".replace("\\","/"),sep=' ', header=False, index=False)

df = pd.read_csv(r"..\external_sources\thrust.dat".replace("\\","/"),sep=' ')

# =============================================================================
# "C:\Users\chris\Dropbox\Uni Delft\Year 3\SVV\SVV2019FD\src\external_sources"
# =============================================================================
