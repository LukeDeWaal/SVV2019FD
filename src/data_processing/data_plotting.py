import numpy as np
import matplotlib.pyplot as plt
from src.data_processing.mat_import import MatFileImport

M = MatFileImport('C:/Users/Luke/Downloads/test1.mat')

keys = M.get_keys()
descr = M.get_descriptions()
data = M.get_data()


plt.plot(data['time'], data['elevator_dte'])