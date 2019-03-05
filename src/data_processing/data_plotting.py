import numpy as np
import matplotlib.pyplot as plt
from src.data_extraction.mat_import import MatFileImport

M = MatFileImport('C:/Users/Luke/Downloads/test1.mat')

keys = M.get_keys()
descr = M.get_descriptions()
data = M.get_data()

for i, key in enumerate(keys[:-1]):
    fig = plt.figure(i)
    plt.plot(data['time'], data[key])
    plt.grid(True)
    plt.xlabel('time')
    plt.ylabel(key)
