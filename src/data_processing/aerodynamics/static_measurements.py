import numpy as np
from src.data_extraction import Data
import matplotlib.pyplot as plt


def plot_trim_curve(data_object):

    data_elev = data_object.get_pfd('StatElev.csv')
    data_cg = data_object.get_pfd('GravShift.csv')

    fig = plt.figure()
    plt.plot(data_elev['TAS'], data_elev['de'], 'rx')
    plt.plot(data_cg['TAS'], data_cg['de'], 'bx')
    plt.plot()
    plt.grid()
    plt.xlabel('TAS [m/s]')
    plt.ylabel('$\delta_e$ [deg]')



if __name__ == "__main__":

    data = Data(r'FlightData.mat', 'StatClCd.csv', 'StatElev.csv', 'GravShift.csv')
    plot_trim_curve(data)
