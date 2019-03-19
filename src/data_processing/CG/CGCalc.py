import numpy as np
import matplotlib.pyplot as plt

ZFW = 4887.17
ZFW_arm = 34806.65
init_fuel = 1837.05
x_fuel = 7.253

data = Data('FlightData.mat')

time = data.get_mat().get_data()['time']
rhfu = data.get_mat().get_data()['rh_engine_FU']
lhfu = data.get_mat().get_data()['lh_engine_FU']

fu = rhfu+lhfu
fuel = init_fuel-fu

x_cg = (ZFW_arm+fuel*x_fuel)/(ZFW+fuel)
