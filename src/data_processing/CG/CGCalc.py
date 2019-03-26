import matplotlib.pyplot as plt
import numpy as np
from src.data_extraction.time_series_tool import TimeSeriesTool
from src.data_extraction import Data

def get_cg(t):
    #Weight & Balance Values of Aircraft for CoG Calculations
    ZFW = 4852.92
    ZFW_arm = 34586.06
    init_fuel = 1837.05
    x_fuel = 7.253

    #Time Data for CoG Shift
    t_start = 2573
    t_end = 2660

    #CoG Shift Data
    x_old = 7.315
    x_new = 3.3
    m_shift = 87

    #Aircraft Geometry
    x_LEMAC = 6.643624
    MAC = 2.056892

    #Idx Location of CoG Shift
    ts_tool = TimeSeriesTool()
    idx = ts_tool.get_mdat_tstep_list_idx_for_matching_pdat_tstep(t)
    idx_start = ts_tool.get_mdat_tstep_list_idx_for_matching_pdat_tstep(t_start)
    idx_end = ts_tool.get_mdat_tstep_list_idx_for_matching_pdat_tstep(t_end)

    #Get Fuel Use Data
    data = Data('RefData.mat')

    time = data.get_mat().get_data()['time']
    rhfu = data.get_mat().get_data()['rh_engine_FU']
    lhfu = data.get_mat().get_data()['lh_engine_FU']

    fu = rhfu+lhfu
    fuel = init_fuel-fu

    x_cg = (ZFW_arm+fuel*x_fuel)/(ZFW+fuel)

    #Calculate CoG for Passenger Shift
    for i in range(idx_start,idx_end):
        x_cg[i] = (ZFW_arm+fuel[idx_start]*x_fuel-m_shift*(x_old-x_new))/(ZFW+fuel[idx_start])

    x_cg_LEMAC = (x_cg-x_LEMAC)/MAC

    # fig, ax1 = plt.subplots()
    # ax1.plot(time, x_cg_LEMAC, 'b-')
    # ax1.set_xlabel('Time [s]')
    # plt.xlim(0,max(time))
    # plt.title('Center of Gravity Location')
    # # Make the y-axis label, ticks and tick labels match the line color.
    # ax1.set_ylabel('$x_{CoG} [\%MAC]$', color='b')
    # ax1.tick_params('y', colors='b')
    #
    # ax2 = ax1.twinx()
    # ax2.plot(time, x_cg, 'b-')
    # ax2.set_ylabel('$x_{CoG,0} [m]$', color='r')
    # ax2.tick_params('y', colors='r')
    #
    # fig.tight_layout()
    # plt.show()

    return [x_cg[idx],x_cg_LEMAC[idx]]
