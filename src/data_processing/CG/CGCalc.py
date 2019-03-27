import matplotlib.pyplot as plt
import numpy as np
from src.data_extraction.time_series_tool import TimeSeriesTool
from src.data_extraction import Data
from src.misc.NumericalTools import linear_spline

def get_cg(t):
    #Weight & Balance Values of Aircraft for CoG Calculations
    ZFW = 4852.17
    ZFW_arm = 34586.06
    init_fuel = 4050*0.453592
    xdata = list(reversed([4100,4000,3900,3800,3700,3600,3500,3400,3300,3200,3100,3000,2900,2800,2700,2600,2500,2400]))
    xdata = [x * 0.453592 for x in xdata]
    ydata = list(reversed([7.2517,7.25055,7.24942,7.2482,7.2471,7.246112,7.2451,7.2442,7.2433,7.2428,7.2423,7.2424,7.2425,7.24293,7.2433,7.244207,7.2451,7.24636]))

    #Time Data for CoG Shift
    t_start = 2573
    t_end = 2659

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
    data = Data('FlightData.mat')
    rhfu = data.get_mat().get_data()['rh_engine_FU'][idx]
    lhfu = data.get_mat().get_data()['lh_engine_FU'][idx]

    fu = rhfu+lhfu
    fuel = init_fuel-fu
    print(ZFW+fuel)

    x_fuel = linear_spline(fuel, xdata, ydata)

    # Calculate CoG for Passenger Shift
    if idx >= idx_start and idx <= idx_end:
        x_cg = (ZFW_arm + fuel * x_fuel - m_shift * (x_old - x_new)) / (ZFW + fuel)
        x_cg_LEMAC = (x_cg - x_LEMAC) / MAC
    else:
        x_cg = (ZFW_arm + fuel * x_fuel) / (ZFW + fuel)
        x_cg_LEMAC = (x_cg - x_LEMAC) / MAC


    return [x_cg,x_cg_LEMAC]

def cg_graph():
    x_cglst = []
    x_cg_LEMAClst = []
    timelst = []

    data = Data('FlightData.mat')
    time = data.get_mat().get_data()['time']
    x=0

    for i in range(0,len(time),10):
        ans = get_cg(time[i][0])
        x_cglst.append(ans[0])
        x_cg_LEMAClst.append(ans[1])
        timelst.append(time[i])
        x+=1
        if x%100 == 0:
            print(i/len(time))

    fig, ax1 = plt.subplots()
    ax1.plot(timelst, x_cg_LEMAClst, 'r-')
    ax1.set_xlabel('Time [s]')
    plt.xlim(0,max(time))
    plt.title('Center of Gravity Location')
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('$x_{CoG} [\%MAC]$', color='b')
    ax1.tick_params('y', colors='b')

    ax2 = ax1.twinx()
    ax2.plot(timelst, x_cglst, 'b-')
    ax2.set_ylabel('$x_{CoG,0} [m]$', color='r')
    ax2.tick_params('y', colors='r')

    fig.tight_layout()
    plt.show()

    return

# if __name__ == '__main__':
#     #cg_graph()