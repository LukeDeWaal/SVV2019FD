import numpy as np


def get_weight_at_t(t, time, rh_FU, lh_FU, W0=60500.0):
    """
    Get weight of plane at time t
    :param t: desired time
    :param time: time array from the .mat file
    :param rh_FU: right hand fuel usage array in kg
    :param lh_FU: left hand fuel usage array in kg
    :param W0: Starting weight in newtons
    :return: Current weight in N
    """

    i = 0
    while True:
        if t >= time[i] and t < time[i+1]:
            total_fuel_usage = rh_FU[i] + lh_FU[i]
            current_weight = W0 - total_fuel_usage*9.80665
            return current_weight
        else:
            i += 1
            continue


def get_weight(time, rh_FU, lh_FU, W0=60500.0):
    """
    Get weight at all times in the time array
    :param time: time array
    :param rh_FU: right hand fuel usage array in kg
    :param lh_FU: left hand fuel usage array in kg
    :param W0: Starting weight in newtons
    :return: Weight in N at time[i]
    """

    W = []

    for i in range(len(time)):
        W.append(W0 - (rh_FU[i] + lh_FU[i])*9.80665)

    return np.array(W).reshape((len(W), 1))
