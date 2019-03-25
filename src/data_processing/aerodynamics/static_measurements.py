import numpy as np
import pandas as pd
from src.data_extraction import Data
import matplotlib.pyplot as plt
from src.data_extraction.get_thrust_data import get_thrust
from src.data_processing.aerodynamics.aerodynamic_coefficients import ISA
from src.data_processing.CG.CGCalc import get_cg
from src.data_processing.get_weight import get_weight, get_weight_at_t
from src.misc.NumericalTools import deg_2_rad, quadratic_least_squares, linear_least_squares, derive

cmtc = -0.0064

#%% Constants for Wing
S = 30.0
b = 15.911
c = 2.0569


def cm_delta(Cn, delta_cg, delta_delta_e, c):
    return -(Cn*delta_cg)/(delta_delta_e*c)


def cn(h, v, w=60500.0, S=30.0):
    _, _, rho = ISA(h)
    return w/(0.5*rho*S*v**2)


def cm_alpha(cm_delta, delta_delta_e, delta_alpha):
    return -cm_delta*(delta_delta_e/delta_alpha)


def get_cm_delta(data_object, t=(2560, 2610)):

    data_1 = data_object.get_pfd('StatElev.csv')
    data_2 = data_object.get_pfd('GravShift.csv')
    matdata = data_object.get_mat().get_data()

    times = np.concatenate([data_1['time'], data_2['time']], axis=0)[-2:]

    Ve = np.concatenate([data_1['IAS'], data_2['IAS']], axis=0)[-2:]
    Vt = np.concatenate([data_1['TAS'], data_2['TAS']], axis=0)[-2:]

    h = np.concatenate([data_1['hp'], data_2['hp']])[-2:]

    delta_e = deg_2_rad(np.concatenate([data_1['de'], data_2['de']], axis=0)[-2:])
    delta_delta_e = delta_e[1] - delta_e[0]

    cgs = [get_cg(time)[0][0] for time in t]
    delta_cg = cgs[1] - cgs[0]

    Cn = cn(h[0], Vt[0], get_weight_at_t(t[0], matdata['time'], matdata['rh_engine_FU'], matdata['lh_engine_FU']))[0]

    cmdelta = cm_delta(Cn, delta_cg, delta_delta_e, c)

    return cmdelta


def get_cm_alpha(data_object):

    cm_delta = get_cm_delta(data_object)

    data_1 = data_object.get_pfd('StatElev.csv')
    data_2 = data_object.get_pfd('GravShift.csv')

    alpha = deg_2_rad(np.concatenate([data_1['a'], data_2['a']], axis=0)[-2:])
    delta_alpha = alpha[1] - alpha[0]

    delta_e = deg_2_rad(np.concatenate([data_1['de'], data_2['de']], axis=0)[-2:])
    delta_delta_e = delta_e[1] - delta_e[0]

    return -cm_delta*(delta_delta_e/delta_alpha)


def thrust_coefficient(T, Ve, rho, d=0.6858):
    if type(T) == list or type(T) == np.array or type(T) == pd.Series:
        if type(rho) == list or type(rho) == np.array or type(rho) == pd.Series:
            return [T_i / (0.5 * rho_i * (v_i ** 2) * (d ** 2)) for T_i, v_i, rho_i in zip(T, Ve, rho)]
        else:
            return [T_i/(0.5*rho*v_i**2*d**2) for T_i, v_i in zip(T, Ve)]

    else:
        return T/(0.5*rho*Ve**2*(2*d)**2)


def trimcurve_fit(ve, de):

    a, b, c = quadratic_least_squares(ve, de)

    def fit(ve):
        return a*ve**2+b*ve+c

    return fit

def delta_alpha_fit(alpha, de):

    a, b = linear_least_squares(alpha, de)

    def fit(alpha):
        return a*alpha + b

    return fit

def stickforce_fit(ve, Fe):

    a, b, c = quadratic_least_squares(ve, Fe)

    def fit(ve):
        return a*ve**2 + b*ve + c

    return fit

def plot_trim_curve(data_object):

    data_1 = data_object.get_pfd('StatElev.csv')
    data_2 = data_object.get_pfd('GravShift.csv')
    matdata = data_object.get_mat().get_data()

    times = np.concatenate([data_1['time'], data_2['time']], axis=0)

    cmdelta = get_cm_delta(data_object)
    delta_e = deg_2_rad(np.concatenate([data_1['de'], data_2['de']], axis=0))

    Ve = np.concatenate([data_1['IAS'], data_2['IAS']], axis=0)
    Vt = np.concatenate([data_1['TAS'], data_2['TAS']], axis=0)

    W0 = 60500.0
    Weight = [get_weight_at_t(t, matdata['time'], matdata['rh_engine_FU'], matdata['lh_engine_FU'],W0=W0) for t in times]

    Ve_tilde = [ve * np.sqrt(W0 / w) for ve, w in zip(Ve, Weight)]

    h = np.concatenate([data_1['hp'], data_2['hp']])

    Ts = np.sum(get_thrust('thrust_stand.dat'), axis=1)
    T = np.sum(get_thrust('thrust.dat'), axis=1)

    Tcs = thrust_coefficient(Ts, Vt, rho=1.225)
    Tc = thrust_coefficient(T, Vt, rho=[ISA(h_i)[2] for h_i in h])

    delta_e_eq_star = [de - cmtc / cmdelta * (tcs - tc) for de, tcs, tc in zip(delta_e, Tcs, Tc)]

    bestfit_ve = trimcurve_fit(Ve_tilde[:-2], delta_e_eq_star[:-2])

    ve_tilde_range = np.linspace(0.9*min(Ve_tilde), 1.1*max(Ve_tilde), 100)

    fig1 = plt.figure()
    plt.plot(Ve_tilde[:-2], delta_e_eq_star[:-2], 'rx', label='Measured Data')
    plt.plot(ve_tilde_range, bestfit_ve(ve_tilde_range), 'b-', label='Best Fit')
    plt.ylim(max(delta_e_eq_star), min(delta_e_eq_star))
    plt.grid()
    plt.xlabel('$\~V_e$ [m/s]')
    plt.ylabel('$\delta_e$ [rad]')
    plt.title('$\delta_e^{*} - \~V_{e}$')
    plt.legend()

    alpha = deg_2_rad(np.concatenate([data_1['a'], data_2['a']], axis=0)[:-2])

    alpharange = np.linspace(0.02, 0.14, 100)

    bestfit_alpha = delta_alpha_fit(alpha, delta_e_eq_star[:-2])

    fig2 = plt.figure()
    plt.plot(alpha, delta_e_eq_star[:-2], 'rx', label='Measured Data')
    plt.plot(alpharange, bestfit_alpha(alpharange), 'b-', label='Best Fit')
    plt.ylim(max(delta_e_eq_star[:-2]), min(delta_e_eq_star[:-2]))
    plt.xlabel(r'$\alpha$ [rad]')
    plt.ylabel('$\delta_e$ [rad]')
    plt.legend()
    plt.title(r'$\delta_e^{*} - \alpha$')
    plt.grid()


    return bestfit_alpha, bestfit_ve


def stick_force_curve(data_object):

    data_1 = data_object.get_pfd('StatElev.csv')
    data_2 = data_object.get_pfd('GravShift.csv')
    matdata = data_object.get_mat().get_data()

    times = np.concatenate([data_1['time'], data_2['time']], axis=0)

    Fe = np.concatenate([data_1['Fe'], data_2['Fe']], axis=0)

    W0 = 60500.0
    Weight = [get_weight_at_t(t, matdata['time'], matdata['rh_engine_FU'], matdata['lh_engine_FU'], W0=W0) for t in times]

    Ve = np.concatenate([data_1['IAS'], data_2['IAS']], axis=0)
    Ve_tilde = [ve * np.sqrt(W0 / w) for ve, w in zip(Ve, Weight)]

    Fe_s = [fe*W0/W for fe, W in zip(Fe, Weight)]

    ve_tilde_range = np.linspace(0.9 * min(Ve_tilde), 1.1 * max(Ve_tilde), 100)

    force_fit_curve = stickforce_fit(Ve_tilde, Fe_s)

    fig1 = plt.figure()
    plt.plot(Ve_tilde[:-2], Fe_s[:-2], 'rx', label='Measured Data')
    plt.plot(ve_tilde_range, force_fit_curve(ve_tilde_range), 'b-', label='Best Fit')
    plt.ylim(max(Fe_s), min(Fe_s))
    plt.grid()
    plt.xlabel('$\~V_e$ [m/s]')
    plt.ylabel('$F_{e}^{*}$ [N]')
    plt.title('$F_{e}^{*} - \~V_{e}$')
    plt.legend()


if __name__ == "__main__":

    Ts = np.sum(get_thrust('thrust_stand.dat'), axis=1)
    T = np.sum(get_thrust('thrust.dat'), axis=1)
    data = Data(r'FlightData.mat', 'StatClCd.csv', 'StatElev.csv', 'GravShift.csv')

    b1, b2 = plot_trim_curve(data)

    stick_force_curve(data)

    slope = derive(b1)
