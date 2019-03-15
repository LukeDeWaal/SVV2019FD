import numpy as np
import matplotlib.pyplot as plt
from src.misc import least_squares, newtons_method
from src.data_processing.get_weight import get_weight_at_t

__all__ = ['indicated_to_true_airspeed', 'reynolds_number', 'prandtl_glauert', 'ISA', 'Layers']

#%% Constants for ISA

g0  = 9.80665
R   = 287.0
p0  = 101325.0
d0  = 1.225

Nn = np.array(["Troposphere", "Tropopause", "Stratosphere", "Stratosphere", "Stratopause", "Mesosphere", "Mesosphere", "Mesopause", "Thermosphere", "Thermosphere", "Thermosphere"])
Tn = np.array([288.15, 216.65, 216.65, 228.65, 270.65, 270.65, 214.65, 186.95, 186.95, 201.95, 251.95])
Hn = np.array([0, 11000.0, 20000.0, 32000.0, 47000.0, 51000.0, 71000.0, 84852.0, 90000.0, 100000.0, 110000.0, 120000.0])
Lt = np.array([1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1])


def indicated_to_true_airspeed(v_eas, rho, rho_0=1.225):
    """
    Convert IAS to TAS
    :param v_eas: EAS
    :param rho: Density
    :param rho_0: density at sealevel = 1.225 kg/m^3
    :return: TAS
    """
    return v_eas*np.sqrt(rho_0/rho)


def reynolds_number(u, L, T, rho):
    """
    Reynolds Number
    :param u: Velocity
    :param L: Reference length
    :param T: Temperature
    :param rho: Density
    :return: Re
    """
    mu = sutherland_dynamic_viscosity(T)
    v = kinematic_viscosity(mu, rho)
    return (u*L)/v


def sutherland_dynamic_viscosity(T, b=1.458e-6, S=110.4):
    """
    Calculate air viscosity at Temperature T
    :param T: Temperature of air
    :param b: Constant for air
    :param S: Constant for air
    :return: Dynamic Viscosity
    """
    return b*T**(3/2)/(T+S)


def kinematic_viscosity(dyn_visc, rho):
    """
    Calculate Kinematic Viscosity from Dynamic Viscosity and density
    :param dyn_visc: Dynamic Viscosity
    :param rho: Density
    :return: Kinematic Viscosity
    """
    return dyn_visc/rho


def prandtl_glauert(c, M):
    """
    Prandtly Glauert Correction for subsonic compressible flow
    :param c: coefficient
    :param M: Mach number
    :return: Corrected coefficient
    """
    return c/np.sqrt(1-M**2)


def reduced_eq_airspeed(V, W, Ws=60500.0):
    """
    Calcukate V-tilde
    :param V: TAS
    :param W: True Weight
    :param Ws: Standard Weight
    :return: V-tilde
    """
    return V*np.sqrt(Ws/W)

# Helper Class for ISA Calculator
class Layers:

    @staticmethod
    def Isothermal_Layer(h, k, P0, D0):

        T0 = Tn[k]

        T = T0
        P = P0
        D = D0

        if h < Hn[k + 1] and h > Hn[k]:
            P = P * np.exp(-g0 / (R * T) * (h - Hn[k]))
            D = D * np.exp(-g0 / (R * T) * (h - Hn[k]))

        elif h >= Hn[k + 1]:
            P = P * np.exp(-g0 / (R * T) * (Hn[k + 1] - Hn[k]))
            D = D * np.exp(-g0 / (R * T) * (Hn[k + 1] - Hn[k]))

        elif h == Hn[k]:
            P = P0
            D = D0

        return (T, P, D)

    @staticmethod
    def Normal_Layer(h, k, P0, D0):

        T0 = Tn[k]

        T = T0
        P = P0
        D = D0

        if h < Hn[k + 1] and h > Hn[k]:

            L = (Tn[k + 1] - Tn[k]) / (Hn[k + 1] - Hn[k])
            C = -g0 / (L * R)  # To Simplify and shorten code we define the following expression for the exponent

            T = T0 + L * (h - Hn[k])
            P = P0 * (T / Tn[k]) ** C
            D = D0 * (T / Tn[k]) ** (C - 1)

        elif h >= Hn[k + 1]:

            L = (Tn[k + 1] - Tn[k]) / (Hn[k + 1] - Hn[k])
            C = -g0 / (L * R)  # To Simplify and shorten code we define the following expression for the exponent

            T = T0 + L * (Hn[k + 1] - Hn[k])
            P = P0 * (T / Tn[k]) ** C
            D = D0 * (T / Tn[k]) ** (C - 1)

        elif h == Hn[k]:

            T = T0
            P = P0
            D = D0

        return T, P, D


# ISA calculator to obtain density
def ISA(h):
    """
    ISA calculator to obtain Temp, Press and Dens at h
    :param h: height in m
    :return: Temp, Press and Dens at h
    """

    T = Tn[0]
    P = p0
    D = d0

    if h < 0:
        return T, P, D

    i = 0
    while True:

        if i >= len(Lt):
            return T, P, D

        base_h = Hn[i]
        try:
            top_h  = Hn[i+1]
        except IndexError:
            top_h  = Hn[i]+100

        layer_type = Lt[i]

        if base_h <= h < top_h:
            if layer_type == 1:
                T, P, D = Layers.Normal_Layer(h, i, P, D)

            elif layer_type == 0:
                T, P, D = Layers.Isothermal_Layer(h, i, P, D)

        elif h >= top_h:
            if layer_type == 1:
                T, P, D = Layers.Normal_Layer(top_h, i, P, D)

            elif layer_type == 0:
                T, P, D = Layers.Isothermal_Layer(top_h, i, P, D)

        i += 1


def calc_Cl(W_list, rho_list, V_list, alpha_list, S=0.0):
    """
    Least Squares Solution for Cl function of alpha
    :param W_list: Weight at all times
    :param rho_list: Density at all times
    :param V_list: Velocity at all times
    :param alpha_list: AOA at all times
    :param S: Surface area (constant)
    :return: Function Cl(alpha)
    """

    cl_list = [W/(0.5*S*rho*V**2) for W, rho, V in zip(W_list, rho_list, V_list)]

    cl_alpha, c0 = least_squares(alpha_list, cl_list).reshape(2,)
    alpha_0 = -c0/cl_alpha

    def Cl(alpha):
        return cl_alpha*(alpha - alpha_0)

    return Cl, cl_list, alpha_list


def get_CL_alpha(data_object):

    files = data_object.get_pfd_files()

    mdat = data.get_mat().get_data()
    pdat = data.get_pfd()

    mtime = mdat['time']
    lhfu = mdat['lh_engine_FU']
    rhfu = mdat['rh_engine_FU']

    cl_functions = []
    cl_lists = []
    alpha_lists = []

    for file in files:
        ptime = pdat[file]['time']
        pheight = pdat[file]['hp']

        W = [get_weight_at_t(t, mtime, lhfu, rhfu) for t in ptime]
        rho = [ISA(h)[2] for h in pheight]
        V = [v for v in pdat[file]['TAS']]
        a = [alpha for alpha in pdat[file]['a']]

        clcurve, cllist, alist = calc_Cl(W, rho, V, a, S=30.0)

        cl_functions.append(clcurve), cl_lists.append(cllist), alpha_lists.append(alist)

    alpharange = np.linspace(-5, 12, 100)
    clrange = np.linspace(-0.4, 1.0, 100)

    fig = plt.figure()
    plt.plot(alpharange, [0] * len(alpharange), 'k-')
    plt.plot([0] * len(clrange), clrange, 'k-')
    plt.plot(alpha_lists[0], cl_lists[0], 'rx', label='Measured Data')
    plt.plot(alpharange, [cl_functions[0](alpha) for alpha in alpharange], 'b-', label='Best Fit')
    plt.grid()
    plt.xlabel('Alpha [deg]')
    plt.ylabel('Cl [-]')
    plt.title('Cl - Alpha Curve')
    plt.legend()


if __name__ == "__main__":

    from src.data_extraction import Data

    data = Data(r'FlightData.mat', 'StatClCd.csv', 'StatElev.csv', 'GravShift.csv')

    get_CL_alpha(data)