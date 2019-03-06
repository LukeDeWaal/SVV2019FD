import numpy as np
from src.data_extraction import mat_import
from src.misc.NumericalTools import least_squares, newtons_method
from src.data_extraction.mat_import import MatFileImport
from src.data_processing.get_weight import get_weight
from src.misc.data_access import get_data_file_path
import matplotlib.pyplot as plt

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


def reynolds_number(u, L, v):
    """
    Reynolds Number
    :param u: Velocity
    :param L: Reference length
    :param v: Dynamic viscosity
    :return: Re
    """
    return (u*L)/v


def prandtl_glauert(c, M):
    """
    Prandtly Glauert Correction for subsonic compressible flow
    :param c: coefficient
    :param M: Mach number
    :return: Corrected coefficient
    """
    return c/np.sqrt(1-M**2)


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

    print(cl_list.index(max(cl_list)))

    cl_alpha, c0 = least_squares(alpha_list, cl_list).reshape(2,)
    alpha_0 = -c0/cl_alpha

    def Cl(alpha):
        return cl_alpha*(alpha - alpha_0)

    return Cl, cl_list, alpha_list


data = MatFileImport(get_data_file_path('ExampleData.mat')).get_data()

time = data['time']
AOA = data['vane_AOA']
V = data['Dadc1_tas']
rhfu = data['rh_engine_FU']
lhfu = data['lh_engine_FU']
height = data['Dadc1_alt']
dens = np.array([ISA(h)[2] for h in height])
W = get_weight(time, rhfu, lhfu)

c, c1, a = calc_Cl(W[5000:10000], dens[5000:10000], V[5000:10000], AOA[5000:10000], S=30.0)

plt.plot(a, c1, 'rx')
plt.plot(a, [c(i) for i in a], 'b-')