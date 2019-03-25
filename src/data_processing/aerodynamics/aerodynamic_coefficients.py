import numpy as np
import matplotlib.pyplot as plt
from src.misc import linear_least_squares, newtons_method
from src.data_processing.get_weight import get_weight_at_t
from src.data_extraction.get_thrust_data import get_thrust

#%% Constants for ISA

g0  = 9.80665
R   = 287.0
p0  = 101325.0
d0  = 1.225

Nn = np.array(["Troposphere", "Tropopause", "Stratosphere", "Stratosphere", "Stratopause", "Mesosphere", "Mesosphere", "Mesopause", "Thermosphere", "Thermosphere", "Thermosphere"])
Tn = np.array([288.15, 216.65, 216.65, 228.65, 270.65, 270.65, 214.65, 186.95, 186.95, 201.95, 251.95])
Hn = np.array([0, 11000.0, 20000.0, 32000.0, 47000.0, 51000.0, 71000.0, 84852.0, 90000.0, 100000.0, 110000.0, 120000.0])
Lt = np.array([1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1])


#%% Constants for Wing
S = 30.0
b = 15.911
c = 2.0569

def indicated_to_true_airspeed(v_eas, rho, rho_0=1.225):
    """
    Convert IAS to TAS
    :param v_eas: EAS
    :param rho: Density
    :param rho_0: density at sealevel = 1.225 kg/m^3
    :return: TAS
    """
    return v_eas*np.sqrt(rho_0/rho)


def mach_from_cas(vc, h, gamma=1.4, rho0=1.225, p0=101325.0):
    T, p, rho = ISA(h)
    return np.sqrt(2/(gamma-1)*((1 + p0/p*((1 + (gamma-1)/(2*gamma)*rho0/p0*vc**2)**(gamma/(gamma-1))-1))**((gamma-1)/gamma) -1))


def temp_correction(Tm, M, gamma=1.4):
    return Tm/(1+(gamma-1)/2*M**2)


def ve_tilde(ve, w, w0=60500.0):
    return ve*np.sqrt(w0/w)


def speed_of_sound(T, gamma=1.4, R=287.0):
    return np.sqrt(gamma*R*T)


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

    Cl_list = [W/(0.5*S*rho*V**2) for W, rho, V in zip(W_list, rho_list, V_list)]

    cl_alpha, c0 = linear_least_squares(alpha_list, Cl_list).reshape(2, )
    alpha_0 = -c0/cl_alpha

    def Cl(alpha):
        return cl_alpha*(alpha - alpha_0)

    return Cl, Cl_list, alpha_list, cl_alpha, alpha_0


def calc_Cd(T_list, rho_list, V_list, Cl_list, S=0.0):
    """

    :param T_list: Thrust at all times
    :param rho_list: Density at all times
    :param V_list: Velocity at all times
    :param Cl_list: Lift Coefficient at all times
    :param S: Surface area (constant)
    :return: Function Cd(Cl)
    """
    Cd_list = [T/(0.5*S*rho*V**2) for T, rho, V in zip(T_list, rho_list, V_list)]
    Cl_sq_list = [cl**2 for cl in Cl_list]

    c_i, cd0 = linear_least_squares(Cl_sq_list, Cd_list).reshape(2, )

    AR = 15.911**2/S
    oswald = 1.0/(c_i*np.pi*AR)

    def Cd(Cl):
        return cd0 + c_i*Cl**2

    return Cd, Cd_list, Cl_list, c_i, cd0


def get_CD_alpha(data_object):

    mdat = data_object.get_mat().get_data()
    pdat = data_object.get_pfd()

    mtime = mdat['time']
    lhfu = mdat['lh_engine_FU']
    rhfu = mdat['rh_engine_FU']

    ptime = pdat['StatClCd.csv']['time']
    pheight = pdat['StatClCd.csv']['hp']

    thrust = get_thrust(which=1)

    W = [get_weight_at_t(t, mtime, lhfu, rhfu) for t in ptime]
    T = [tl + tr for i, (tl, tr) in thrust.iterrows()]
    rho = [ISA(h)[2] for h in pheight]
    temperature = [ISA(h)[0] for h in pheight]
    V = [v for v in pdat['StatClCd.csv']['TAS']]
    a = [alpha for alpha in pdat['StatClCd.csv']['a']]

    clcurve, cllist, alist, cl_alpha, alpha_0 = calc_Cl(W, rho, V, a, S=30.0)
    cdcurve, cdlist, cllist, c_i, cd0 = calc_Cd(T, rho, V, cllist, S=30.0)

    def cd_alpha(a):
        cl = clcurve(a)
        return cdcurve(cl)

    def theoretical_cdalpha(a):

        cd0 = 0.04
        e = 0.8
        AR = 15.911**2/30.0
        a0 = alpha_0
        cl_alpha = 5.084 * np.pi / 180.0

        return cd0 + 1.0/(np.pi*AR*e)*(cl_alpha*(a - a0))**2

    alpharange = np.linspace(-2.5, 12.5, 100)
    cdrange = np.linspace(0.0, 0.2, 100)

    Re_range = [reynolds_number(u, c, ISA(h)[0], ISA(h)[2]) for u, h in zip(V, pheight)]
    Re_range = (round(min(Re_range)), round(max(Re_range)))
    Re_range = ('{:0.3e}'.format(Re_range[0]), '{:0.3e}'.format(Re_range[1]))
    print("CDa: ",Re_range)

    M_range = [v/speed_of_sound(temp) for v, temp in zip(V, temperature)]
    M_range = (min(M_range), max(M_range))
    print("CDa: ",M_range)

    fig = plt.figure()
    plt.plot([0] * len(cdrange), cdrange, 'k-')
    plt.plot(alpharange, [0] * len(alpharange), 'k-')
    plt.plot(alist, cdlist, 'rx', label='Measured Data')
    plt.plot(alpharange, [cd_alpha(a) for a in alpharange], 'b-', label='Best Fit')
    plt.plot(alpharange, [theoretical_cdalpha(a) for a in alpharange], 'g-', label='Theoretical Values')
    plt.grid()
    plt.xlabel(r'$\alpha [deg]$')
    plt.ylabel('$Cd [-]$')
    plt.title(r'$Cd - \alpha$' + '  Curve')
    plt.legend()


def get_CD_CL(data_object):

    mdat = data_object.get_mat().get_data()
    pdat = data_object.get_pfd()

    mtime = mdat['time']
    lhfu = mdat['lh_engine_FU']
    rhfu = mdat['rh_engine_FU']

    ptime = pdat['StatClCd.csv']['time']
    pheight = pdat['StatClCd.csv']['hp']

    thrust = get_thrust(which=1)

    W = [get_weight_at_t(t, mtime, lhfu, rhfu) for t in ptime]
    T = [tl+tr for i, (tl, tr) in thrust.iterrows()]
    rho = [ISA(h)[2] for h in pheight]
    temperature = [ISA(h)[0] for h in pheight]
    V = [v for v in pdat['StatClCd.csv']['TAS']]
    a = [alpha for alpha in pdat['StatClCd.csv']['a']]

    clcurve, cllist, alist, cl_alpha, alpha_0 = calc_Cl(W, rho, V, a, S=30.0)
    cdcurve, cdlist, cllist, c_i, cd0 = calc_Cd(T, rho, V, cllist, S=30.0)

    # print('cd0 = ',cd0)
    # print('oswald = ',(c_i*np.pi*b/c)**(-1))

    def theoretical_cd(cl):
        cd0 = 0.04
        e = 0.8
        AR = 15.911**2/30.0
        return cd0+1.0/(np.pi*AR*e)*cl**2

    cl_range = np.linspace(0.0, 1.5, 100)
    cd_range = np.linspace(0.0, 0.2, 100)

    Re_range = [reynolds_number(u, c, ISA(h)[0], ISA(h)[2]) for u, h in zip(V, pheight)]
    Re_range = (round(min(Re_range)), round(max(Re_range)))
    Re_range = ('{:0.3e}'.format(Re_range[0]), '{:0.3e}'.format(Re_range[1]))
    print("CDCL: ",Re_range)

    M_range = [v/speed_of_sound(temp) for v, temp in zip(V, temperature)]
    M_range = (min(M_range), max(M_range))
    print("CDCL: ",M_range)

    fig = plt.figure()
    plt.plot([0] * len(cl_range), cl_range, 'k-')
    plt.plot(cd_range, [0] * len(cd_range), 'k-')
    plt.plot(cdlist, cllist, 'rx', label='Measured Data')
    plt.plot([cdcurve(cl) for cl in cl_range], cl_range, 'b-', label='Best Fit')
    plt.plot([theoretical_cd(cl) for cl in cl_range], cl_range, 'g-', label='Theoretical Values')
    plt.grid()
    #plt.text(6, 0.4, r'$Cd = Cd_{0} + (c_i \cdot Cl^{2)$')
    #plt.text(6.5, 0.375,f'$= {round(cl_alpha, 4)} $' + r'$\cdot (\alpha - $' + f'{round(np.sign(alpha_0) * alpha_0, 4)})')
    plt.xlabel(r'$Cd [-]$')
    plt.ylabel('$Cl [-]$')
    plt.title(r'$Cl - Cd$' + '  Curve')
    plt.legend()


def get_CL_alpha(data_object):

    mdat = data_object.get_mat().get_data()
    pdat = data_object.get_pfd()

    mtime = mdat['time']
    lhfu = mdat['lh_engine_FU']
    rhfu = mdat['rh_engine_FU']

    ptime = pdat['StatClCd.csv']['time']
    pheight = pdat['StatClCd.csv']['hp']

    W = [get_weight_at_t(t, mtime, lhfu, rhfu) for t in ptime]
    rho = [ISA(h)[2] for h in pheight]
    temperature = [ISA(h)[0] for h in pheight]
    V = [v for v in pdat['StatClCd.csv']['TAS']]
    a = [alpha for alpha in pdat['StatClCd.csv']['a']]

    clcurve, cllist, alist, cl_alpha, alpha_0 = calc_Cl(W, rho, V, a, S=30.0)

    # print("Cl_alpha = ", cl_alpha)
    # print("a0 = ", alpha_0)

    def theoretical_cl(a):
        cl_alpha = 5.084*np.pi/180.0
        return cl_alpha*(a - alpha_0)

    alpharange = np.linspace(-2.5, 12.5, 100)
    clrange = np.linspace(-0.2, 1.2, 100)

    Re_range = [reynolds_number(u, c, ISA(h)[0], ISA(h)[2]) for u, h in zip(V, pheight)]
    Re_range = (round(min(Re_range)), round(max(Re_range)))
    Re_range = ('{:0.3e}'.format(Re_range[0]), '{:0.3e}'.format(Re_range[1]))
    print("Cla: ",Re_range)

    M_range = [v/speed_of_sound(temp) for v, temp in zip(V, temperature)]
    M_range = (min(M_range), max(M_range))
    print("Cla: ",M_range)

    fig = plt.figure()
    plt.plot(alpharange, [0] * len(alpharange), 'k-')
    plt.plot([0] * len(clrange), clrange, 'k-')
    plt.plot(alist, cllist, 'rx', label='Measured Data')
    plt.plot(alpharange, [clcurve(alpha) for alpha in alpharange], 'b-', label='Best Fit')
    plt.plot(alpharange, [theoretical_cl(a) for a in alpharange], 'g-', label='Theoretical Values')
    plt.grid()
    #plt.text(6, 0.4, r'$Cl = Cl_{\alpha} \cdot (\alpha - \alpha_0)$')
    #plt.text(6.5, 0.375, f'$= {round(cl_alpha, 4)} $' + r'$\cdot (\alpha - $' + f'{round(np.sign(alpha_0)*alpha_0, 4)})')
    plt.xlabel(r'$\alpha [deg]$')
    plt.ylabel('$Cl [-]$')
    plt.title(r'$Cl - \alpha$' + '  Curve')
    plt.legend()


if __name__ == "__main__":

    from src.data_extraction import Data

    data = Data(r'FlightData.mat', 'StatClCd.csv', 'StatElev.csv', 'GravShift.csv')

    get_CL_alpha(data)
    get_CD_CL(data)
    get_CD_alpha(data)
