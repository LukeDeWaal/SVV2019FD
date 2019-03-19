from data.Cit_par import *
from src.data_extraction.data_main import Data
from src.data_processing.get_weight import get_weight_at_t
from src.data_processing.aerodynamics import ISA

import cmath

class Eigenmotions:
    def __init__(self, time_spm=0, time_phugoid=0, time_dutch_roll=0, time_aperiodic_roll=0, time_spiral_motion=0):
        if time_spm != 0:
            self.eigenvalue_spm, self.prop_spm = self.__calc_spm(time_spm)
        else:
            self.eigenvalue_spm = "Time of motion not provided"

        if time_phugoid != 0:
            self.eigenvalue_phugoid, self.prop_phugoid = self.__calc_phugoid(time_phugoid)
        else:
            self.eigenvalue_phugoid = "Time of motion not provided"

        if time_dutch_roll != 0:
            self.eigenvalue_dutch_roll, self.prop_dutch_roll = self.__calc_dutch_roll(time_dutch_roll)
        else:
            self.eigenvalue_dutch_roll = "Time of motion not provided"

        if time_aperiodic_roll != 0:
            self.eigenvalue_aperiodic_roll, self.prop_aperiodic_roll = self.__calc_aperiodic_roll(time_aperiodic_roll)
        else:
            self.eigenvalue_aperiodic_roll = "Time of motion not provided"

        if time_spiral_motion != 0:
            self.eigenvalue_spiral_motion, self.prop_spiral_motion = self.__calc_spiral_motion(time_spiral_motion)
        else:
            self.eigenvalue_spiral_motion = "Time of motion not provided"

    @staticmethod
    def __get_flight_conditions(t):
        data = Data(r'FlightData.mat')
        mat_data = data.get_mat().get_data()
        time = mat_data['time']
        rh_fu = mat_data['rh_engine_FU']
        lh_fu = mat_data['lh_engine_FU']

        alt = mat_data['Dadc1_alt']

        for idx, t_i in enumerate(time):
            if time[idx] < t <= time[idx+1]:
                break

        m = get_weight_at_t(t, time, rh_fu, lh_fu)

        h = alt[idx]
        rho = ISA(h)[2]

        mub = m / (rho * S * b)
        muc = m / (rho * S * c)

        return mub, muc

    @staticmethod
    def __calc_eigenvalues(u, v, w):
        return (-v+cmath.sqrt(v**2-4*u*w))/(2*u), (-v-cmath.sqrt(v**2-4*u*w))/(2*u)

    @staticmethod
    def __calc_eigenvalue_properties(eigenvalue):
        zeta = -eigenvalue.real/cmath.sqrt(eigenvalue.real**2+eigenvalue.imag)
        half_t = cmath.log(1 / 2) / eigenvalue.real

        if eigenvalue.imag == 0:
            P = None
            nat_frequency = None

        else:
            P = (2*pi)/eigenvalue.imag
            nat_frequency = (2*pi)/P

        return {"zeta": zeta.real, "period": P, "natural_freq": nat_frequency, "half_time": half_t.real}

    # ----------------------------------------------------------------------------------------
    def __calc_spm(self, t):
        mub, muc = self.__get_flight_conditions(t)

        coef_a = 2*muc*KY2*(2*muc-CZadot)
        coef_b = -2*muc*KY2*CZa-(2*muc-CZq)*Cmadot-(2*muc-CZadot)*Cma
        coef_c = CZa*Cmq-(2*muc+CZq)*Cma

        eigenvalue = list(self.__calc_eigenvalues(coef_a, coef_b, coef_c))[0][0]

        return eigenvalue, self.__calc_eigenvalue_properties(eigenvalue)

    def __calc_phugoid(self, t):
        mub, muc = self.__get_flight_conditions(t)

        coef_a = 2*muc*(CZa*Cmq-2*muc*Cma)
        coef_b = 2*muc*(CXu*Cma-Cmu*CXa)+Cmq*(CZu*CXa-CXu*CZa)
        coef_c = CZ0*(Cmu*CZa-Cma*CZu)

        eigenvalue = list(self.__calc_eigenvalues(coef_a, coef_b, coef_c))[0][0]

        return eigenvalue, self.__calc_eigenvalue_properties(eigenvalue)

    def __calc_dutch_roll(self, t):
        mub, muc = self.__get_flight_conditions(t)

        coef_a = 8*mub**2*KZ2
        coef_b = -2*mub*(Cnr+2*KZ2*CYb)
        coef_c = 4*mub*Cnb+CYb*Cnr

        eigenvalue = (self.__calc_eigenvalues(coef_a, coef_b, coef_c))[0][0]

        return eigenvalue, self.__calc_eigenvalue_properties(eigenvalue)

    def __calc_aperiodic_roll(self, t):
        mub, muc = self.__get_flight_conditions(t)
        eigenvalue = (Clp/(4*mub*KX2))[0].astype(complex)

        return eigenvalue, self.__calc_eigenvalue_properties(eigenvalue)

    def __calc_spiral_motion(self, t):
        mub, muc = self.__get_flight_conditions(t)

        eigenvalue = (2*CL*(Clb*Cnr-Cnb-Clr))/(Clp*(CYb*Cnr+4*mub*Cnb)-Cnp*(CYb*Clr+4*mub*Clb))[0].astype(complex)

        return eigenvalue, self.__calc_eigenvalue_properties(eigenvalue)


if __name__ == "__main__":
    eigenmotions_1 = Eigenmotions(time_spm=2772, time_phugoid=2864, time_dutch_roll=3067, time_aperiodic_roll=3310, time_spiral_motion=3391)

    print("Short Period motion:", eigenmotions_1.eigenvalue_spm)
    print(eigenmotions_1.prop_spm)

    print("\nPhugoid:", eigenmotions_1.eigenvalue_phugoid)
    print(eigenmotions_1.prop_phugoid)

    print("\nDutch roll:", eigenmotions_1.eigenvalue_dutch_roll)
    print(eigenmotions_1.prop_dutch_roll)

    print("\nAperiodic roll:", eigenmotions_1.eigenvalue_aperiodic_roll)
    print(eigenmotions_1.prop_aperiodic_roll)

    print("\nSpiral motion:", eigenmotions_1.eigenvalue_spiral_motion)
    print(eigenmotions_1.prop_spiral_motion)