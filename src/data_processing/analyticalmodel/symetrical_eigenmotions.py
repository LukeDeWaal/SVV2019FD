from data.Cit_par import *
from src.data_extraction.data_main import Data
from src.data_processing.get_weight import get_weight_at_t
from src.data_processing.aerodynamics import ISA

import cmath


class Eigenmotions:
    def __init__(self, time_spm=0, time_phugoid=0, time_dutch_roll=0, time_aperiodic_roll=0, time_spiral_motion=0):
        if time_spm != 0:
            self.eigenvalue_spm = self.__calc_spm(time_spm)
        else:
            self.eigenvalue_spm = None

        if time_phugoid != 0:
            self.eigenvalue_phugoid = self.__calc_phugoid(time_phugoid)
        else:
            self.eigenvalue_phugoid = None

        if time_dutch_roll != 0:
            self.eigenvalue_dutch_roll = self.__calc_dutch_roll(time_dutch_roll)
        else:
            self.eigenvalue_dutch_roll = None

        if time_aperiodic_roll != 0:
            self.eigenvalue_aperiodic_roll = self.__calc_aperiodic_roll(time_aperiodic_roll)
        else:
            self.eigenvalue_aperiodic_roll = None

        if time_spiral_motion != 0:
            self.eigenvalue_spiral_motion = self.__calc_spiral_motion(time_spiral_motion)
        else:
            self.eigenvalue_spiral_motion = None

    @staticmethod
    def __get_flight_conditions(t):
        data = Data(r'FlightData.mat')
        mat_data = data.get_mat().get_data()
        time = mat_data['time']
        rh_fu = mat_data['rh_engine_FU']
        lh_fu = mat_data['lh_engine_FU']

        alt = mat_data['Dadc1_alt']

        for idx, t_i in enumerate(time):
            if t_i > time[idx] and t_i < time[idx+1]:
                break

        m = get_weight_at_t(t, time, rh_fu, lh_fu)

        h = alt[idx]
        rho = ISA(h)[2]

        mub = m / (rho * S * b)
        muc = m / (rho * S * c)

        return mub, muc

    @staticmethod
    def __calc_eigenvalues(u, v, w):
        return [(-v+cmath.sqrt(v**2-4*u*w))/(2*u), (-v-cmath.sqrt(v**2-4*u*w))/(2*u)]

    def __calc_spm(self, t):
        coef_a = 2*muc*KY2*(2*muc-CZadot)
        coef_b = -2*muc*KY2*CZa*(2*muc-CZq)*Cmadot-(2*muc-CZadot)*Cma
        coef_c = CZa*Cmq-(2*muc+CZq)*Cma

        return self.__calc_eigenvalues(coef_a, coef_b, coef_c)

    def __calc_phugoid(self, t):
        mub, muc = self.__get_flight_conditions(t)

        coef_a = 2*muc*(CZa*Cmq-2*muc*Cma)
        coef_b = 2*muc*(CXu*Cma-Cmu*CXa)+Cmq*(CZu*CXa-CXu*CZa)
        coef_c = CZ0*(Cmu*CZa-Cma*CZu)

        return self.__calc_eigenvalues(coef_a, coef_b, coef_c)

    def __calc_dutch_roll(self, t):
        mub, muc = self.__get_flight_conditions(t)

        coef_a = 8*mub**2*KZ2
        coef_b = -2*mub*(Cnr+2*KZ2*CYb)
        coef_c = 4*mub*Cnb+CYb*Cnr

        return self.__calc_eigenvalues(coef_a, coef_b, coef_c)

    def __calc_aperiodic_roll(self, t):
        mub, muc = self.__get_flight_conditions(t)

        return Clp/(4*mub*KX2)

    def __calc_spiral_motion(self, t):
        mub, muc = self.__get_flight_conditions(t)

        return (2*CL*(Clb*Cnr-Cnb-Clr))/(Clp*(CYb*Cnr+4*mub*Cnb)-Cnp*(CYb*Clr+4*mub*Clb))


if __name__ == "__main__":
    eigenmotions_1 = Eigenmotions(time_spm=2772, time_phugoid=2864, time_dutch_roll=3067, time_aperiodic_roll=3310, time_spiral_motion=3391)

    print("Short Period motion:", eigenmotions_1.eigenvalue_spm)
    print("Phugoid:", eigenmotions_1.eigenvalue_phugoid)
    print("Dutch roll:", eigenmotions_1.eigenvalue_dutch_roll)
    print("Aperiodic roll:", eigenmotions_1.eigenvalue_aperiodic_roll)
    print("Spiral motion:", eigenmotions_1.eigenvalue_spiral_motion)
