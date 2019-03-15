from data.Cit_par import *
import numpy as np


class Eigenmotions:
    def __init__(self):
        self.eigenvalue_spm = self.__calc_spm()
        self.eigenvalue_phugoid = self.__calc_phugoid()
        self.eigenvalue_dutch_roll = self.__calc_dutch_roll()
        self.eigenvalue_aperiodic_roll = self.__calc_aperiodic_roll()
        self.eigenvalue_spiral_motion = self.__calc_spiral_motion()

    @staticmethod
    def __calc_eigenvalues(u, v, w):
        return [(-v+np.sqrt(v**2-4*u*w))/(2*u), (-v-np.sqrt(v**2-4*u*w))/(2*u)]

    def __calc_spm(self):
        coef_a = 2*muc*KY2*(2*muc-CZadot)
        coef_b = -2*muc*KY2*CZa*(2*muc-CZq)*Cmadot-(2*muc-CZadot)*Cma
        coef_c = CZa*Cmq-(2*muc+CZq)*Cma

        return self.__calc_eigenvalues(coef_a, coef_b, coef_c)

    def __calc_phugoid(self):
        coef_a = 2*muc*(CZa*Cmq-2*muc*Cma)
        coef_b = 2*muc*(CXu*Cma-Cmu*CXa)+Cmq*(CZu*CXa-CXu*CZa)
        coef_c = CZ0*(Cmu*CZa-Cma*CZu)

        return self.__calc_eigenvalues(coef_a, coef_b, coef_c)

    def __calc_dutch_roll(self):
        coef_a = 8*mub**2*KZ2
        coef_b = -2*mub*(Cnr+2*KZ2*CYb)
        coef_c = 4*mub*Cnb+CYb*Cnr

        return self.__calc_eigenvalues(coef_a, coef_b, coef_c)

    @staticmethod
    def __calc_aperiodic_roll():
        return Clp/(4*mub*KX2)

    @staticmethod
    def __calc_spiral_motion():
        return (2*CL*(Clb*Cnr-Cnb-Clr))/(Clp*(CYb*Cnr+4*mub*Cnb)-Cnp*(CYb*Clr+4*mub*Clb))


if __name__ == "__main__":
    eigenmotions_1 = Eigenmotions()

    print(eigenmotions_1.eigenvalue_spm)
    print(eigenmotions_1.eigenvalue_phugoid)
    print(eigenmotions_1.eigenvalue_dutch_roll)
    print(eigenmotions_1.eigenvalue_aperiodic_roll)
    print(eigenmotions_1.eigenvalue_spiral_motion)