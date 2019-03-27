# from statespace.test_Cit_par import *
from data.Cit_par import *


# C_n_b: Change in the moment Cn due to the change in side slip angle β, also called the static directional stability
# or Weathercock stability. If Cnβ is positive, then the aircraft is stable for yawing motions. Its main contributor is the vertical tailplane.
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 15:40:15 2019

@author: Matheus
"""
import control
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import math

# Import reference values:

# dirname = os.path.dirname(os.path.realpath(__file__))
# Cit_parStr = os.path.join(dirname, r'..\\..\\data')
# Cit_parStr = os.path.abspath(os.path.realpath(Cit_parStr))
#
from src.data_extraction.time_series_tool import TimeSeriesTool
from src.data_extraction.data_main import Data
from src.data_processing.get_weight import get_weight_at_t
from src.data_processing.aerodynamics import ISA

# sys.path.append(Cit_parStr)

from data.Cit_par import *

# These reference values are missing from Cit_par:

class AsymmetricStateSpace:

    def __init__(self, time_start, time_length, Cma, Cmde, maneuver_name):
        self.time_start = time_start
        self.time_length= time_length
        self.time_end = self.time_start + self.time_length
        self.Cma = Cma
        self.Cmde = Cmde
        self.ts_tool = TimeSeriesTool()
        self.maneuver_name = maneuver_name
        # self.data_tool = Data()

    def maneuver_vals(self):
        t = list(range(self.time_start, self.time_start + self.time_length))
        da = []
        dr = []
        phi = []
        p = []
        r = []
        V = []
        yawRate = []
        beta = [0.0]

        for time in t:
            specific_t_mdat_vals = self.ts_tool.get_t_specific_mdat_values(time)
            da.append(specific_t_mdat_vals['delta_a'][0])
            dr.append(specific_t_mdat_vals['delta_r'][0])
            phi.append(specific_t_mdat_vals['Ahrs1_Roll'][0])
            p.append(specific_t_mdat_vals['Ahrs1_bRollRate'][0])
            r.append(specific_t_mdat_vals['Ahrs1_bYawRate'][0])
            V.append(specific_t_mdat_vals['Dadc1_tas'][0])
            yawRate.append(specific_t_mdat_vals['Ahrs1_bYawRate'][0])

        for i in range(len(yawRate) - 1):
            beta.append(beta[i] + yawRate[i])

        t = np.asarray(t)
        phi = np.asarray(phi)
        p = np.asarray(p)
        r = np.asarray(r)
        da = np.asarray(da)
        dr = np.asarray(dr)
        V = np.asarray(V)
        beta = np.asarray(beta)

        x0 = np.array([[0.0],  # beta
                       [phi[0]],  # phi
                       [p[0]],  # p
                       [r[0]]])  # r
        return t, phi, p, r, da, dr, x0, V[0], beta


    def get_flight_conditions(self):
        data = Data(r'FlightData.mat')
        mat_data = data.get_mat().get_data()
        time = mat_data['time']
        rh_fu = mat_data['rh_engine_FU']
        lh_fu = mat_data['lh_engine_FU']

        alt = mat_data['Dadc1_alt']

        for idx, t_i in enumerate(time):
            if time[idx] < self.time_start <= time[idx + 1]:
                break

        m = get_weight_at_t(self.time_start, time, rh_fu, lh_fu) / 9.80665

        h = alt[idx]
        rho = ISA(h)[2]

        mub = m / (rho * S * b)
        muc = m / (rho * S * c)

        return mub, muc, m, h, rho


    def L2error(self, x_exact: np.array, x_numerical: np.array):
        error = 0.0
        if (x_exact.shape[0] != x_numerical.shape[0]):
            print("Vectors must be of the same size!")
            return 1
        for i in range(x_exact.shape[0]):
            error += (x_numerical[i] - x_exact[i]) * (x_numerical[i] - x_exact[i])

        error = math.sqrt(error) / x_exact.shape[0]
        return error


    def create_assymetric_ss_representation(self):
        forced_response_inputs = self.maneuver_vals()

        g = 9.80665

        # DUTCH ROLL:
        # State-space representation of asymmetric EOM:

        # C1*x' + C2*x + C3*u = 0

        V0 = forced_response_inputs[7]

        mub, muc, m, h, rho = self.get_flight_conditions()
        mub = float(mub[0])
        muc = float(muc[0])
        m = float(m[0])
        h = float(h[0])
        rho = float(rho[0])

        CX0 = m * g * sin(forced_response_inputs[3][0] * np.pi / 180.0) / (0.5 * rho * (V0 ** 2) * S)
        CZ0 = -m * g * cos(forced_response_inputs[3][0] * np.pi / 180.0) / (0.5 * rho * (V0 ** 2) * S)

        c = 2.0569

        C1 = np.array([[(CYbdot - 2.0 * mub) * b / V0, 0.0, 0.0, 0.0],
                       [0.0, -0.5 * b / V0, 0.0, 0.0],
                       [0.0, 0.0, -4.0 * mub * KX2 * b / V0, 4.0 * mub * KXZ * b / V0],
                       [Cnbdot * b / V0, 0.0, 4.0 * mub * KXZ * b / V0, -4.0 * mub * KZ2 * b / V0]])

        C2 = np.array([[-CYb, -CL, -CYp, -(CYr - 4.0 * mub)],
                       [0.0, 0.0, -1.0, 0.0],
                       [-Clb, 0.0, -Clp, -Clr],
                       [-Cnb, 0.0, -Cnp, -Cnr]])

        C3 = np.array([[-CYda, -CYdr],
                       [0.0, 0.0],
                       [-Clda, -Cldr],
                       [-Cnda, -Cndr]])

        # x' = A*x + B*u
        # y  = C*x + D*u

        # now u = [da]
        #         [dr]

        A = np.matmul(np.linalg.inv(C1), C2)
        B = np.matmul(np.linalg.inv(C1), C3)
        C = np.identity(4)  # y = x, meaning we output the state
        D = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])

        # Make control.ss state-space

        system = control.ss(A, B, C, D)

        # t = np.linspace(0.0, 200.0, num = 201)

        # Input:
        # u = np.zeros((2, t.shape[0]))

        # for i in range(t.shape[0]):
        #    u[0, i] = da[i] #Insert magnitude of "da" (aileron deflection) or comment out if none
        #    u[1, i] = dr[i] #Insert magnitude of "dr" (rudder deflection) or comment out if none

        u = np.array([forced_response_inputs[4],
                      forced_response_inputs[5]])

        # Calculate response to arbitrary input
        t, y, x = control.forced_response(system, forced_response_inputs[0], u, forced_response_inputs[6], transpose=False)

        # Change dimensionless pb/(2V) and rb/(2V) to p and r
        y[1, :] = -y[1, :] + 2 * y[1, 0]
        y[2, :] = -(2 * abs(V0) * y[2, :] / b) + 2 * y[2, 0]
        y[3, :] = -(2 * abs(V0) * y[3, :] / b) + 2 * y[3, 0]

        control.damp(system, doprint=True)

        fig = plt.figure(figsize=(12, 9))
        # fig.suptitle(self.maneuver_name, fontsize=16, )

        ax1 = fig.add_subplot(221)
        ax1.plot(t, y[0, :], label='Simulated response')
        ax1.plot(t, forced_response_inputs[8], label='Test Flight Data Response', color='tab:orange', linestyle='--')
        ax1.legend(loc='upper right')
        ax1.set_xlabel("Time [s]")
        ax1.set_ylabel("beta (yaw angle) [deg]")
        print("{0} Error in Beta:".format(self.maneuver_name))
        eigen_motion_error_beta = self.L2error(forced_response_inputs[8], y[0, :])
        print(eigen_motion_error_beta)
        plt.grid("True")

        ax2 = fig.add_subplot(222)
        ax2.plot(t, y[1, :], label='Simulated response')
        ax2.plot(t, forced_response_inputs[1], label='Test Flight Data Response', color='tab:orange', linestyle='--')
        ax2.legend(loc='upper right')
        ax2.set_xlabel("Time [s]")
        ax2.set_ylabel("phi (roll angle) [deg]")
        print("{0} Error in Phi:".format(self.maneuver_name))
        eigen_motion_error_phi = self.L2error(forced_response_inputs[1], y[1, :])
        print(eigen_motion_error_phi)
        plt.grid("True")

        ax3 = fig.add_subplot(223)
        ax3.plot(t, y[2, :], label='Simulated response')
        ax3.plot(t, forced_response_inputs[2], label='Test Flight Data Response', color='tab:orange', linestyle='--')
        ax3.legend(loc='upper right')
        ax3.set_xlabel("Time [s]")
        ax3.set_ylabel("p (roll rate) [deg/sec]")
        print("{0} Error in p:".format(self.maneuver_name))
        eigen_motion_error_p = self.L2error(forced_response_inputs[2], y[2, :])
        print(eigen_motion_error_p)
        plt.grid("True")

        ax4 = fig.add_subplot(224)
        ax4.plot(t, y[3, :], label='Simulated response')
        ax4.plot(t, forced_response_inputs[3], label='Test Flight Data Response', color='tab:orange', linestyle='--')
        ax4.legend(loc='upper right')
        ax4.set_xlabel("Time [s]")
        ax4.set_ylabel("r (yaw rate) [deg/s]")
        print("{0} Error in r:".format(self.maneuver_name))
        eigen_motion_error_r = self.L2error(forced_response_inputs[3], y[3, :])
        print(eigen_motion_error_r)
        plt.grid("True")

        print("Avg. {0} Error:".format(self.maneuver_name))
        avg_eigen_motion_error = (eigen_motion_error_beta + eigen_motion_error_p + eigen_motion_error_phi + eigen_motion_error_r) / 4.0
        print(avg_eigen_motion_error)
        # plt.grid("True")
        plt.show()


    def compute_assymetric_ss_representation_error(self, stability_coeffs_array):
        """

        :param stability_coeffs_array: [Cnb, Cnr, CYb]
        :return:
        """
        forced_response_inputs = self.maneuver_vals()
        # spiral = maneuver_vals(3305, 30, 'Spiral')
        # aperiodic = maneuver_vals(3380, 30, 'Aperiodic')

        g = 9.80665
        # State-space representation of asymmetric EOM:
        V0 = forced_response_inputs[7]

        mub, muc, m, h, rho = self.get_flight_conditions()
        mub = float(mub[0])
        muc = float(muc[0])
        m = float(m[0])
        h = float(h[0])
        rho = float(rho[0])

        CX0 = m * g * sin(forced_response_inputs[3][0] * np.pi / 180.0) / (0.5 * rho * (V0 ** 2) * S)
        CZ0 = -m * g * cos(forced_response_inputs[3][0] * np.pi / 180.0) / (0.5 * rho * (V0 ** 2) * S)

        c = 2.0569

        C1 = np.array([[(CYbdot - 2.0 * mub) * b / V0, 0.0, 0.0, 0.0],
                       [0.0, -0.5 * b / V0, 0.0, 0.0],
                       [0.0, 0.0, -4.0 * mub * KX2 * b / V0, 4.0 * mub * KXZ * b / V0],
                       [Cnbdot * b / V0, 0.0, 4.0 * mub * KXZ * b / V0, -4.0 * mub * KZ2 * b / V0]])

        C2 = np.array([[-stability_coeffs_array[2], -CL, -CYp, -(CYr - 4.0 * mub)],
                       [0.0, 0.0, -1.0, 0.0],
                       [-Clb, 0.0, -Clp, -Clr],
                       [-stability_coeffs_array[0], 0.0, -Cnp, -stability_coeffs_array[1]]])

        C3 = np.array([[-CYda, -CYdr],
                       [0.0, 0.0],
                       [-Clda, -Cldr],
                       [-Cnda, -Cndr]])

        A = np.matmul(np.linalg.inv(C1), C2)
        B = np.matmul(np.linalg.inv(C1), C3)
        C = np.identity(4)  # y = x, meaning we output the state
        D = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])

        # Make control.ss state-space

        system = control.ss(A, B, C, D)

        # t = np.linspace(0.0, 200.0, num = 201)

        # Input:
        # u = np.zeros((2, t.shape[0]))

        # for i in range(t.shape[0]):
        #    u[0, i] = da[i] #Insert magnitude of "da" (aileron deflection) or comment out if none
        #    u[1, i] = dr[i] #Insert magnitude of "dr" (rudder deflection) or comment out if none

        u = np.array([forced_response_inputs[4],
                      forced_response_inputs[5]])

        # Calculate response to arbitrary input
        t, y, x = control.forced_response(system, forced_response_inputs[0], u, forced_response_inputs[6], transpose=False)

        # Change dimensionless pb/(2V) and rb/(2V) to p and r
        y[1, :] = -y[1, :] + 2 * y[1, 0]
        y[2, :] = -(2 * abs(V0) * y[2, :] / b) + 2 * y[2, 0]
        y[3, :] = -(2 * abs(V0) * y[3, :] / b) + 2 * y[3, 0]

        control.damp(system, doprint=True)

        eigen_motion_error_beta = self.L2error(forced_response_inputs[8], y[0, :])
        eigen_motion_error_phi = self.L2error(forced_response_inputs[1], y[1, :])
        eigen_motion_error_p = self.L2error(forced_response_inputs[2], y[2, :])
        eigen_motion_error_r = self.L2error(forced_response_inputs[3], y[3, :])
        avg_eigen_motion_error = (eigen_motion_error_beta + eigen_motion_error_p + eigen_motion_error_phi + eigen_motion_error_r) / 4.0

        print("The Average Error Computed for the {0} is: {1}".format(self.maneuver_name, avg_eigen_motion_error))
        return avg_eigen_motion_error

if __name__ == "__main__":


    Cma = -0.5669
    Cmde = -1.2312

    # Dutch Roll
    dutch_roll_start_time = 3060
    dutch_roll_time_length = 20
    dutch_roll_maneuver_name = 'Dutch Roll'
    dutch_roll_found_optimized_stability_coeffs = np.array([0.10738637, -0.22933068, 0.21555655])

    # Spiral
    spiral_start_time = 3305
    spiral_time_length = 30
    spiral_maneuver_name = 'Spiral'
    # -0.28135457 -7.32891292 -6.86734376
    spiral_found_optimized_stability_coeffs = np.array([-0.28135457, -7.32891292, -6.86734376])

    # Aperiodic
    aperiodic_start_time = 3380
    aperiodic_time_length = 30
    aperiodic_maneuver_name = 'Aperiodic'


    # stability_coeffs_input_array = np.array([0.1348, -0.2061, -0.7500])
    # found_optimized_stability_coeffs = np.array([0.10738637, -0.22933068, 0.21555655])

    ss_sim = AsymmetricStateSpace(aperiodic_start_time, aperiodic_time_length, Cma, Cmde, aperiodic_maneuver_name)
    ss_sim.create_assymetric_ss_representation()

    # stability_coeffs_array: [Cnb, Cnr, CYb]
    # stability_coeffs_array: [+0.1348, -0.2061, -0.7500]
    # stability_coeffs_input_array = np.array([0.1348, -0.2061, -0.7500])
    #
    # func = AsymmetricStateSpace(spiral_start_time, spiral_time_length, Cma, Cmde, spiral_maneuver_name)
    # from scipy.optimize import minimize
    # res = minimize(func.compute_assymetric_ss_representation_error, stability_coeffs_input_array, method='nelder-mead',
    #                options={'disp': True, 'maxiter': 100, 'maxfev': None, 'xtol': 0.0001, 'ftol': 0.0001})
    # print(res.x)
    #
    # found_optimized_stability_coeffs = np.array([0.10738637, -0.22933068, 0.21555655])
    # opt_error_computed = AsymmetricStateSpace(dutch_roll_start_time, dutch_roll_time_length, Cma, Cmde, dutch_roll_maneuver_name)
    # opt_error_computed.compute_assymetric_ss_representation_error(stability_coeffs_input_array)