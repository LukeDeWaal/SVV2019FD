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

class SymmetricalStateSpace:

    def __init__(self, time_start, time_length, Cma, Cmde, maneuver_name):
        self.time_start = time_start
        self.time_length= time_length
        self.time_end = self.time_start + self.time_length
        self.Cma = Cma
        self.Cmde = Cmde
        self.ts_tool = TimeSeriesTool()
        self.maneuver_name = maneuver_name
        # self.data_tool = Data()

    def maneuver_vals(self, t, dt):
        t = list(range(t, t + dt))
        de = []
        aoa = []
        pitch = []
        q = []
        V = []
        u = []
        w = []
        for time in t:
            specific_t_mdat_vals = self.ts_tool.get_t_specific_mdat_values(time)
            de.append(specific_t_mdat_vals['delta_e'][0])
            aoa.append(specific_t_mdat_vals['vane_AOA'][0])
            pitch.append(specific_t_mdat_vals['Ahrs1_Pitch'][0])
            V.append(specific_t_mdat_vals['Dadc1_tas'][0])
            q.append(specific_t_mdat_vals['Ahrs1_bPitchRate'][0])
            u.append(
                specific_t_mdat_vals['Dadc1_tas'][0] * math.cos(specific_t_mdat_vals['vane_AOA'][0] * np.pi / 180.0))
            w.append(
                specific_t_mdat_vals['Dadc1_tas'][0] * math.sin(specific_t_mdat_vals['vane_AOA'][0] * np.pi / 180.0))
        t = np.asarray(t)
        aoa = np.asarray(aoa)
        pitch = np.asarray(pitch)
        q = np.asarray(q)
        V = np.asarray(V)
        u = np.asarray(u)
        w = np.asarray(w)

        u = u - u[0]
        w = w - w[0]

        # Initial conditions
        x0 = np.array([[0.0], [0.0], [pitch[0]], [q[0]]])

        return t, de, aoa, pitch, q, x0, u, V[0], w

    def get_flight_conditions(self, t):
        data = Data(r'FlightData.mat')
        mat_data = data.get_mat().get_data()
        time = mat_data['time']
        rh_fu = mat_data['rh_engine_FU']
        lh_fu = mat_data['lh_engine_FU']

        alt = mat_data['Dadc1_alt']

        for idx, t_i in enumerate(time):
            if time[idx] < t <= time[idx + 1]:
                break

        m = get_weight_at_t(t, time, rh_fu, lh_fu) / 9.80665

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

    def create_ss_rep_of_sym_eom(self):
        # Cmq = -14.32232691
        forced_response_inputs = self.maneuver_vals(self.time_start, self.time_length)
        g = 9.80665
        V0 = forced_response_inputs[7]
        # short = maneuver_vals(2760, 50)
        mub, muc, m, h, rho = self.get_flight_conditions(self.time_start)
        mub = float(mub[0])
        muc = float(muc[0])
        m = float(m[0])
        h = float(h[0])
        rho = float(rho[0])

        CX0 = m * g * sin(forced_response_inputs[3][0] * np.pi / 180.0) / (0.5 * rho * (V0 ** 2) * S)
        CZ0 = -m * g * cos(forced_response_inputs[3][0] * np.pi / 180.0) / (0.5 * rho * (V0 ** 2) * S)

        c = 2.0569

        C1 = np.array([[-2.0 * muc * c / V0, 0.0, 0.0, 0.0],
                       [0.0, (CZadot - 2.0 * muc) * c / V0, 0.0, 0.0],
                       [0.0, 0.0, -c / V0, 0.0],
                       [0.0, Cmadot * c / V0, 0.0, -2.0 * muc * KY2 * c / V0]])

        C2 = np.array([[CXu, CXa, CZ0, CXq],
                       [CZu, CZa, -CX0, (CZq + 2.0 * muc)],
                       [0.0, 0.0, 0.0, 1.0],
                       [Cmu, Cma, 0.0, Cmq]])

        C3 = np.array([[CXde],
                       [CZde],
                       [0.0],
                       [self.Cmde]])

        # x' = A*x + B*u
        # y  = C*x + D*u

        A = -np.matmul(np.linalg.inv(C1), C2)
        B = -np.matmul(np.linalg.inv(C1), C3)
        C = np.identity(4)  # y = x, meaning we output the state
        D = np.array([[0.0], [0.0], [0.0], [0.0]])

        # Make control.ss state-space

        system = control.ss(A, B, C, D)

        t, y, x = control.forced_response(system, forced_response_inputs[0], forced_response_inputs[1], forced_response_inputs[5], transpose=False)

        # Change dimensionless qc/V to q
        y[2, :] = y[2, :] + forced_response_inputs[3][0]
        y[3, :] = forced_response_inputs[7] * y[3, :] / c

        # y[1, 0] = phugoid[2][0]
        y[3, 0] = 0.0

        fig = plt.figure(figsize=(12, 9))
        # fig = plt.figure(figsize=(10, 12))
        # fig.suptitle(self.maneuver_name, fontsize=16)

        ax1 = fig.add_subplot(221)
        ax1.plot(t, y[0, :], label="Simulation Response") # simulation response curve plot
        ax1.plot(t, forced_response_inputs[6], label="Test Flight Data Response") # test flight data response curve plot
        plt.legend(loc="best")
        ax1.set_xlabel("Time [s]")
        ax1.set_ylabel("u (x-dir. disturbance in velocity) [m/s]")
        ax1.grid()
        print("{0} Error in u:".format(self.maneuver_name))
        eigen_motion_error_u = self.L2error(forced_response_inputs[6], y[0, :])
        print(eigen_motion_error_u)

        ax2 = fig.add_subplot(222)
        ax2.plot(t, y[1, :], label="Simulation Response")
        # alpha
        ax2.plot(t, forced_response_inputs[8], label="Test Flight Data Response")
        plt.legend(loc="best")
        ax2.set_xlabel("Time [s]")
        ax2.set_ylabel("w (z-dir. disturbance in velocity) [m/s]")
        ax2.grid()
        print("{0} Error in w:".format(self.maneuver_name))
        eigen_motion_error_w = self.L2error(forced_response_inputs[8], y[1, :])
        print(eigen_motion_error_w)

        ax3 = fig.add_subplot(223)
        ax3.plot(t, y[2, :], label="Simulation Response")
        # theta
        ax3.plot(t, forced_response_inputs[3], label="Test Flight Data Response")
        plt.legend(loc="best")
        ax3.set_xlabel("Time [s]")
        ax3.set_ylabel("Theta (Pitch Angle) [deg]")
        ax3.grid()
        print("{0} Error in Theta:".format(self.maneuver_name))
        eigen_motion_error_th = self.L2error(forced_response_inputs[3], y[2, :])
        print(eigen_motion_error_th)

        ax4 = fig.add_subplot(224)
        ax4.plot(t, y[3, :], label="Simulation Response")
        # q
        ax4.plot(t, forced_response_inputs[4], label="Test Flight Data Response")
        plt.legend(loc="best")
        ax4.set_xlabel("Time [s]")
        ax4.set_ylabel("q (Pitch Rate) [deg/s]")
        ax4.grid()
        print("{0} Error in q:".format(self.maneuver_name))
        eigen_motion_error_q = self.L2error(forced_response_inputs[4], y[3, :])
        print(eigen_motion_error_q)

        print("Avg. {0} Error:".format(self.maneuver_name))
        avg_eigen_motion_error = (eigen_motion_error_q + eigen_motion_error_th + eigen_motion_error_w + eigen_motion_error_u) / 4
        print(avg_eigen_motion_error)
        # fig.suptitle(self.maneuver_name, fontsize=16)
        fig.suptitle(self.maneuver_name)
        # plt.tight_layout()

        # plt.legend(loc="best")
        plt.show()

    def compute_ss_rep_of_sym_eom(self, array_input):
        forced_response_inputs = self.maneuver_vals(int(array_input[1]), int(array_input[2]))

        g = 9.80665
        V0 = forced_response_inputs[7]
        # short = maneuver_vals(2760, 50)
        mub, muc, m, h, rho = self.get_flight_conditions(int(array_input[1]))
        mub = float(mub[0])
        muc = float(muc[0])
        m = float(m[0])
        h = float(h[0])
        rho = float(rho[0])

        CX0 = m * g * sin(forced_response_inputs[3][0] * np.pi / 180.0) / (0.5 * rho * (V0 ** 2) * S)
        CZ0 = -m * g * cos(forced_response_inputs[3][0] * np.pi / 180.0) / (0.5 * rho * (V0 ** 2) * S)

        c = 2.0569

        C1 = np.array([[-2.0 * muc * c / V0, 0.0, 0.0, 0.0],
                       [0.0, (CZadot - 2.0 * muc) * c / V0, 0.0, 0.0],
                       [0.0, 0.0, -c / V0, 0.0],
                       [0.0, Cmadot * c / V0, 0.0, -2.0 * muc * KY2 * c / V0]])

        C2 = np.array([[CXu, CXa, CZ0, CXq],
                       [CZu, CZa, -CX0, (CZq + 2.0 * muc)],
                       [0.0, 0.0, 0.0, 1.0],
                       [Cmu, Cma, 0.0, array_input[0]]])

        C3 = np.array([[CXde],
                       [CZde],
                       [0.0],
                       [self.Cmde]])

        # x' = A*x + B*u
        # y  = C*x + D*u

        A = -np.matmul(np.linalg.inv(C1), C2)
        B = -np.matmul(np.linalg.inv(C1), C3)
        C = np.identity(4)  # y = x, meaning we output the state
        D = np.array([[0.0], [0.0], [0.0], [0.0]])

        # Make control.ss state-space

        system = control.ss(A, B, C, D)

        t, y, x = control.forced_response(system, forced_response_inputs[0], forced_response_inputs[1], forced_response_inputs[5], transpose=False)

        # Change dimensionless qc/V to q
        y[2, :] = y[2, :] + forced_response_inputs[3][0]
        y[3, :] = forced_response_inputs[7] * y[3, :] / c

        # y[1, 0] = phugoid[2][0]
        y[3, 0] = 0.0

        phugoid_error_u = self.L2error(forced_response_inputs[6], y[0, :])
        phugoid_error_w = self.L2error(forced_response_inputs[8], y[1, :])
        phugoid_error_th = self.L2error(forced_response_inputs[3], y[2, :])
        phugoid_error_q = self.L2error(forced_response_inputs[4], y[3, :])
        avg_phugoid_error = (phugoid_error_q + phugoid_error_th + phugoid_error_w + phugoid_error_u) / 4
        # eigen_motion_error_u = self.L2error(forced_response_inputs[6], y[0, :])
        # eigen_motion_error_u_array = np.array([eigen_motion_error_u])
        # return eigen_motion_error_u
        return avg_phugoid_error

if __name__ == "__main__":
    # IMPORTANT NOTE: The minimum error computed via the Nelder-Simplex optimization is Cmq= -14.32232691

    Cma = -0.5626
    Cmde = -1.23048801

    phugoid_time_start = 2836
    phugoid_time_length = 200
    phugoid_ss_sim = SymmetricalStateSpace(phugoid_time_start, phugoid_time_length, Cma, Cmde, "Phugoid")
    phugoid_ss_sim.create_ss_rep_of_sym_eom()

    # short_period_time_start = 2760
    # short_period_time_length = 50
    # short_ss_sim = SymmetricalStateSpace(short_period_time_start, short_period_time_length, Cma, Cmde, "Short-Period")
    # short_ss_sim.create_ss_rep_of_sym_eom()

    from scipy.optimize import minimize_scalar
    #
    #
    # Cmq_array_test = np.array([-8.79415, phugoid_time_start, phugoid_time_length])
    # Cmq_array_test_scalar = -8.79415
    # func = SymmetricalStateSpace(phugoid_time_start, phugoid_time_length, Cma, Cmde, "Phugoid")
    # from scipy.optimize import minimize
    # res = minimize(func.compute_ss_rep_of_sym_eom, Cmq_array_test, method='nelder-mead',
    #                options={'disp': True, 'maxiter': 100, 'maxfev': None, 'xtol': 0.0001, 'ftol': 0.0001})
    # print(res.x)

