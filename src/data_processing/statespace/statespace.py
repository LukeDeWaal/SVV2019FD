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

#Import reference values:

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

#These reference values are missing from Cit_par:

Cma = -0.5626
Cmde = -1.1642

#Import data for given time step
ts_tool = TimeSeriesTool()
def maneuver_vals(time_start, time_length):
    t = list(range(time_start,time_start+time_length))
    de = []
    aoa = []
    pitch = []
    q = []
    V = []
    u = []
    w = []
    for time in t:
        specific_t_mdat_vals = ts_tool.get_t_specific_mdat_values(time)
        de.append(specific_t_mdat_vals['delta_e'][0])
        aoa.append(specific_t_mdat_vals['vane_AOA'][0])
        pitch.append(specific_t_mdat_vals['Ahrs1_Pitch'][0])
        V.append(specific_t_mdat_vals['Dadc1_tas'][0])
        q.append(specific_t_mdat_vals['Ahrs1_bPitchRate'][0])
        u.append(specific_t_mdat_vals['Dadc1_tas'][0] * math.cos(specific_t_mdat_vals['vane_AOA'][0] * np.pi/180.0))
        w.append(specific_t_mdat_vals['Dadc1_tas'][0] * math.sin(specific_t_mdat_vals['vane_AOA'][0] * np.pi/180.0))
    # print(ts_tool.get_t_specific_mdat_values(1665))
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
    x0 = np.array([[0.0],
                   [0.0],
                   [pitch[0]],
                   [q[0]]])

    return t, de, aoa, pitch, q, x0, u, V[0], w

def get_flight_conditions(t):
    data = Data(r'FlightData.mat')
    mat_data = data.get_mat().get_data()
    time = mat_data['time']
    rh_fu = mat_data['rh_engine_FU']
    lh_fu = mat_data['lh_engine_FU']

    alt = mat_data['Dadc1_alt']

    for idx, t_i in enumerate(time):
        if time[idx] < t <= time[idx+1]:
            break

    m = get_weight_at_t(t, time, rh_fu, lh_fu)/9.80665

    h = alt[idx]
    rho = ISA(h)[2]

    mub = m / (rho * S * b)
    muc = m / (rho * S * c)

    return mub, muc, m, h, rho

def L2error(x_exact: np.array, x_numerical: np.array):
    error = 0.0
    if (x_exact.shape[0] != x_numerical.shape[0]):
        print("Vectors must be of the same size!")
        return 1
    for i in range(x_exact.shape[0]):
        error += (x_numerical[i] - x_exact[i])*(x_numerical[i] - x_exact[i])

    error = math.sqrt(error)/x_exact.shape[0]
    return error

phugoid = maneuver_vals(2836, 200)
short = maneuver_vals(2760, 50)

g = 9.80665

#State-space representation of symmetric EOM:

#C1*x' + C2*x + C3*u = 0

V0 = phugoid[7]

mub, muc, m, h, rho = get_flight_conditions(2836)
mub = float(mub[0])
muc = float(muc[0])
m = float(m[0])
h = float(h[0])
rho = float(rho[0])

CX0 = m * g * sin(phugoid[3][0]*np.pi/180.0) / (0.5 * rho * (V0 ** 2) * S)
CZ0 = -m * g * cos(phugoid[3][0]*np.pi/180.0) / (0.5 * rho * (V0 ** 2) * S)

c = 2.0569

C1 = np.array([[-2.0*muc*c/V0, 0.0, 0.0, 0.0],
               [0.0, (CZadot - 2.0*muc)*c/V0, 0.0, 0.0],
               [0.0, 0.0, -c/V0, 0.0],
               [0.0, Cmadot*c/V0, 0.0, -2.0*muc*KY2*c/V0]])

C2 = np.array([[CXu, CXa, CZ0, CXq], 
               [CZu, CZa, -CX0, (CZq + 2.0*muc)], 
               [0.0, 0.0, 0.0, 1.0], 
               [Cmu, Cma, 0.0, Cmq]])

C3 = np.array([[CXde],
               [CZde],
               [0.0],
               [Cmde]])

#x' = A*x + B*u
#y  = C*x + D*u

A = -np.matmul(np.linalg.inv(C1), C2)
B = -np.matmul(np.linalg.inv(C1), C3)
C = np.identity(4) #y = x, meaning we output the state
D = np.array([[0.0], [0.0], [0.0], [0.0]])

#Make control.ss state-space

system = control.ss(A, B, C, D)

#t = np.linspace(0.0, 300.0, num=301)

# u = np.zeros(t.shape[0])
#
# #length of pulse
# #tpulse = 12.0 #phugoid
#
# for i in range(t.shape[0]):
#     u[i] = de[i] #Insert magnitude of "de" (elevator deflection)
    
#Calculate response to arbitrary input
# 0=t, 1=u, 2=aoa, 3=pitch, 4=q 5=x0

t, y, x = control.forced_response(system, phugoid[0], phugoid[1], phugoid[5], transpose=False)

#Change dimensionless qc/V to q
y[2, :] = y[2, :] + phugoid[3][0]
y[3, :] = phugoid[7]*y[3, :]/c

#y[1, 0] = phugoid[2][0]
y[3, 0] = 0.0

fig = plt.figure(figsize=(12,9))
fig.suptitle('Phugoid',fontsize=16)

ax1 = fig.add_subplot(221)
ax1.plot(t, y[0, :])
ax1.plot(t, phugoid[6])
ax1.set_xlabel("Time [s]")
ax1.set_ylabel("u (x-dir. disturbance in velocity) [m/s]")
print("Phugoid Error in u:")
print(L2error(phugoid[6], y[0, :]))


ax2 = fig.add_subplot(222)
ax2.plot(t, y[1, :])
#alpha
ax2.plot(t, phugoid[8])
ax2.set_xlabel("Time [s]")
ax2.set_ylabel("w (z-dir. disturbance in velocity) [m/s]")
print("Phugoid Error in w:")
print(L2error(phugoid[8], y[1, :]))

ax3 = fig.add_subplot(223)
ax3.plot(t, y[2, :])
#theta
ax3.plot(t, phugoid[3])
ax3.set_xlabel("Time [s]")
ax3.set_ylabel("Theta (Pitch Angle) [deg]")
print("Phugoid Error in Theta:")
print(L2error(phugoid[3], y[2, :]))

ax4 = fig.add_subplot(224)
ax4.plot(t, y[3, :])
#q
ax4.plot(t, phugoid[4])
ax4.set_xlabel("Time [s]")
ax4.set_ylabel("q (Pitch Rate) [deg/s]")
print("Phugoid Error in q:")
print(L2error(phugoid[4], y[3, :]))

#SHORT-PERIOD

V0 = short[7]

mub, muc, m, h, rho = get_flight_conditions(2750)
mub = float(mub[0])
muc = float(muc[0])
m = float(m[0])
h = float(h[0])
rho = float(rho[0])

CX0 = m * g * sin(short[3][0]*np.pi/180.0) / (0.5 * rho * (V0 ** 2) * S)
CZ0 = -m * g * cos(short[3][0]*np.pi/180.0) / (0.5 * rho * (V0 ** 2) * S)

C1 = np.array([[-2.0*muc*c/V0, 0.0, 0.0, 0.0],
               [0.0, (CZadot - 2.0*muc)*c/V0, 0.0, 0.0],
               [0.0, 0.0, -c/V0, 0.0],
               [0.0, Cmadot*c/V0, 0.0, -2.0*muc*KY2*c/V0]])

C2 = np.array([[CXu, CXa, CZ0, CXq],
               [CZu, CZa, -CX0, (CZq + 2.0*muc)],
               [0.0, 0.0, 0.0, 1.0],
               [Cmu, Cma, 0.0, Cmq] ])

C3 = np.array([[CXde],
               [CZde],
               [0.0],
               [Cmde]])

#x' = A*x + B*u
#y  = C*x + D*u

A = -np.matmul(np.linalg.inv(C1), C2)
B = -np.matmul(np.linalg.inv(C1), C3)
C = np.identity(4) #y = x, meaning we output the state
D = np.array([[0.0], [0.0], [0.0], [0.0]])

#Make control.ss state-space

system = control.ss(A, B, C, D)

#Calculate response to arbitrary input
# 0=t, 1=u, 2=aoa, 3=pitch, 4=q 5=x0

t, y, x = control.forced_response(system, short[0], short[1], short[5], transpose=False)

#Change dimensionless û and qc/V to u and q
#y[0, :] = short[7]*y[0, :]
y[2, :] = y[2, :] + short[3][0]
y[3, :] = short[7]*y[3, :]/c

y[2, 0] = 0.0
y[3, 0] = 0.0

fig2 = plt.figure(figsize=(12,9))
fig2.suptitle('Short Period',fontsize=16)

ax1 = fig2.add_subplot(221)
ax1.plot(t, y[0, :])
ax1.plot(t, short[6])
ax1.set_xlabel("Time [s]")
ax1.set_ylabel("u (x-dir. disturbance in velocity) [m/s]")
print("Short Error in u:")
short_error_u = L2error(short[6], y[0, :])
print(short_error_u)

ax2 = fig2.add_subplot(222)
ax2.plot(t, y[1, :])
#alpha
ax2.plot(t, short[8])
ax2.set_xlabel("Time [s]")
ax2.set_ylabel("w (z-dir. disturbance in velocity) [m/s]")
print("Short Error in w:")
short_error_w = L2error(short[8], y[1, :])
print(short_error_w)

ax3 = fig2.add_subplot(223)
ax3.plot(t, y[2, :])
#theta
ax3.plot(t, short[3])
ax3.set_xlabel("Time [s]")
ax3.set_ylabel("Theta (Pitch Angle) [deg]")
print("Short Error in theta:")
short_error_th = L2error(short[3], y[2, :])
print(short_error_th)

ax4 = fig2.add_subplot(224)
ax4.plot(t, y[3, :])
#q
ax4.plot(t, short[4])
ax4.set_xlabel("Time [s]")
ax4.set_ylabel("q (Pitch Rate) [deg/s]")
print("Short Error in q:")
short_error_q = L2error(short[4], y[3, :])
print(short_error_q)

print("Avg. Short-Period Error:")
avg_short_error = (short_error_q + short_error_th + short_error_u + short_error_w)/4
print(avg_short_error)

plt.show()

control.damp(system, doprint=True)
