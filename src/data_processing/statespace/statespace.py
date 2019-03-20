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

#Import reference values:

# dirname = os.path.dirname(os.path.realpath(__file__))
# Cit_parStr = os.path.join(dirname, r'..\\..\\data')
# Cit_parStr = os.path.abspath(os.path.realpath(Cit_parStr))
#
from src.data_extraction.time_series_tool import TimeSeriesTool
#
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
    for time in t:
        specific_t_mdat_vals = ts_tool.get_t_specific_mdat_values(time)
        de.append(specific_t_mdat_vals['delta_e'][0])
        aoa.append(specific_t_mdat_vals['vane_AOA'][0])
        pitch.append(specific_t_mdat_vals['Ahrs1_Pitch'][0])
        V.append(specific_t_mdat_vals['Dadc1_mach'][0]*np.sqrt(1.4*287.0*specific_t_mdat_vals['Dadc1_sat'][0]))
        q.append(specific_t_mdat_vals['Ahrs1_bPitchRate'][0])
        print("At t= {0} the corresponding recorded 'black-box' data is:\n {1}".format(time, specific_t_mdat_vals))
    # print(ts_tool.get_t_specific_mdat_values(1665))
    t = np.asarray(t)
    aoa = np.asarray(aoa)
    pitch = np.asarray(pitch)
    q = np.asarray(q)
    V = np.asarray(V)
    u = V - V[0]
    # Initial conditions
    x0 = np.array([[0.0],
                   [0.0],
                   [pitch[0]],
                   [0.0]])

    return t, de, aoa, pitch, q, x0, u, V[0]

phugoid = maneuver_vals(2860, 60)
short = maneuver_vals(2770, 60)

#State-space representation of symmetric EOM:

#C1*x' + C2*x + C3*u = 0

V0 = phugoid[7]

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

#t = np.linspace(0.0, 300.0, num=301)

# u = np.zeros(t.shape[0])
#
# #length of pulse
# #tpulse = 12.0 #phugoid
#
# for i in range(t.shape[0]):
#     u[i] = de[i] #Insert magnitude of "de" (elevator deflection)

# t, de, aoa, pitch, q, x0, u
    
#Calculate response to arbitrary input
t, y, x = control.forced_response(system, phugoid[0], phugoid[1], phugoid[5], transpose=False)

#Change dimensionless û and qc/V to u and q
y[0, :] = phugoid[7]*y[0, :]
y[3, :] = phugoid[7]*y[3, :]/c
y[1, :] = y[1, :] + phugoid[2][0]

fig = plt.figure(figsize=(12,9))
fig.suptitle('Phugoid',fontsize=16)

ax1 = fig.add_subplot(221)
ax1.plot(t, y[0, :])
ax1.plot(t, phugoid[6])
ax1.set_xlabel("Time [s]")
ax1.set_ylabel("u (disturbance in velocity) [m/s]")

ax2 = fig.add_subplot(222)
ax2.plot(t, y[1, :])
#alpha
ax2.plot(t, phugoid[2])
ax2.set_xlabel("Time [s]")
ax2.set_ylabel("Alpha (AoA) [deg]")

ax3 = fig.add_subplot(223)
ax3.plot(t, y[3, :])
#theta
ax3.plot(t, phugoid[3])
ax3.set_xlabel("Time [s]")
ax3.set_ylabel("Theta (Pitch Angle) [deg]")

ax4 = fig.add_subplot(224)
ax4.plot(t, y[3, :])
#q
ax4.plot(t, phugoid[4])
ax4.set_xlabel("Time [s]")
ax4.set_ylabel("q (Pitch Rate) [deg/s]")

#SHORT-PERIOD

V0 = short[7]

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
t, y, x = control.forced_response(system, short[0], short[1], short[5], transpose=False)

#Change dimensionless û and qc/V to u and q
y[0, :] = short[7]*y[0, :]
y[3, :] = short[7]*y[3, :]/c
y[1, :] = y[1, :] + short[2][0]

fig2 = plt.figure(figsize=(12,9))
fig2.suptitle('Short Period',fontsize=16)

ax1 = fig2.add_subplot(221)
ax1.plot(t, y[0, :])
ax1.plot(t, short[6])
ax1.set_xlabel("Time [s]")
ax1.set_ylabel("u (disturbance in velocity) [m/s]")

ax2 = fig2.add_subplot(222)
ax2.plot(t, y[1, :])
#alpha
ax2.plot(t, short[2])
ax2.set_xlabel("Time [s]")
ax2.set_ylabel("Alpha (AoA) [deg]")

ax3 = fig2.add_subplot(223)
ax3.plot(t, y[3, :])
#theta
ax3.plot(t, short[3])
ax3.set_xlabel("Time [s]")
ax3.set_ylabel("Theta (Pitch Angle) [deg]")

ax4 = fig2.add_subplot(224)
ax4.plot(t, y[3, :])
#q
ax4.plot(t, short[4])
ax4.set_xlabel("Time [s]")
ax4.set_ylabel("q (Pitch Rate) [deg/s]")

plt.show()

control.damp(system, doprint=True)
