# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 15:40:15 2019

@author: Matheus
"""
import control
import control.matlab
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import math

#Import reference values:

from data.Cit_par import *
from src.data_extraction.time_series_tool import TimeSeriesTool

from src.data_extraction.time_series_tool import TimeSeriesTool
from src.data_extraction.data_main import Data
from src.data_processing.get_weight import get_weight_at_t
from src.data_processing.aerodynamics import ISA

#These reference values are missing from Cit_par:

Cma = -0.5669
#Cmde = -1.1642
Cmde = -1.2312

#Import data for given time step
ts_tool = TimeSeriesTool()

def maneuver_vals(time_start, length,name):
    t = list(range(time_start, time_start + length))
    da = []
    dr = []
    phi = []
    p = []
    r = []
    V = []
    yawRate = []
    beta = [0.0]

    for time in t:
        specific_t_mdat_vals = ts_tool.get_t_specific_mdat_values(time)
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

    charPlot = plt.figure(figsize=(12,9))
    charPlot.suptitle('Characteristic Plot '+name, fontsize=20)

    ax1 = charPlot.add_subplot(211)
    ax1.plot(t, da, label='aileron deflection')
    ax1.plot(t, dr, label = 'rudder deflection')
    ax1.legend(loc='upper right',fontsize=14)
    ax1.set_ylabel("da, dr [deg]", fontsize=20)

    ax2 = charPlot.add_subplot(212)
    ax2.plot(t, beta, label='beta')
    ax2.plot(t, phi, label='phi')
    ax2.plot(t, p, label='p')
    ax2.plot(t, r, label='r')
    ax2.legend(loc='upper right',fontsize=14)
    ax2.set_xlabel("Time [s]", fontsize=20)
    ax2.set_ylabel("beta, phi [deg], p, r [deg/s]", fontsize=20)

    charPlot.savefig('CharacterPlot_'+name)

    return t, phi, p, r, da, dr, x0, V[0], beta

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

dutch = maneuver_vals(3060, 20, 'Dutch Roll')
spiral = maneuver_vals(3305, 30, 'Spiral')
aperiodic = maneuver_vals(3380, 30, 'Aperiodic')

g = 9.80665

#DUTCH ROLL:
#State-space representation of asymmetric EOM:

#C1*x' + C2*x + C3*u = 0

V0 = dutch[7]

mub, muc, m, h, rho = get_flight_conditions(3060)
mub = float(mub[0])
muc = float(muc[0])
m = float(m[0])
h = float(h[0])
rho = float(rho[0])

CX0 = m * g * sin(dutch[3][0]*np.pi/180.0) / (0.5 * rho * (V0 ** 2) * S)
CZ0 = -m * g * cos(dutch[3][0]*np.pi/180.0) / (0.5 * rho * (V0 ** 2) * S)

c = 2.0569

C1 = np.array([[(CYbdot - 2.0*mub)*b/V0, 0.0, 0.0, 0.0], 
               [0.0, -0.5*b/V0, 0.0, 0.0], 
               [0.0, 0.0, -4.0*mub*KX2*b/V0, 4.0*mub*KXZ*b/V0],
               [Cnbdot*b/V0, 0.0, 4.0*mub*KXZ*b/V0, -4.0*mub*KZ2*b/V0]])

C2 = np.array([[-CYb, -CL, -CYp, -(CYr - 4.0*mub)],
               [0.0, 0.0, -1.0, 0.0],
               [-Clb, 0.0, -Clp, -Clr],
               [-Cnb, 0.0, -Cnp, -Cnr] ])

C3 = np.array([[-CYda, -CYdr],
               [0.0, 0.0],
               [-Clda, -Cldr],
               [-Cnda, -Cndr]])

#x' = A*x + B*u
#y  = C*x + D*u

# now u = [da]
#         [dr]

A = np.matmul(np.linalg.inv(C1), C2)
B = np.matmul(np.linalg.inv(C1), C3)
C = np.identity(4) #y = x, meaning we output the state
D = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])

#Make control.ss state-space

system = control.ss(A, B, C, D)

# t = np.linspace(0.0, 200.0, num = 201)

#Input:
#u = np.zeros((2, t.shape[0]))

#for i in range(t.shape[0]):
#    u[0, i] = da[i] #Insert magnitude of "da" (aileron deflection) or comment out if none
#    u[1, i] = dr[i] #Insert magnitude of "dr" (rudder deflection) or comment out if none

u = np.array([dutch[4],
              dutch[5]]);

#Calculate response to arbitrary input    
t, y, x = control.forced_response(system, dutch[0], u, dutch[6], transpose=False)

#Change dimensionless pb/(2V) and rb/(2V) to p and r
# y[1, :] = -y[1, :]
y[2, :] = 2*abs(V0)*y[2, :]/b
y[3, :] = 2*abs(V0)*y[3, :]/b

y[2, 0] = dutch[2][0]
y[3, 0] = dutch[3][0]

control.damp(system, doprint=True)

fig = plt.figure(figsize=(12,9))
fig.suptitle('Dutch Roll',fontsize=16)

ax1 = fig.add_subplot(221)
ax1.plot(t, y[0, :])
ax1.plot(t, dutch[8], color='tab:orange',linestyle='--')
ax1.set_xlabel("Time [s]")
ax1.set_ylabel("beta (yaw angle) [deg]")
print("Dutch Roll Error in Beta:")
dutch_error_beta = L2error(dutch[8], y[0, :])
print(dutch_error_beta)

ax2 = fig.add_subplot(222)
ax2.plot(t, y[1, :])
ax2.plot(t, dutch[1], color='tab:orange',linestyle='--')
ax2.set_xlabel("Time [s]")
ax2.set_ylabel("phi (roll angle) [deg]")
print("Dutch Roll Error in Phi:")
dutch_error_phi = L2error(dutch[1], y[1, :])
print(dutch_error_phi)

ax3 = fig.add_subplot(223)
ax3.plot(t, y[2, :])
ax3.plot(t, dutch[2], color='tab:orange',linestyle='--')
ax3.set_xlabel("Time [s]")
ax3.set_ylabel("p (roll rate) [deg/sec]")
print("Dutch Roll Error in p:")
dutch_error_p = L2error(dutch[2], y[2, :])
print(dutch_error_p)

ax4 = fig.add_subplot(224)
ax4.plot(t, y[3, :])
ax4.plot(t, dutch[3], color='tab:orange',linestyle='--')
ax4.set_xlabel("Time [s]")
ax4.set_ylabel("r (yaw rate) [deg/s]")
print("Dutch Roll Error in r:")
dutch_error_r = L2error(dutch[3], y[3, :])
print(dutch_error_r)

print("Avg. Dutch Error:")
avg_dutch_error = (dutch_error_beta + dutch_error_p + dutch_error_phi + dutch_error_r)/4.0
print(avg_dutch_error)

#SPIRAL:
#State-space representation of asymmetric EOM:

#C1*x' + C2*x + C3*u = 0

V0 = spiral[7]

mub, muc, m, h, rho = get_flight_conditions(3305)
mub = float(mub[0])
muc = float(muc[0])
m = float(m[0])
h = float(h[0])
rho = float(rho[0])

CX0 = m * g * sin(spiral[3][0]*np.pi/180.0) / (0.5 * rho * (V0 ** 2) * S)
CZ0 = -m * g * cos(spiral[3][0]*np.pi/180.0) / (0.5 * rho * (V0 ** 2) * S)

c = 2.0569

C1 = np.array([[(CYbdot - 2.0*mub)*b/V0, 0.0, 0.0, 0.0],
               [0.0, -0.5*b/V0, 0.0, 0.0],
               [0.0, 0.0, -4.0*mub*KX2*b/V0, 4.0*mub*KXZ*b/V0],
               [Cnbdot*b/V0, 0.0, 4.0*mub*KXZ*b/V0, -4.0*mub*KZ2*b/V0]])

C2 = np.array([[-CYb, -CL, -CYp, -(CYr - 4.0*mub)],
               [0.0, 0.0, -1.0, 0.0],
               [-Clb, 0.0, -Clp, -Clr],
               [-Cnb, 0.0, -Cnp, -Cnr] ])

C3 = np.array([[-CYda, -CYdr],
               [0.0, 0.0],
               [-Clda, -Cldr],
               [-Cnda, -Cndr]])

#x' = A*x + B*u
#y  = C*x + D*u

# now u = [da]
#         [dr]

A = np.matmul(np.linalg.inv(C1), C2)
B = np.matmul(np.linalg.inv(C1), C3)
C = np.identity(4) #y = x, meaning we output the state
D = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])

#Make control.ss state-space

system = control.ss(A, B, C, D)

# t = np.linspace(0.0, 200.0, num = 201)

#Input:
#u = np.zeros((2, t.shape[0]))

#for i in range(t.shape[0]):
#    u[0, i] = da[i] #Insert magnitude of "da" (aileron deflection) or comment out if none
#    u[1, i] = dr[i] #Insert magnitude of "dr" (rudder deflection) or comment out if none

u = np.array([spiral[4],
              spiral[5]]);

#Calculate response to arbitrary input
t, y, x = control.forced_response(system, spiral[0], u, spiral[6], transpose=False)

#Change dimensionless pb/(2V) and rb/(2V) to p and r
# y[1, :] = -y[1, :]
y[2, :] = 2*abs(V0)*y[2, :]/b
y[3, :] = 2*abs(V0)*y[3, :]/b

control.damp(system, doprint=True)

fig2 = plt.figure(figsize=(12,9))
fig2.suptitle('Spiral',fontsize=16)

ax1 = fig2.add_subplot(221)
ax1.plot(t, y[0, :])
ax1.plot(t, spiral[8], color='tab:orange',linestyle='--')
ax1.set_xlabel("Time [s]")
ax1.set_ylabel("beta (yaw angle) [deg]")
print("Spiral Roll Error in Beta:")
spiral_error_beta = L2error(spiral[8], y[0, :])
print(spiral_error_beta)

ax2 = fig2.add_subplot(222)
ax2.plot(t, y[1, :])
ax2.plot(t, spiral[1], color='tab:orange',linestyle='--')
ax2.set_xlabel("Time [s]")
ax2.set_ylabel("phi (roll angle) [deg]")
print("Spiral Roll Error in Phi:")
spiral_error_phi = L2error(spiral[1], y[1, :])
print(spiral_error_phi)

ax3 = fig2.add_subplot(223)
ax3.plot(t, y[2, :])
ax3.plot(t, spiral[2], color='tab:orange',linestyle='--')
ax3.set_xlabel("Time [s]")
ax3.set_ylabel("p (roll rate) [deg/sec]")
print("Spiral Roll Error in p:")
spiral_error_p = L2error(spiral[2], y[2, :])
print(spiral_error_p)

ax4 = fig2.add_subplot(224)
ax4.plot(t, y[3, :])
ax4.plot(t, spiral[3], color='tab:orange',linestyle='--')
ax4.set_xlabel("Time [s]")
ax4.set_ylabel("r (yaw rate) [deg/s]")
print("Spiral Roll Error in r:")
spiral_error_r = L2error(spiral[3], y[3, :])
print(spiral_error_r)

print("Avg. Spiral Error:")
avg_spiral_error = (spiral_error_beta + spiral_error_p + spiral_error_phi + spiral_error_r)/4.0
print(avg_spiral_error)

#APERIODIC ROLL:
#State-space representation of asymmetric EOM:

#C1*x' + C2*x + C3*u = 0

V0 = aperiodic[7]

mub, muc, m, h, rho = get_flight_conditions(3380)
mub = float(mub[0])
muc = float(muc[0])
m = float(m[0])
h = float(h[0])
rho = float(rho[0])

CX0 = m * g * sin(aperiodic[3][0]*np.pi/180.0) / (0.5 * rho * (V0 ** 2) * S)
CZ0 = -m * g * cos(aperiodic[3][0]*np.pi/180.0) / (0.5 * rho * (V0 ** 2) * S)

c = 2.0569

C1 = np.array([[(CYbdot - 2.0*mub)*b/V0, 0.0, 0.0, 0.0],
               [0.0, -0.5*b/V0, 0.0, 0.0],
               [0.0, 0.0, -4.0*mub*KX2*b/V0, 4.0*mub*KXZ*b/V0],
               [Cnbdot*b/V0, 0.0, 4.0*mub*KXZ*b/V0, -4.0*mub*KZ2*b/V0]])

C2 = np.array([[-CYb, -CL, -CYp, -(CYr - 4.0*mub)],
               [0.0, 0.0, -1.0, 0.0],
               [-Clb, 0.0, -Clp, -Clr],
               [-Cnb, 0.0, -Cnp, -Cnr] ])

C3 = np.array([[-CYda, -CYdr],
               [0.0, 0.0],
               [-Clda, -Cldr],
               [-Cnda, -Cndr]])

#x' = A*x + B*u
#y  = C*x + D*u

# now u = [da]
#         [dr]

A = np.matmul(np.linalg.inv(C1), C2)
B = np.matmul(np.linalg.inv(C1), C3)
C = np.identity(4) #y = x, meaning we output the state
D = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])

#Make control.ss state-space

system = control.ss(A, B, C, D)

# t = np.linspace(0.0, 200.0, num = 201)

#Input:
#u = np.zeros((2, t.shape[0]))

#for i in range(t.shape[0]):
#    u[0, i] = da[i] #Insert magnitude of "da" (aileron deflection) or comment out if none
#    u[1, i] = dr[i] #Insert magnitude of "dr" (rudder deflection) or comment out if none

u = np.array([aperiodic[4],
              aperiodic[5]]);

#Calculate response to arbitrary input
t, y, x = control.forced_response(system, aperiodic[0], u, aperiodic[6], transpose=False)

#Change dimensionless pb/(2V) and rb/(2V) to p and r
# y[1, :] = -y[1, :]
y[2, :] = 2*abs(V0)*y[2, :]/b
y[3, :] = 2*abs(V0)*y[3, :]/b

y[2, 0] = aperiodic[2][0]
y[3, 0] = aperiodic[3][0]

control.damp(system, doprint=True)

fig3 = plt.figure(figsize=(12,9))
fig3.suptitle('Aperiodic',fontsize=16)

ax1 = fig3.add_subplot(221)
ax1.plot(t, y[0, :])
ax1.plot(t, aperiodic[8], color='tab:orange',linestyle='--')
ax1.set_xlabel("Time [s]")
ax1.set_ylabel("beta (yaw angle) [deg]")
print("Aperiodic Roll Error in Beta:")
aperiodic_error_beta = L2error(aperiodic[8], y[0, :])
print(aperiodic_error_beta)

ax2 = fig3.add_subplot(222)
ax2.plot(t, y[1, :])
ax2.plot(t, aperiodic[1], color='tab:orange',linestyle='--')
ax2.set_xlabel("Time [s]")
ax2.set_ylabel("phi (roll angle) [deg]")
print("Aperiodic Roll Error in Phi:")
aperiodic_error_phi = L2error(aperiodic[1], y[1, :])
print(aperiodic_error_phi)

ax3 = fig3.add_subplot(223)
ax3.plot(t, y[2, :])
ax3.plot(t, aperiodic[2], color='tab:orange',linestyle='--')
ax3.set_xlabel("Time [s]")
ax3.set_ylabel("p (roll rate) [deg/sec]")
print("Aperiodic Roll Error in p:")
aperiodic_error_p = L2error(aperiodic[2], y[2, :])
print(aperiodic_error_p)

ax4 = fig3.add_subplot(224)
ax4.plot(t, y[3, :])
ax4.plot(t, aperiodic[3], color='tab:orange',linestyle='--')
ax4.set_xlabel("Time [s]")
ax4.set_ylabel("r (yaw rate) [deg/s]")
print("Aperiodic Roll Error in r:")
aperiodic_error_r = L2error(aperiodic[3], y[3, :])
print(aperiodic_error_r)

print("Avg. Aperiodic Error:")
avg_aperiodic_error = (aperiodic_error_beta + aperiodic_error_p + aperiodic_error_phi + aperiodic_error_r)/4.0
print(avg_aperiodic_error)

plt.show()

fig.savefig('Dutch_maneuver.png')
fig2.savefig('Spiral_maneuver.png')
fig3.savefig('Aperiodic_maneuver')