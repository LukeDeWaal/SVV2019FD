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

#Import reference values:

from data.Cit_par import *
from src.data_extraction.time_series_tool import TimeSeriesTool

#These reference values are missing from Cit_par:

Cma = -0.5626
Cmde = -1.1642

#Import data for given time step
ts_tool = TimeSeriesTool()
t = list(range(3060,3080))
da = []
dr = []
phi = []
p = []
r = []
for time in t:
    specific_t_mdat_vals = ts_tool.get_t_specific_mdat_values(time)
    da.append(specific_t_mdat_vals['delta_a'][0])
    dr.append(specific_t_mdat_vals['delta_r'][0])
    phi.append(specific_t_mdat_vals['Ahrs1_Pitch'][0])
    p.append(specific_t_mdat_vals['Ahrs1_bRollRate'][0])
    r.append(specific_t_mdat_vals['Ahrs1_bYawRate'][0])
    print("At t= {0} the corresponding recorded 'black-box' data is:\n {1}".format(time, specific_t_mdat_vals))
# print(ts_tool.get_t_specific_mdat_values(1665))
t = np.asarray(t)
phi = np.asarray(phi)
p = np.asarray(p)
r = np.asarray(r)

#State-space representation of asymmetric EOM:

#C1*x' + C2*x + C3*u = 0

C1 = np.array([[(CYbdot - 2.0*mub)*b/V0, 0.0, 0.0, 0.0], 
               [0.0, -0.5*b/V0, 0.0, 0.0], 
               [0.0, 0.0, -4.0*mub*KX2*b/V0, 4.0*mub*KXZ*b/V0],
               [Cnbdot*b/V0, 0.0, 4.0*mub*KXZ*b/V0, -4.0*mub*KZ2*b/V0]])

C2 = np.array([[CYb, CL, CYp, (CYr - 4.0*mub)], 
               [0.0, 0.0, 1.0, 0.0], 
               [Clb, 0.0, Clp, Clr], 
               [Cnb, 0.0, Cnp, Cnr] ])

C3 = np.array([[CYda, CYdr],
               [0.0, 0.0],
               [Clda, Cldr],
               [Cnda, Cndr]])

#x' = A*x + B*u
#y  = C*x + D*u

# now u = [da]
#         [dr]

A = -np.matmul(np.linalg.inv(C1), C2)
B = -np.matmul(np.linalg.inv(C1), C3)
C = np.identity(4) #y = x, meaning we output the state
D = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])

#Make control.ss state-space

system = control.ss(A, B, C, D)

x0 = np.array([[0.0], 
               [0.0], 
               [0.0], 
               [0.0]])

# t = np.linspace(0.0, 200.0, num = 201)

#Input:
u = np.zeros((2, t.shape[0]))

for i in range(t.shape[0]):
    u[0, i] = da[i] #Insert magnitude of "da" (aileron deflection) or comment out if none
    u[1, i] = dr[i] #Insert magnitude of "dr" (rudder deflection) or comment out if none

#Calculate response to arbitrary input    
t, y, x = control.forced_response(system, t, u, x0, transpose=False)

#Change dimensionless pb/(2V) and rb/(2V) to p and r
y[2, :] = 2*V0*y[2, :]/b
y[3, :] = 2*V0*y[3, :]/b

fig = plt.figure(figsize=(12,9))

ax1 = fig.add_subplot(221)
ax1.plot(t, y[0, :])
ax1.set_xlabel("Time [s]")
ax1.set_ylabel("beta (yaw angle) [deg]")
# q

ax2 = fig.add_subplot(222)
ax2.plot(t, y[1, :])
#alpha
ax2.plot(t, phi)
ax2.set_xlabel("Time [s]")
ax2.set_ylabel("phi (roll angle) [deg]")

ax3 = fig.add_subplot(223)
ax3.plot(t, y[3, :])
#theta
ax3.plot(t,p)
ax3.set_xlabel("Time [s]")
ax3.set_ylabel("p (roll rate) [deg/sec]")

ax4 = fig.add_subplot(224)
ax4.plot(t, y[3, :])
#q
ax4.plot(t, r)
ax4.set_xlabel("Time [s]")
ax4.set_ylabel("r (yaw rate) [deg/s]")

plt.show()



