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

dirname = os.path.dirname(os.path.realpath(__file__))
Cit_parStr = os.path.join(dirname, r'..\\..\\data')
Cit_parStr = os.path.abspath(os.path.realpath(Cit_parStr))

sys.path.append(Cit_parStr)

from Cit_par import *

#These reference values are missing from Cit_par:

Cma = -0.5626
Cmde = -1.1642

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

t = np.linspace(0.0, 200.0, num = 1200)

#Input:
u = np.zeros((2, t.shape[0]))

#length of pulse
tpulse = 1.0
i = 0
while (t[i] < tpulse):
    #u[0, i] = 1.0 #Insert magnitude of "da" (aileron deflection) or comment out if none
    u[1, i] = 1.0 #Insert magnitude of "dr" (rudder deflection) or comment out if none
    i += 1

#Calculate response to arbitrary input    
t, y, x = control.forced_response(system, t, u, x0, transpose=False)

#Change dimensionless pb/(2V) and rb/(2V) to p and r
y[2, :] = 2*V0*y[2, :]/b
y[3, :] = 2*V0*y[3, :]/b

fig = plt.figure()
ax1 = fig.add_subplot(221)
ax1.plot(t, y[0, :])
# ax1.xlabel("Time [s]")
# ax1.ylabel("u (disturbance in velocity) [m/s]")
q
ax2 = fig.add_subplot(222)
ax2.plot(t, y[1, :])
ax3 = fig.add_subplot(223)
ax3.plot(t, y[3, :])
ax4 = fig.add_subplot(224)
ax4.plot(t, y[3, :])

plt.show()



