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

dirname = os.path.dirname(os.path.realpath(__file__))
Cit_parStr = os.path.join(dirname, r'..\\..\\data')
Cit_parStr = os.path.abspath(os.path.realpath(Cit_parStr))

sys.path.append(Cit_parStr)

from Cit_par import *

#These reference values are missing from Cit_par:

Cma = -0.5626
Cmde = -1.1642

#State-space representation of symmetric EOM:

#C1*x' + C2*x + C3*u = 0

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

x0 = np.array([[0.0], 
               [alpha0], 
               [th0], 
               [0.0]])

t = np.linspace(0.0, 300.0, num = 200)

u = np.zeros(t.shape[0])

#length of pulse
tpulse = 1.0
i = 0
while (t[i] < tpulse):
    u[i] = 1.0 #Insert magnitude of "de" (elevator deflection)
    i += 1
    
#Calculate response to arbitrary input
t, y, x = control.forced_response(system, t, u, x0, transpose=False)

#Change dimensionless û and qc/V to u and q
y[0, :] = V0*y[0, :]
y[3, :] = V0*y[3, :]/c

fig = plt.figure()
ax1 = fig.add_subplot(221)
ax1.plot(t, y[0, :])
# ax1.xlabel("Time [s]")
# ax1.ylabel("u (disturbance in velocity) [m/s]")

ax2 = fig.add_subplot(222)
ax2.plot(t, y[1, :])
ax3 = fig.add_subplot(223)
ax3.plot(t, y[3, :])
ax4 = fig.add_subplot(224)
ax4.plot(t, y[3, :])

plt.show()



